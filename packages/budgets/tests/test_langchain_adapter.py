import asyncio
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool

from agent_toolkit.budgets import BudgetLimits, Pricing
from agent_toolkit.budgets.adapters.langchain import BudgetCallbackHandler, BudgetExceededError


def _reply(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _config(handler: BudgetCallbackHandler) -> RunnableConfig:
    return RunnableConfig(callbacks=[handler])


def _model(*messages: AIMessage) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(messages))


class CountingModel(GenericFakeChatModel):
    calls: int = 0

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        return super()._generate(*args, **kwargs)


def test_tokens_and_iterations_accumulate_across_calls() -> None:
    pricing = Pricing(input_per_mtok_usd=Decimal(3), output_per_mtok_usd=Decimal(15))
    handler = BudgetCallbackHandler(BudgetLimits(), pricing)
    model = _model(_reply(100, 10), _reply(200, 20))
    model.invoke("a", config=_config(handler))
    model.invoke("b", config=_config(handler))
    assert handler.state.iterations == 2
    assert (handler.state.input_tokens, handler.state.output_tokens) == (300, 30)
    assert handler.state.cost_usd == Decimal("0.00135")


def test_response_without_usage_metadata_counts_only_the_iteration() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(), None)
    _model(AIMessage(content="hi")).invoke("a", config=_config(handler))
    assert handler.state.iterations == 1
    assert handler.state.input_tokens == handler.state.output_tokens == 0


def test_third_call_over_max_tokens_raises() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_tokens=250), None)
    model = _model(_reply(100, 10), _reply(100, 10), _reply(100, 10))
    config = _config(handler)
    model.invoke("a", config=config)
    model.invoke("b", config=config)
    with pytest.raises(BudgetExceededError) as raised:
        model.invoke("c", config=config)
    assert raised.value.kinds == ("tokens",)


def test_max_iterations_raises_before_the_model_is_invoked() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_iterations=1), None)
    model = CountingModel(messages=iter([_reply(1, 1), _reply(1, 1)]))
    model.invoke("a", config=_config(handler))
    with pytest.raises(BudgetExceededError, match="iterations"):
        model.invoke("b", config=_config(handler))
    assert model.calls == 1
    assert handler.state.iterations == 2


def test_handler_is_sticky_after_the_first_raise_is_swallowed() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_tool_calls=0), None)
    config = _config(handler)
    with pytest.raises(BudgetExceededError):
        ok_tool.invoke({}, config=config)
    model = CountingModel(messages=iter([_reply(1, 1)]))
    with pytest.raises(BudgetExceededError):
        model.invoke("a", config=config)
    assert model.calls == 0
    with pytest.raises(BudgetExceededError):
        ok_tool.invoke({}, config=config)


def test_sticky_even_after_consecutive_failures_would_reset() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_consecutive_failures=0), None)
    config = _config(handler)
    with pytest.raises(BudgetExceededError):
        failing_tool.invoke({}, config=config)
    with pytest.raises(BudgetExceededError):
        ok_tool.invoke({}, config=config)


def test_tool_error_is_chained_as_the_cause() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_consecutive_failures=0), None)
    with pytest.raises(BudgetExceededError) as raised:
        failing_tool.invoke({}, config=_config(handler))
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_handled_tool_errors_count_as_failures() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_consecutive_failures=1), None)
    config = _config(handler)
    call = {"name": "handled_tool", "args": {}, "id": "call-1", "type": "tool_call"}
    assert handled_tool.invoke(call, config=config).status == "error"
    assert handler.state.consecutive_failures == 1
    with pytest.raises(BudgetExceededError) as raised:
        handled_tool.invoke(call, config=config)
    assert raised.value.kinds == ("consecutive_failures",)


def test_cost_limit_without_pricing_is_rejected() -> None:
    with pytest.raises(ValueError, match="pricing"):
        BudgetCallbackHandler(BudgetLimits(max_cost_usd=Decimal(1)), None)


def test_ainvoke_over_max_tokens_raises() -> None:
    asyncio.run(_ainvoke_over_max_tokens())


async def _ainvoke_over_max_tokens() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_tokens=15), None)
    model = _model(_reply(10, 10))
    with pytest.raises(BudgetExceededError) as raised:
        await model.ainvoke("a", config=_config(handler))
    assert raised.value.kinds == ("tokens",)


def test_tool_ainvoke_over_max_tool_calls_raises() -> None:
    asyncio.run(_tool_ainvoke_over_max_tool_calls())


async def _tool_ainvoke_over_max_tool_calls() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_tool_calls=1), None)
    config = _config(handler)
    await ok_tool.ainvoke({}, config=config)
    with pytest.raises(BudgetExceededError) as raised:
        await ok_tool.ainvoke({}, config=config)
    assert raised.value.kinds == ("tool_calls",)


@tool
def ok_tool() -> str:
    """Succeeds."""
    return "done"


@tool
def failing_tool() -> str:
    """Always fails."""
    raise RuntimeError("boom")


@tool
def handled_tool() -> str:
    """Raises a ToolException that LangChain turns into an error message."""
    raise ToolException("handled")


handled_tool.handle_tool_error = True


def test_tool_calls_are_counted_and_success_resets_failures() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(), None)
    config = _config(handler)
    with pytest.raises(RuntimeError, match="boom"):
        failing_tool.invoke({}, config=config)
    assert handler.state.consecutive_failures == 1
    ok_tool.invoke({}, config=config)
    assert handler.state.consecutive_failures == 0
    assert handler.state.tool_calls == 2


def test_consecutive_tool_failures_raise_budget_error() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_consecutive_failures=1), None)
    config = _config(handler)
    with pytest.raises(RuntimeError, match="boom"):
        failing_tool.invoke({}, config=config)
    with pytest.raises(BudgetExceededError) as raised:
        failing_tool.invoke({}, config=config)
    assert raised.value.kinds == ("consecutive_failures",)


def test_max_tool_calls_stops_the_tool_before_it_runs() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_tool_calls=1), None)
    config = _config(handler)
    ok_tool.invoke({}, config=config)
    with pytest.raises(BudgetExceededError) as raised:
        ok_tool.invoke({}, config=config)
    assert raised.value.kinds == ("tool_calls",)

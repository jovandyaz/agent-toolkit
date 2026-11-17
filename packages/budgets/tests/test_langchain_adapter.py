from decimal import Decimal

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

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


def test_max_iterations_raises_on_the_call_that_exceeds() -> None:
    handler = BudgetCallbackHandler(BudgetLimits(max_iterations=1), None)
    model = _model(_reply(1, 1), _reply(1, 1))
    model.invoke("a", config=_config(handler))
    with pytest.raises(BudgetExceededError, match="iterations"):
        model.invoke("b", config=_config(handler))


def test_handler_propagates_errors() -> None:
    assert BudgetCallbackHandler(BudgetLimits(), None).raise_error is True


@tool
def ok_tool() -> str:
    """Succeeds."""
    return "done"


@tool
def failing_tool() -> str:
    """Always fails."""
    raise RuntimeError("boom")


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

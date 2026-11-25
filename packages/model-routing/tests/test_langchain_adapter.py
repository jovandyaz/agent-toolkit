import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import ConfigurableField, Runnable

from agent_toolkit.model_routing import ProviderCooldown
from agent_toolkit.model_routing.adapters.langchain import (
    CooldownFallbacks,
    CooldownRunnable,
    ProviderCoolingDown,
    langchain_is_retryable,
    with_cooldown_fallbacks,
)

A, B = "anthropic:a", "openrouter:b"
CHAIN = (A, B)


class BudgetStop(Exception):
    """Stands in for agent_toolkit.budgets.BudgetExceededError."""


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedModel(GenericFakeChatModel):
    """Answers or raises from `script`, in order, and counts calls."""

    script: list[Any]  # Any: pydantic would coerce an exception into an AIMessage.
    calls: int = 0
    messages: Any = iter(())

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        self.messages = iter([step])
        return super()._generate(*args, **kwargs)


def model(*script: AIMessage | BaseException) -> ScriptedModel:
    return ScriptedModel(script=list(script))


def reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def tracker(clock: FakeClock | None = None) -> ProviderCooldown:
    classify = partial(langchain_is_retryable, never=(BudgetStop,))
    return ProviderCooldown(clock=clock or FakeClock(), classify=classify)


def open_it(cooldown: ProviderCooldown, key: str) -> None:
    for _ in range(3):
        assert cooldown.admit(key, CHAIN)
        cooldown.record_failure(key)


def chain_of(
    cooldown: ProviderCooldown, primary: ScriptedModel, fallback: ScriptedModel
) -> Runnable[LanguageModelInput, BaseMessage]:
    return with_cooldown_fallbacks([(A, primary), (B, fallback)], cooldown)


# --- langchain_is_retryable ----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ModelTimeoutError("t"),
        ModelRateLimitError("r"),
        ModelAPIError("5"),
        ModelConnectionError("c"),
    ],
    ids=["timeout", "rate-limit", "api", "connection"],
)
def test_retryable_model_errors(error: BaseException) -> None:
    assert langchain_is_retryable(error)


def test_a_non_retryable_model_error_decides_over_its_cause() -> None:
    class StatusError(Exception):
        status_code = 500

    overflow = ModelInvalidRequestError("exceeds the context window")
    overflow.__cause__ = StatusError()
    assert not langchain_is_retryable(overflow)


def test_never_and_retryable_pass_through() -> None:
    late = ModelTimeoutError("late")
    late.__cause__ = BudgetStop()
    assert not langchain_is_retryable(late, never=(BudgetStop,))
    assert langchain_is_retryable(KeyError("x"), retryable=(KeyError,))


# --- through with_cooldown_fallbacks -------------------------------------------------------------


def test_an_open_primary_is_not_called_and_the_fallback_answers() -> None:
    cooldown = tracker()
    open_it(cooldown, A)
    primary, fallback = model(), model(reply("from b"))
    assert chain_of(cooldown, primary, fallback).invoke("hola").content == "from b"
    assert (primary.calls, fallback.calls) == (0, 1)


def test_a_primary_that_keeps_failing_opens_through_the_chain() -> None:
    cooldown = tracker()
    primary = model(*[ModelAPIError("5xx") for _ in range(3)])
    fallback = model(*[reply("b") for _ in range(4)])
    runnable = chain_of(cooldown, primary, fallback)
    for _ in range(4):
        assert runnable.invoke("hola").content == "b"
    assert (primary.calls, fallback.calls) == (3, 4)


@pytest.mark.parametrize(
    "failure",
    [ModelInvalidRequestError("bad"), ValueError("bug"), BudgetStop()],
    ids=["invalid-request", "value-error", "budget"],
)
def test_request_programming_and_budget_errors_never_open_the_primary(
    failure: Exception,
) -> None:
    cooldown = tracker()
    primary = model(*[failure for _ in range(5)])
    fallback = model(*[reply("b") for _ in range(5)])
    runnable = chain_of(cooldown, primary, fallback)
    for _ in range(5):
        runnable.invoke("hola")
    assert primary.calls == 5


def test_the_half_open_trial_goes_through_the_chain_and_closes_on_success() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    clock.advance(61)
    primary, fallback = model(reply("a back")), model()
    assert chain_of(cooldown, primary, fallback).invoke("hola").content == "a back"
    assert cooldown.admit(A, CHAIN) == "call"


def test_every_entry_open_still_makes_a_call() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    clock.advance(10)
    open_it(cooldown, B)
    primary, fallback = model(reply("a, soonest")), model()
    assert chain_of(cooldown, primary, fallback).invoke("hola").content == "a, soonest"
    assert fallback.calls == 0


def test_every_entry_open_forces_the_second_when_it_reopens_first() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    open_it(cooldown, B)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    clock.advance(1)
    cooldown.record_failure(A, trial)  # A now reopens last.
    primary, fallback = model(), model(reply("b, soonest"))
    assert chain_of(cooldown, primary, fallback).invoke("hola").content == "b, soonest"
    assert primary.calls == 0


def test_when_every_call_is_skipped_the_first_error_is_a_cooling_down_one() -> None:
    cooldown = tracker()
    open_it(cooldown, A)
    open_it(cooldown, B)
    cooldown.admit(A, CHAIN)  # A's forced trial is in flight: nothing can be admitted.
    with pytest.raises(ProviderCoolingDown) as raised:
        chain_of(cooldown, model(), model()).invoke("hola")
    assert raised.value.key == A


def test_ainvoke_goes_through_the_breaker() -> None:
    cooldown = tracker()
    primary = model(*[ModelRateLimitError("429") for _ in range(3)])
    fallback = model(*[reply("b") for _ in range(4)])
    runnable = chain_of(cooldown, primary, fallback)

    async def scenario() -> list[str]:
        return [str((await runnable.ainvoke("hola")).content) for _ in range(4)]

    assert asyncio.run(scenario()) == ["b"] * 4
    assert primary.calls == 3


def test_a_single_entry_still_goes_through_the_breaker() -> None:
    cooldown = tracker()
    runnable = with_cooldown_fallbacks([(A, model(reply("only")))], cooldown)
    assert [entry.key for entry in runnable.entries] == [A]
    assert runnable.invoke("hola").content == "only"


@pytest.mark.parametrize(
    ("make_entries", "message"),
    [
        (lambda: [], "at least one"),
        (lambda: [(A, model()), (A, model())], "unique"),
    ],
    ids=["empty", "duplicate"],
)
def test_rejects_empty_or_duplicate_keys(
    make_entries: Callable[[], list[tuple[str, ScriptedModel]]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        with_cooldown_fallbacks(make_entries(), tracker())


# --- review fixes --------------------------------------------------------------------------


def test_a_skip_never_hides_the_fallbacks_real_error() -> None:
    cooldown = tracker()
    open_it(cooldown, A)
    with pytest.raises(BudgetStop):
        chain_of(cooldown, model(), model(BudgetStop())).invoke("hola")


def test_a_skip_never_hides_the_fallbacks_real_error_async() -> None:
    cooldown = tracker()
    open_it(cooldown, A)
    runnable = chain_of(cooldown, model(), model(ModelInvalidRequestError("too long")))
    with pytest.raises(ModelInvalidRequestError):
        asyncio.run(runnable.ainvoke("hola"))


def test_when_both_fail_the_first_real_error_is_raised() -> None:
    cooldown = tracker()
    runnable = chain_of(cooldown, model(ModelAPIError("a down")), model(ModelAPIError("b down")))
    with pytest.raises(ModelAPIError, match="a down"):
        runnable.invoke("hola")


def test_an_error_outside_exceptions_to_handle_stops_the_chain() -> None:
    cooldown = tracker()
    primary, fallback = model(BudgetStop()), model(reply("b"))
    runnable = with_cooldown_fallbacks(
        [(A, primary), (B, fallback)], cooldown, exceptions_to_handle=(ModelError,)
    )
    with pytest.raises(BudgetStop):
        runnable.invoke("hola")
    assert fallback.calls == 0


def test_a_skip_moves_on_even_with_narrow_exceptions_to_handle() -> None:
    cooldown = tracker()
    open_it(cooldown, A)
    runnable = with_cooldown_fallbacks(
        [(A, model()), (B, model(reply("b")))], cooldown, exceptions_to_handle=(ModelError,)
    )
    assert runnable.invoke("hola").content == "b"


def test_a_never_error_wrapped_in_a_timeout_does_not_open_the_entry() -> None:
    def late_budget_stop() -> TimeoutError:
        late = TimeoutError("outlived its deadline")
        late.__cause__ = BudgetStop()
        return late

    cooldown = tracker()
    primary = model(*[late_budget_stop() for _ in range(4)])
    fallback = model(*[reply("b") for _ in range(4)])
    runnable = chain_of(cooldown, primary, fallback)
    for _ in range(4):
        runnable.invoke("hola")
    assert primary.calls == 4


def test_a_wrapped_model_error_decides() -> None:
    def wrapped(cause: BaseException) -> Exception:
        try:
            raise RuntimeError("wrapper") from cause
        except RuntimeError as outer:
            return outer

    assert langchain_is_retryable(wrapped(ModelRateLimitError("r")))
    assert not langchain_is_retryable(wrapped(ModelInvalidRequestError("bad")))


def test_schemas_and_config_specs_are_the_wrapped_models() -> None:
    inner = model(reply("x"))
    runnable = with_cooldown_fallbacks([(A, inner)], tracker())
    assert runnable.InputType == inner.InputType
    assert runnable.OutputType == inner.OutputType
    assert runnable.config_specs == inner.config_specs


def test_the_chain_reports_one_run_with_each_attempt_as_a_child() -> None:
    class Runs(BaseCallbackHandler):
        def __init__(self) -> None:
            self.starts: list[tuple[UUID, UUID | None]] = []

        def on_chain_start(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.starts.append((run_id, parent_run_id))

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            self.starts.append((run_id, parent_run_id))

    runs = Runs()
    cooldown = tracker()
    runnable = chain_of(cooldown, model(ModelAPIError("down")), model(reply("b")))
    runnable.invoke("hola", config={"callbacks": [runs]})
    (root, no_parent), *children = runs.starts
    assert no_parent is None
    assert len(children) == 2
    assert all(parent == root for _, parent in children)


def test_the_chain_reports_the_config_specs_of_every_entry() -> None:
    fallback = model(reply("b")).configurable_fields(
        messages=ConfigurableField(id="fallback_messages")
    )
    runnable = with_cooldown_fallbacks([(A, model()), (B, fallback)], tracker())
    assert [spec.id for spec in runnable.config_specs] == ["fallback_messages"]


def test_an_error_outside_exceptions_to_handle_stops_the_chain_async() -> None:
    cooldown = tracker()
    fallback = model(reply("b"))
    runnable = with_cooldown_fallbacks(
        [(A, model(BudgetStop())), (B, fallback)], cooldown, exceptions_to_handle=(ModelError,)
    )
    with pytest.raises(BudgetStop):
        asyncio.run(runnable.ainvoke("hola"))
    assert fallback.calls == 0


def test_cooldown_fallbacks_needs_an_entry() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CooldownFallbacks([])


def test_a_raw_with_fallbacks_still_skips_but_reports_the_skip() -> None:
    """The limit the module docstring warns about: prefer with_cooldown_fallbacks."""
    cooldown = tracker()
    open_it(cooldown, A)
    primary = CooldownRunnable(model(), A, CHAIN, cooldown)
    fallback = CooldownRunnable(model(BudgetStop()), B, CHAIN, cooldown)
    with pytest.raises(ProviderCoolingDown):
        primary.with_fallbacks([fallback]).invoke("hola")

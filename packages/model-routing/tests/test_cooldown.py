import asyncio
import logging
import threading

import pytest

from agent_toolkit.model_routing import (
    COOLDOWN_S,
    FAILURES_TO_OPEN,
    Admission,
    ProviderCooldown,
    ProviderCoolingDown,
    Transition,
)

A, B, C = "anthropic:a", "openrouter:b", "openai:c"
CHAIN = (A, B)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Transitions:
    def __init__(self) -> None:
        self.seen: list[tuple[str, Transition]] = []

    def __call__(self, key: str, state: Transition) -> None:
        self.seen.append((key, state))


def tracker(clock: FakeClock, on_transition: Transitions | None = None) -> ProviderCooldown:
    return ProviderCooldown(
        failures_to_open=3, cooldown_s=60.0, clock=clock, on_transition=on_transition
    )


def open_it(cooldown: ProviderCooldown, key: str = A) -> None:
    for _ in range(3):
        assert cooldown.admit(key, CHAIN)
        cooldown.record_failure(key)


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider said no")
        self.status_code = status_code


# --- configuration -------------------------------------------------------------------------


def test_defaults_match_the_router_readme() -> None:
    assert (FAILURES_TO_OPEN, COOLDOWN_S) == (3, 60.0)


@pytest.mark.parametrize(
    ("failures", "cooldown_s"),
    [(0, 60.0), (3, 0.0), (3, -1.0), (3, float("nan")), (3, float("inf"))],
)
def test_rejects_a_threshold_or_window_that_cannot_work(failures: int, cooldown_s: float) -> None:
    with pytest.raises(ValueError):
        ProviderCooldown(failures_to_open=failures, cooldown_s=cooldown_s)


# --- state machine -------------------------------------------------------------------------


def test_opens_only_after_the_configured_consecutive_failures() -> None:
    cooldown = tracker(FakeClock())
    for _ in range(2):
        cooldown.record_failure(A)
        assert cooldown.admit(A, CHAIN)
    cooldown.record_failure(A)
    assert not cooldown.admit(A, CHAIN)


def test_a_success_between_failures_resets_the_count() -> None:
    cooldown = tracker(FakeClock())
    cooldown.record_failure(A)
    cooldown.record_failure(A)
    cooldown.record_success(A)
    cooldown.record_failure(A)
    cooldown.record_failure(A)
    assert cooldown.admit(A, CHAIN)


def test_other_outcomes_neither_count_nor_reset() -> None:
    cooldown = tracker(FakeClock())
    cooldown.record_failure(A)
    cooldown.record_failure(A)
    cooldown.record_other(A)
    cooldown.record_failure(A)
    assert not cooldown.admit(A, CHAIN)


def test_keys_are_independent() -> None:
    cooldown = tracker(FakeClock())
    open_it(cooldown, A)
    assert not cooldown.admit(A, CHAIN)
    assert cooldown.admit(B, CHAIN)


def test_open_skips_until_the_window_ends_then_admits_exactly_one_trial() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(59.9)
    assert not cooldown.admit(A, CHAIN)
    clock.advance(0.2)
    assert cooldown.admit(A, CHAIN)  # The trial.
    assert not cooldown.admit(A, CHAIN)  # One in flight: B is available, so A waits.


def test_a_trial_success_closes_it() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_success(A, trial)
    assert cooldown.admit(A, CHAIN) == "call"
    cooldown.record_failure(A)  # Back to counting from zero.
    cooldown.record_failure(A)
    assert cooldown.admit(A, CHAIN)


def test_a_trial_failure_reopens_for_another_window() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_failure(A, trial)
    assert not cooldown.admit(A, CHAIN)
    clock.advance(59)
    assert not cooldown.admit(A, CHAIN)
    clock.advance(2)
    assert cooldown.admit(A, CHAIN)


def test_a_trial_ended_by_a_non_retryable_outcome_frees_the_next_trial() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_other(A, trial)
    assert cooldown.admit(A, CHAIN)  # Still open and expired: another trial, not stuck.


def test_when_every_entry_is_cooling_the_soonest_to_reopen_is_tried() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    clock.advance(10)
    open_it(cooldown, B)  # Reopens 10 s after A.
    assert cooldown.admit(A, CHAIN)
    assert not cooldown.admit(B, CHAIN)


def test_all_cooling_tie_goes_to_the_earliest_in_the_chain() -> None:
    cooldown = tracker(FakeClock())
    open_it(cooldown, A)
    open_it(cooldown, B)
    assert not cooldown.admit(B, CHAIN)
    assert cooldown.admit(A, CHAIN)


def test_a_forced_trial_that_fails_reopens_and_hands_over() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    clock.advance(10)
    open_it(cooldown, B)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    clock.advance(1)
    cooldown.record_failure(A, trial)  # A now reopens last: B is the soonest.
    assert cooldown.admit(B, CHAIN)
    assert not cooldown.admit(A, CHAIN)


def test_a_trial_in_flight_keeps_others_from_being_forced() -> None:
    cooldown = tracker(FakeClock())
    open_it(cooldown, A)
    open_it(cooldown, B)
    assert cooldown.admit(A, CHAIN)  # Forced trial in flight.
    assert not cooldown.admit(A, CHAIN)
    assert not cooldown.admit(B, CHAIN)


def test_a_key_outside_the_chain_view_is_still_tracked() -> None:
    cooldown = tracker(FakeClock())
    open_it(cooldown, C)
    assert not cooldown.admit(C, (C, A))


def test_a_call_admitted_before_the_entry_opened_leaves_its_trial_alone() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    assert cooldown.admit(A, CHAIN) == "call"  # Still in flight while the entry opens.
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_other(A)
    cooldown.record_failure(A)
    assert cooldown.admit(A, CHAIN) is None  # The trial is still the only one in flight.
    cooldown.record_success(A, trial)
    assert cooldown.admit(A, CHAIN) == "call"


def test_a_stale_failure_does_not_extend_the_window() -> None:
    clock = FakeClock()
    seen = Transitions()
    cooldown = tracker(clock, seen)
    open_it(cooldown)
    clock.advance(59)
    cooldown.record_failure(A)
    clock.advance(2)
    assert cooldown.admit(A, CHAIN) == "trial"
    assert seen.seen == [(A, "open"), (A, "half_open")]


def test_concurrent_admits_let_exactly_one_trial_through() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    results: list[Admission | None] = []
    start = threading.Barrier(16)

    def admit() -> None:
        start.wait()
        results.append(cooldown.admit(A, CHAIN))

    threads = [threading.Thread(target=admit) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("trial") == 1
    assert results.count(None) == 15


# --- transitions ---------------------------------------------------------------------------


def test_reports_each_transition_with_the_key_and_state() -> None:
    clock = FakeClock()
    seen = Transitions()
    cooldown = tracker(clock, seen)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_success(A, trial)
    assert seen.seen == [(A, "open"), (A, "half_open"), (A, "closed")]


def test_a_failed_trial_reports_open_again() -> None:
    clock = FakeClock()
    seen = Transitions()
    cooldown = tracker(clock, seen)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_failure(A, trial)
    assert [state for _, state in seen.seen] == ["open", "half_open", "open"]


def test_a_forced_trial_reports_half_open() -> None:
    seen = Transitions()
    cooldown = tracker(FakeClock(), seen)
    open_it(cooldown, A)
    open_it(cooldown, B)
    assert cooldown.admit(A, CHAIN) == "trial"
    assert [state for _, state in seen.seen] == ["open", "open", "half_open"]


def test_a_closed_key_reports_nothing() -> None:
    seen = Transitions()
    cooldown = tracker(FakeClock(), seen)
    assert cooldown.admit(A, CHAIN) == "call"
    cooldown.record_failure(A)
    cooldown.record_success(A)
    assert seen.seen == []


def test_without_a_hook_transitions_go_to_the_package_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agent_toolkit.model_routing")
    clock = FakeClock()
    cooldown = ProviderCooldown(clock=clock)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    cooldown.record_success(A, trial)
    records = [r for r in caplog.records if r.name == "agent_toolkit.model_routing"]
    assert [(r.levelno, r.provider, r.state) for r in records] == [  # type: ignore[attr-defined]
        (logging.WARNING, A, "open"),
        (logging.INFO, A, "half_open"),
        (logging.INFO, A, "closed"),
    ]


# --- record_error and call -----------------------------------------------------------------


def test_record_error_counts_only_what_the_classifier_calls_retryable() -> None:
    cooldown = tracker(FakeClock())
    for _ in range(5):
        cooldown.record_error(A, ValueError("bug"))
    assert cooldown.admit(A, CHAIN) == "call"
    for _ in range(3):
        cooldown.record_error(A, TimeoutError("slow"))
    assert cooldown.admit(A, CHAIN) is None


def test_a_custom_classifier_decides() -> None:
    cooldown = ProviderCooldown(clock=FakeClock(), classify=lambda e: isinstance(e, ValueError))
    for _ in range(3):
        cooldown.record_error(A, ValueError("counts here"))
    assert cooldown.admit(A, CHAIN) is None


def test_a_classifier_that_raises_still_ends_the_trial() -> None:
    def broken(error: BaseException) -> bool:
        raise RuntimeError("classifier bug")

    clock = FakeClock()
    cooldown = ProviderCooldown(clock=clock, classify=broken)
    open_it(cooldown)
    clock.advance(61)
    trial = cooldown.admit(A, CHAIN)
    assert trial == "trial"
    with pytest.raises(RuntimeError, match="classifier bug"):
        cooldown.record_error(A, TimeoutError(), trial)
    assert cooldown.admit(A, CHAIN) == "trial"  # Not stuck.


def test_call_runs_and_records_success() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    assert cooldown.call(A, CHAIN, lambda: "ok") == "ok"
    assert cooldown.admit(A, CHAIN) == "call"


def test_call_records_a_retryable_failure_and_reraises() -> None:
    cooldown = tracker(FakeClock())

    def failing() -> str:
        raise StatusError(503)

    for _ in range(3):
        with pytest.raises(StatusError):
            cooldown.call(A, CHAIN, failing)
    with pytest.raises(ProviderCoolingDown) as raised:
        cooldown.call(A, CHAIN, failing)
    assert raised.value.key == A
    assert str(raised.value) == A


def test_call_skips_without_calling_when_cooling() -> None:
    cooldown = tracker(FakeClock())
    open_it(cooldown)
    calls: list[int] = []
    with pytest.raises(ProviderCoolingDown):
        cooldown.call(A, CHAIN, lambda: calls.append(1))
    assert calls == []


def test_acall_records_like_call() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)

    async def failing() -> str:
        raise TimeoutError("slow")

    async def ok() -> str:
        return "ok"

    async def scenario() -> None:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await cooldown.acall(A, CHAIN, failing)
        with pytest.raises(ProviderCoolingDown):
            await cooldown.acall(A, CHAIN, ok)
        clock.advance(61)
        assert await cooldown.acall(A, CHAIN, ok) == "ok"

    asyncio.run(scenario())
    assert cooldown.admit(A, CHAIN) == "call"


def test_a_cancelled_acall_ends_its_trial_without_counting() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)

    async def cancelled() -> str:
        raise asyncio.CancelledError

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await cooldown.acall(A, CHAIN, cancelled)

    asyncio.run(scenario())
    assert cooldown.admit(A, CHAIN) == "trial"  # Not stuck, not reopened.


# --- review fixes --------------------------------------------------------------------------


def test_a_hook_that_raises_never_fails_the_call_nor_wedges_the_trial(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(key: str, state: Transition) -> None:
        raise RuntimeError("metrics down")

    clock = FakeClock()
    cooldown = ProviderCooldown(clock=clock, on_transition=broken)
    open_it(cooldown)
    clock.advance(61)
    assert cooldown.call(A, CHAIN, lambda: "ok") == "ok"  # half_open and closed hooks raised
    assert cooldown.admit(A, CHAIN) == "call"
    assert "transition hook failed" in caplog.text


def test_an_abandoned_trial_is_replaced_after_the_window() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown)
    clock.advance(61)
    assert cooldown.admit(A, CHAIN) == "trial"  # Its outcome never comes back.
    clock.advance(59)
    assert cooldown.admit(A, CHAIN) is None
    clock.advance(1)
    assert cooldown.admit(A, CHAIN) == "trial"


def test_an_abandoned_forced_trial_is_replaced_after_the_window() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, A)
    open_it(cooldown, B)
    assert cooldown.admit(A, CHAIN) == "trial"
    assert cooldown.admit(A, CHAIN) is None
    clock.advance(60)
    assert cooldown.admit(A, CHAIN) == "trial"


def test_a_stale_success_closes_the_key() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    assert cooldown.admit(A, CHAIN) == "call"  # Slow, still in flight while the key opens.
    open_it(cooldown)
    cooldown.record_success(A)  # The provider answered: it is up.
    assert cooldown.admit(A, CHAIN) == "call"


def test_a_key_outside_the_chain_can_get_the_forced_trial() -> None:
    clock = FakeClock()
    cooldown = tracker(clock)
    open_it(cooldown, C)
    clock.advance(1)
    open_it(cooldown, A)
    open_it(cooldown, B)
    assert cooldown.admit(C, CHAIN) == "trial"  # C reopens soonest, though CHAIN omits it.


@pytest.mark.parametrize(
    "late", [ValueError("says nothing"), TimeoutError("late timeout")], ids=["other", "failure"]
)
def test_a_late_outcome_of_an_abandoned_trial_leaves_its_replacement_alone(
    late: Exception,
) -> None:
    clock = FakeClock()
    seen = Transitions()
    cooldown = tracker(clock, seen)
    open_it(cooldown)
    clock.advance(61)

    def hung() -> str:
        clock.advance(60)  # The lease runs out while this trial hangs...
        assert cooldown.admit(A, CHAIN) == "trial"  # ...and a replacement is admitted.
        raise late

    with pytest.raises(type(late)):
        cooldown.call(A, CHAIN, hung)
    assert cooldown.admit(A, CHAIN) is None  # The replacement is still the one in flight.
    assert [state for _, state in seen.seen] == ["open", "half_open", "half_open"]

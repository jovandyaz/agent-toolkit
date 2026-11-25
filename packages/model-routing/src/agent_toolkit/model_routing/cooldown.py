"""A circuit breaker per model provider: a provider that keeps failing is skipped for a while.

A fallback chain (LangChain `with_fallbacks`, a loop over SDK clients) has no memory, so without
this every request retries a dead primary and can wait out its whole timeout before moving on.
`ProviderCooldown` counts consecutive retryable failures per key (a provider or `provider:model`
id). At `failures_to_open` the key opens and `admit` refuses it for `cooldown_s`; then it is
half-open: exactly one trial call is admitted, and its outcome closes the key or reopens it for
another window. When every key of the chain is cooling, the one that reopens soonest is tried
anyway, one trial at a time: a request that comes while that trial runs is refused by every key
(load shedding: a provider that is down is not sent a stampede).

Only failures a retry or another provider can cure count (`is_retryable`, or the `classify`
hook): timeouts, connection errors, 408, 409, 429 and 5xx, the statuses the OpenAI and Anthropic
SDKs retry. A 4xx, a budget stop or a bug never opens a breaker.

Sources: Microsoft Azure Architecture Center, "Circuit Breaker pattern" (closed, open and
half-open, one trial call); Martin Fowler, "CircuitBreaker". Ported from llm-router's
`provider-cooldown.tracker.ts` (defaults `{ allowedFails: 3, cooldownSeconds: 60 }`) through
pialy, which added the half-open trial, the all-cooling fallback of `resolveChainCandidates` and
two review fixes: a stale failure from a call admitted before the key opened neither extends the
window nor ends a trial it was not.
"""

import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

FAILURES_TO_OPEN: Final = 3
COOLDOWN_S: Final = 60.0
RETRYABLE_STATUSES: Final = frozenset({408, 409, 429})

Admission = Literal["call", "trial"]
"""What `admit` let through: a call to a closed key, or an open key's one trial."""

Transition = Literal["open", "half_open", "closed"]
"""A breaker state change, as reported to `on_transition`."""

Classifier = Callable[[BaseException], bool]
"""Whether an error is the provider's fault (counts toward opening) rather than the request's."""

Decider = Callable[[BaseException], bool | None]
"""Decides one link of an error chain for `is_retryable`; `None` leaves it undecided."""

TransitionHook = Callable[[str, Transition], None]
"""Called with the key and its new state, outside the tracker's lock, so two threads' events can
arrive out of order. An exception it raises is logged and dropped: a hook never fails a call or
wedges a trial."""

_logger = logging.getLogger("agent_toolkit.model_routing")


class ProviderCoolingDown(Exception):
    """A key was skipped because its breaker is open; the message is the key."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _links(error: BaseException) -> list[BaseException]:
    """`error` and everything reachable through `__cause__` and `__context__`, breadth first."""
    seen: set[int] = set()
    links: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending:
        link = pending.pop(0)
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        links.append(link)
        pending.extend((link.__cause__, link.__context__))
    return links


def _status_is_retryable(error: BaseException) -> bool:
    try:
        status = getattr(error, "status_code", None)
    except Exception:  # A property that raises says nothing about the provider.
        return False
    if not isinstance(status, int) or isinstance(status, bool):
        return False
    return status in RETRYABLE_STATUSES or 500 <= status <= 599


def is_retryable(
    error: BaseException,
    *,
    retryable: tuple[type[BaseException], ...] = (),
    never: tuple[type[BaseException], ...] = (),
    decide: Decider | None = None,
) -> bool:
    """Whether `error`, or anything in its `__cause__`/`__context__` chain, is a failure of the
    provider rather than of the request.

    An instance of a `never` type anywhere in the chain makes it not retryable (pass your budget
    stop here: one that came after a deadline arrives wrapped in a timeout). Otherwise the links
    are walked in order: the first one `decide` answers for decides; a link that is a
    `TimeoutError`, a `ConnectionError`, one of `retryable` (an HTTP client's transport error)
    or has an `int` `status_code` of 408, 409, 429 or 5xx makes it retryable."""
    links = _links(error)
    if never and any(isinstance(link, never) for link in links):
        return False
    for link in links:
        if decide is not None:
            verdict = decide(link)
            if verdict is not None:
                return verdict
        if isinstance(link, TimeoutError | ConnectionError) or (
            retryable and isinstance(link, retryable)
        ):
            return True
        if _status_is_retryable(link):
            return True
    return False


@dataclass(slots=True)
class _State:
    failures: int = 0
    open_until: float | None = None
    trial_since: float | None = None  # when the trial in flight was admitted
    trial_token: int = 0  # which trial that is: a late outcome of an abandoned one is stale


def _log_transition(key: str, state: Transition) -> None:
    level = logging.WARNING if state == "open" else logging.INFO
    _logger.log(
        level, "model_routing.cooldown %s %s", key, state, extra={"provider": key, "state": state}
    )


class ProviderCooldown:
    """Per-key breakers, safe to share between threads. Keep one per process (or per worker):
    a tracker built per request remembers nothing."""

    def __init__(
        self,
        *,
        failures_to_open: int = FAILURES_TO_OPEN,
        cooldown_s: float = COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
        classify: Classifier = is_retryable,
        on_transition: TransitionHook | None = None,
    ) -> None:
        if failures_to_open < 1:
            raise ValueError("failures_to_open must be at least 1")
        if not (math.isfinite(cooldown_s) and cooldown_s > 0):
            raise ValueError("cooldown_s must be a positive, finite number of seconds")
        self._failures_to_open = failures_to_open
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._classify = classify
        self._hook = on_transition or _log_transition
        self._lock = threading.Lock()
        self._states: dict[str, _State] = {}

    def admit(self, key: str, chain: Sequence[str]) -> Admission | None:
        """Whether a call to `key` may go ahead (`None`: skip it); `chain` is every key of the
        chain, in order. Report the call's outcome with `record_*`, passing what this returned:
        only a trial's outcome ends its trial. A trial whose outcome never comes back (a call
        that hangs, an admission dropped) is abandoned after `cooldown_s`, and the next one is
        admitted. `call`/`acall` also ignore the late outcome of an abandoned trial; with
        `admit` and `record_*` by hand it ends the current trial."""
        return self._admit_locked(key, chain)[0]

    def record_success(self, key: str, admission: Admission = "call") -> None:
        """A call that worked: the key closes and its failure count resets. That holds for a call
        admitted before the key opened too: the provider answered, so it is up."""
        self._success(key, admission, None)

    def record_failure(self, key: str, admission: Admission = "call") -> None:
        """Count a retryable failure; a failed trial reopens the key for another window. A call
        admitted before the key opened changes nothing once it is open."""
        self._failure(key, admission, None)

    def record_other(self, key: str, admission: Admission = "call") -> None:
        """An outcome that says nothing about the provider: it only ends the trial it was."""
        self._other(key, admission, None)

    def record_error(self, key: str, error: BaseException, admission: Admission = "call") -> None:
        """`record_failure` when `classify(error)`, else `record_other`. A classifier that
        raises still ends the trial, then its error propagates."""
        self._error(key, error, admission, None)

    def call[T](self, key: str, chain: Sequence[str], fn: Callable[[], T]) -> T:
        """Run `fn` behind `key`'s breaker: `ProviderCoolingDown` when refused, otherwise its
        outcome is recorded and its result returned or its error re-raised."""
        admission, token = self._admit_locked(key, chain)
        if admission is None:
            raise ProviderCoolingDown(key)
        try:
            result = fn()
        except BaseException as error:
            self._error(key, error, admission, token)
            raise
        self._success(key, admission, token)
        return result

    async def acall[T](self, key: str, chain: Sequence[str], fn: Callable[[], Awaitable[T]]) -> T:
        """`call` for a coroutine function. A cancellation ends a trial without counting."""
        admission, token = self._admit_locked(key, chain)
        if admission is None:
            raise ProviderCoolingDown(key)
        try:
            result = await fn()
        except BaseException as error:
            self._error(key, error, admission, token)
            raise
        self._success(key, admission, token)
        return result

    def _admit_locked(self, key: str, chain: Sequence[str]) -> tuple[Admission | None, int]:
        with self._lock:
            admission = self._admit(key, chain)
            token = self._states[key].trial_token
        if admission == "trial":
            self._on_transition(key, "half_open")
        return admission, token

    @staticmethod
    def _current(state: _State, admission: Admission, token: int | None) -> Admission:
        """A trial outcome whose lease was given to a newer trial counts as a plain call's."""
        if admission == "trial" and token is not None and token != state.trial_token:
            return "call"
        return admission

    def _success(self, key: str, admission: Admission, token: int | None) -> None:
        with self._lock:
            state = self._states.setdefault(key, _State())
            was_open = state.open_until is not None
            state.failures, state.open_until = 0, None
            if self._current(state, admission, token) == "trial":
                state.trial_since = None
        if was_open:
            self._on_transition(key, "closed")

    def _failure(self, key: str, admission: Admission, token: int | None) -> None:
        with self._lock:
            state = self._states.setdefault(key, _State())
            if self._current(state, admission, token) == "trial":
                state.trial_since = None
            elif state.open_until is not None:
                return
            else:
                state.failures += 1
                if state.failures < self._failures_to_open:
                    return
            state.open_until = self._clock() + self._cooldown_s
        self._on_transition(key, "open")

    def _other(self, key: str, admission: Admission, token: int | None) -> None:
        with self._lock:
            state = self._states.setdefault(key, _State())
            if self._current(state, admission, token) == "trial":
                state.trial_since = None

    def _error(
        self, key: str, error: BaseException, admission: Admission, token: int | None
    ) -> None:
        try:
            retryable = self._classify(error)
        except BaseException:
            self._other(key, admission, token)
            raise
        if retryable:
            self._failure(key, admission, token)
        else:
            self._other(key, admission, token)

    def _admit(self, key: str, chain: Sequence[str]) -> Admission | None:
        if key not in chain:
            chain = (*chain, key)
        state = self._states.setdefault(key, _State())
        if state.open_until is None:
            return "call"
        now = self._clock()
        if self._available(state, now):
            return self._grant_trial(state, now)
        if any(self._available(self._states.get(other), now) for other in chain):
            return None
        if self._soonest(chain) != key or self._trial_in_flight(state, now):
            return None
        return self._grant_trial(state, now)

    @staticmethod
    def _grant_trial(state: _State, now: float) -> Admission:
        state.trial_since = now
        state.trial_token += 1
        return "trial"

    def _trial_in_flight(self, state: _State, now: float) -> bool:
        return state.trial_since is not None and now - state.trial_since < self._cooldown_s

    def _available(self, state: _State | None, now: float) -> bool:
        if state is None or state.open_until is None:
            return True
        return now >= state.open_until and not self._trial_in_flight(state, now)

    def _on_transition(self, key: str, state: Transition) -> None:
        try:
            self._hook(key, state)
        except Exception:
            _logger.exception("model_routing.cooldown transition hook failed for %s", key)

    def _soonest(self, chain: Sequence[str]) -> str:
        def reopens_at(key: str) -> float:
            state = self._states.get(key)
            return state.open_until if state and state.open_until is not None else 0.0

        return min(chain, key=reopens_at)

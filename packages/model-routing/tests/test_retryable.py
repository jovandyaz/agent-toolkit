import pytest

from agent_toolkit.model_routing import is_retryable


class StatusError(Exception):
    def __init__(self, status_code: object) -> None:
        super().__init__("provider said no")
        self.status_code = status_code


class TransportError(Exception):
    """Stands in for an HTTP client's transport error (httpx.TransportError)."""


class BudgetStop(Exception):
    """Stands in for agent_toolkit.budgets.BudgetExceededError."""


class SelfDeciding(Exception):
    def __init__(self, retryable: bool) -> None:
        super().__init__("decides")
        self.retryable = retryable


def _wrapped(cause: BaseException) -> Exception:
    try:
        try:
            raise cause
        except BaseException as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        return outer


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("t"),
        ConnectionResetError("c"),
        StatusError(500),
        StatusError(503),
        StatusError(599),
        StatusError(429),
        StatusError(408),
        StatusError(409),
        _wrapped(TimeoutError("t")),
        _wrapped(StatusError(502)),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', '')}",
)
def test_retryable_errors(error: BaseException) -> None:
    assert is_retryable(error)


@pytest.mark.parametrize(
    "error",
    [
        StatusError(400),
        StatusError(401),
        StatusError(404),
        StatusError(600),
        StatusError("503"),
        StatusError(True),
        ValueError("nope"),
        KeyboardInterrupt(),
        _wrapped(ValueError("nope")),
    ],
    ids=lambda e: f"{type(e).__name__}-{getattr(e, 'status_code', '')!r}",
)
def test_non_retryable_errors(error: BaseException) -> None:
    assert not is_retryable(error)


def test_extra_retryable_types_count() -> None:
    assert not is_retryable(TransportError())
    assert is_retryable(TransportError(), retryable=(TransportError,))
    assert is_retryable(_wrapped(TransportError()), retryable=(TransportError,))


def test_a_never_type_anywhere_in_the_chain_wins() -> None:
    late = TimeoutError("outlived its deadline")
    late.__cause__ = BudgetStop()
    assert is_retryable(late)
    assert not is_retryable(late, never=(BudgetStop,))
    assert not is_retryable(_wrapped(BudgetStop()), never=(BudgetStop,))


def test_the_first_link_a_decider_answers_for_decides() -> None:
    def decide(link: BaseException) -> bool | None:
        return link.retryable if isinstance(link, SelfDeciding) else None

    overflow = SelfDeciding(retryable=False)
    overflow.__cause__ = StatusError(500)
    assert not is_retryable(overflow, decide=decide)
    assert is_retryable(_wrapped(SelfDeciding(retryable=True)), decide=decide)
    assert is_retryable(StatusError(503), decide=decide)  # None falls through.


def test_a_status_property_that_raises_is_not_retryable() -> None:
    class Broken(Exception):
        @property
        def status_code(self) -> int:
            raise RuntimeError("unreadable")

    assert not is_retryable(Broken())


def test_a_cause_cycle_ends() -> None:
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first
    assert not is_retryable(first)


def test_context_is_followed_too() -> None:
    try:
        try:
            raise TimeoutError("t")
        except TimeoutError:
            raise RuntimeError("while handling") from None
    except RuntimeError as error:
        # `from None` suppresses display but keeps __context__: the timeout is still there.
        assert is_retryable(error)

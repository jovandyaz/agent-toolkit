import pytest

from agent_toolkit.payments_core import RequestOutcome, may_retry_request


def test_a_request_that_was_never_sent_may_be_retried() -> None:
    assert may_retry_request("not_sent") is True


@pytest.mark.parametrize("outcome", ["rejected", "ambiguous"])
def test_a_request_the_provider_may_have_processed_is_never_retried(
    outcome: RequestOutcome,
) -> None:
    assert may_retry_request(outcome) is False


def test_an_unknown_outcome_fails_loudly() -> None:
    with pytest.raises(AssertionError):
        may_retry_request("timeout")  # type: ignore[arg-type]

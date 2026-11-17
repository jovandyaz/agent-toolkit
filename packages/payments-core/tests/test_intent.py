"""Invariants the models enforce, so no policy or adapter can build an inconsistent intent."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_toolkit.payments_core import (
    Attempt,
    Money,
    PaymentIntent,
    PaymentOutcome,
    ProviderPayment,
    RefundFlag,
)

T = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)


def mxn(amount_minor: int) -> Money:
    return Money(amount_minor=amount_minor, currency="MXN")


def attempt_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "id": "att_1",
        "rail": "codi_push",
        "provider_ref": "ref_1",
        "amount": mxn(35000),
        "created_at": T,
        "expires_at": T + timedelta(minutes=10),
        "status": "requested",
    }
    return fields | overrides


def intent_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "id": "pi_1",
        "principal_id": "merchant_1",
        "reference": "R-1001",
        "amount": mxn(35000),
        "idempotency_key": "idem_1",
        "status": "pending",
        "attempts": (Attempt.model_validate(attempt_fields()),),
        "paid_total": mxn(0),
        "late": False,
        "applied_event_ids": frozenset(),
    }
    return fields | overrides


def test_a_consistent_intent_is_accepted() -> None:
    intent = PaymentIntent.model_validate(intent_fields())

    assert intent.status == "pending"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"amount": mxn(0), "paid_total": mxn(0)}, "positive"),
        ({"paid_total": Money(amount_minor=0, currency="USD")}, "currency"),
        ({"paid_total": mxn(35001), "status": "paid"}, "exceed"),
        ({"paid_total": mxn(35000), "status": "pending"}, "disagrees"),
        ({"paid_total": mxn(20000), "status": "paid"}, "disagrees"),
        ({"paid_total": mxn(20000), "status": "expired"}, "disagrees"),
        ({"paid_total": mxn(0), "status": "partially_paid"}, "disagrees"),
        ({"paid_total": mxn(0), "status": "paid"}, "disagrees"),
    ],
)
def test_intent_rejects_inconsistent_money(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PaymentIntent.model_validate(intent_fields(**overrides))


def test_intent_rejects_attempts_in_another_currency() -> None:
    usd = Attempt.model_validate(attempt_fields(amount=Money(amount_minor=1, currency="USD")))

    with pytest.raises(ValidationError, match="currency"):
        PaymentIntent.model_validate(intent_fields(attempts=(usd,)))


def test_intent_rejects_repeated_attempt_ids() -> None:
    first = Attempt.model_validate(attempt_fields())
    second = Attempt.model_validate(attempt_fields(provider_ref="ref_2"))

    with pytest.raises(ValidationError, match="attempt ids"):
        PaymentIntent.model_validate(intent_fields(attempts=(first, second)))


def test_intent_rejects_two_attempts_for_the_same_provider_request() -> None:
    """A payment is matched to its attempt by `provider_ref`; a shared one is ambiguous."""
    first = Attempt.model_validate(attempt_fields())
    second = Attempt.model_validate(attempt_fields(id="att_2"))

    with pytest.raises(ValidationError, match="provider_ref"):
        PaymentIntent.model_validate(intent_fields(attempts=(first, second)))


def test_attempts_without_provider_ref_may_coexist() -> None:
    first = Attempt.model_validate(attempt_fields(provider_ref=None, status="ambiguous"))
    second = Attempt.model_validate(attempt_fields(id="att_2", provider_ref=None))

    intent = PaymentIntent.model_validate(intent_fields(attempts=(first, second)))

    assert len(intent.attempts) == 2


def test_intent_is_immutable() -> None:
    intent = PaymentIntent.model_validate(intent_fields())

    with pytest.raises(ValidationError):
        intent.status = "paid"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"expires_at": T},
        {"expires_at": T - timedelta(seconds=1)},
        {"amount": mxn(0)},
        {"created_at": datetime(2026, 11, 16, 18, 0)},
        {"id": ""},
        {"status": "failed"},
    ],
)
def test_attempt_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Attempt.model_validate(attempt_fields(**overrides))


def test_attempt_with_open_amount_is_accepted() -> None:
    attempt = Attempt.model_validate(attempt_fields(rail="codi_qr", amount=None))

    assert attempt.amount is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"amount": mxn(0)},
        {"received_at": datetime(2026, 11, 16, 18, 5)},
        {"provider_event_id": ""},
        {"unexpected": "field"},
    ],
)
def test_provider_payment_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "provider_event_id": "evt_1",
        "provider_ref": "ref_1",
        "amount": mxn(1),
        "received_at": T,
        "rail": "codi_push",
    }
    with pytest.raises(ValidationError):
        ProviderPayment.model_validate(fields | overrides)


def test_refund_flag_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        RefundFlag(reason="excess", amount=mxn(0), provider_event_id="evt_1")


def test_outcome_has_no_intent_exactly_when_orphan() -> None:
    intent = PaymentIntent.model_validate(intent_fields())

    with pytest.raises(ValidationError):
        PaymentOutcome(kind="orphan", intent=intent, refund_flags=())
    with pytest.raises(ValidationError):
        PaymentOutcome(kind="confirmed", intent=None, refund_flags=())

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_toolkit.payments_core import (
    Attempt,
    Money,
    PaymentIntent,
    PaymentOutcome,
    ProviderPayment,
    ProviderStatus,
    RefundFlag,
    apply_provider_payment,
    reconcile_attempt,
)

T = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)
EXPIRES_AT = T + timedelta(minutes=10)
BEFORE_EXPIRY = EXPIRES_AT - timedelta(minutes=1)
AFTER_EXPIRY = EXPIRES_AT + timedelta(seconds=15)

PENDING = ProviderStatus(kind="pending", payment=None)
EXPIRED_UNPAID = ProviderStatus(kind="expired_unpaid", payment=None)
NOT_FOUND = ProviderStatus(kind="not_found", payment=None)


def mxn(amount_minor: int) -> Money:
    return Money(amount_minor=amount_minor, currency="MXN")


def an_attempt(**overrides: object) -> Attempt:
    fields: dict[str, object] = {
        "id": "att_1",
        "rail": "codi_push",
        "provider_ref": "ref_1",
        "amount": mxn(35000),
        "created_at": T,
        "expires_at": EXPIRES_AT,
        "status": "requested",
    }
    return Attempt.model_validate(fields | overrides)


def an_intent(**overrides: object) -> PaymentIntent:
    fields: dict[str, object] = {
        "id": "pi_1",
        "principal_id": "merchant_1",
        "reference": "R-1001",
        "amount": mxn(35000),
        "idempotency_key": "idem_1",
        "status": "pending",
        "attempts": (an_attempt(),),
        "paid_total": mxn(0),
        "late": False,
        "applied_event_ids": frozenset(),
    }
    return PaymentIntent.model_validate(fields | overrides)


def a_payment(**overrides: object) -> ProviderPayment:
    fields: dict[str, object] = {
        "provider_event_id": "evt_1",
        "provider_ref": "ref_1",
        "amount": mxn(35000),
        "received_at": T + timedelta(minutes=5),
        "rail": "codi_push",
    }
    return ProviderPayment.model_validate(fields | overrides)


def paid(payment: ProviderPayment) -> ProviderStatus:
    return ProviderStatus(kind="paid", payment=payment)


def reconciled_intent(outcome: PaymentOutcome) -> PaymentIntent:
    assert outcome.intent is not None
    return outcome.intent


def test_paid_status_not_yet_applied_matches_apply_provider_payment() -> None:
    intent = an_intent()

    outcome = reconcile_attempt(intent, "att_1", paid(a_payment()), AFTER_EXPIRY)

    assert outcome == apply_provider_payment(intent, a_payment())
    assert outcome.kind == "confirmed"


def test_pending_leaves_the_intent_unchanged() -> None:
    # pialy: reconcile-attempt.test.ts > "leaves an unpaid attempt awaiting before its expiry"
    intent = an_intent()

    outcome = reconcile_attempt(intent, "att_1", PENDING, BEFORE_EXPIRY)

    assert outcome.kind == "pending"
    assert outcome.intent == intent
    assert outcome.refund_flags == ()


def test_expired_unpaid_expires_the_attempt_and_the_intent() -> None:
    # pialy: reconcile-attempt.test.ts > "expires an unpaid attempt past expiry"
    # pialy: domain.test.ts > expireAttempt > "expires the intent when no other attempt is
    # awaiting"
    outcome = reconcile_attempt(an_intent(), "att_1", EXPIRED_UNPAID, AFTER_EXPIRY)

    intent = reconciled_intent(outcome)
    assert outcome.kind == "expired"
    assert intent.attempts[0].status == "expired"
    assert intent.status == "expired"


def test_expired_unpaid_keeps_the_intent_open_while_another_attempt_is_live() -> None:
    # pialy: reconcile-attempt.test.ts > "keeps the intent open when another attempt of it is
    # still awaiting"
    qr = an_attempt(id="att_2", provider_ref="ref_2", expires_at=EXPIRES_AT + timedelta(hours=1))
    intent = an_intent(attempts=(an_attempt(), qr))

    outcome = reconcile_attempt(intent, "att_1", EXPIRED_UNPAID, AFTER_EXPIRY)

    assert outcome.kind == "expired"
    assert reconciled_intent(outcome).attempts[0].status == "expired"
    assert reconciled_intent(outcome).attempts[1].status == "requested"
    assert reconciled_intent(outcome).status == "pending"


def test_expired_unpaid_expires_the_intent_when_the_other_attempt_is_past_its_validity() -> None:
    stale = an_attempt(id="att_2", provider_ref="ref_2", expires_at=AFTER_EXPIRY)
    intent = an_intent(attempts=(an_attempt(), stale))

    outcome = reconcile_attempt(intent, "att_1", EXPIRED_UNPAID, AFTER_EXPIRY)

    assert reconciled_intent(outcome).status == "expired"
    assert reconciled_intent(outcome).attempts[1].status == "requested"


def test_expired_unpaid_never_expires_a_partially_paid_intent() -> None:
    paid_qr = an_attempt(rail="codi_qr", amount=None, status="paid")
    rest = an_attempt(id="att_2", provider_ref="ref_2", amount=mxn(15000))
    intent = an_intent(
        status="partially_paid",
        attempts=(paid_qr, rest),
        paid_total=mxn(20000),
        applied_event_ids=frozenset({"evt_1"}),
    )

    outcome = reconcile_attempt(intent, "att_2", EXPIRED_UNPAID, AFTER_EXPIRY)

    assert outcome.kind == "expired"
    assert reconciled_intent(outcome).status == "partially_paid"
    assert reconciled_intent(outcome).paid_total == mxn(20000)


def test_not_found_on_an_ambiguous_attempt_expires_it() -> None:
    """The request never reached the provider, so no money can arrive for it."""
    intent = an_intent(attempts=(an_attempt(status="ambiguous"),))

    outcome = reconcile_attempt(intent, "att_1", NOT_FOUND, AFTER_EXPIRY)

    assert outcome.kind == "expired"
    assert reconciled_intent(outcome).attempts[0].status == "expired"
    assert reconciled_intent(outcome).status == "expired"


def test_not_found_on_an_ambiguous_attempt_waits_until_its_expiry() -> None:
    """Before expiry a not-found may only mean the provider has not indexed it yet."""
    intent = an_intent(attempts=(an_attempt(status="ambiguous"),))

    outcome = reconcile_attempt(intent, "att_1", NOT_FOUND, BEFORE_EXPIRY)

    assert outcome.kind == "pending"
    assert outcome.intent == intent


def test_not_found_on_a_requested_attempt_changes_nothing() -> None:
    """The provider gave us this reference; only `expired_unpaid` proves no money is coming."""
    intent = an_intent()

    outcome = reconcile_attempt(intent, "att_1", NOT_FOUND, AFTER_EXPIRY)

    assert outcome.kind == "pending"
    assert outcome.intent == intent


def test_paid_status_for_another_request_raises() -> None:
    with pytest.raises(ValueError, match="ref_2"):
        reconcile_attempt(an_intent(), "att_1", paid(a_payment(provider_ref="ref_2")), AFTER_EXPIRY)


def test_reconcile_requires_a_timezone_aware_now() -> None:
    with pytest.raises(ValueError, match="timezone"):
        reconcile_attempt(an_intent(), "att_1", PENDING, datetime(2026, 11, 16, 18, 20))


@pytest.mark.parametrize("kind", ["pending", "expired_unpaid", "not_found"])
def test_provider_status_without_payment_rejects_one(kind: str) -> None:
    with pytest.raises(ValidationError):
        ProviderStatus.model_validate({"kind": kind, "payment": a_payment()})


def test_paid_provider_status_requires_the_payment() -> None:
    with pytest.raises(ValidationError):
        ProviderStatus(kind="paid", payment=None)


# Ported from pialy (cursor_hackathon_nov_2026, packages/payments/test). Each test names its
# source; cases already covered above carry the same kind of comment where they live.


def paid_intent(*more_attempts: Attempt) -> PaymentIntent:
    return an_intent(
        status="paid",
        paid_total=mxn(35000),
        attempts=(an_attempt(status="paid"), *more_attempts),
        applied_event_ids=frozenset({"evt_1"}),
    )


def test_payment_found_after_expiry_is_on_time_when_received_in_time() -> None:
    """Lateness follows when the payer paid, not when we asked the provider."""
    # pialy: reconcile-attempt.test.ts > "applies a payment the rail reports PAID for an
    # awaiting attempt past expiry"
    outcome = reconcile_attempt(an_intent(), "att_1", paid(a_payment()), AFTER_EXPIRY)

    assert outcome.kind == "confirmed"
    assert reconciled_intent(outcome).late is False
    assert reconciled_intent(outcome).attempts[0].status == "paid"


def test_paid_status_on_an_attempt_without_provider_ref_records_it() -> None:
    # pialy: apply-provider-payment.test.ts > "matches by our attempt id when the rail
    # transaction id is not stored yet"
    intent = an_intent(attempts=(an_attempt(provider_ref=None, status="ambiguous"),))

    outcome = reconcile_attempt(intent, "att_1", paid(a_payment()), AFTER_EXPIRY)

    assert outcome.kind == "confirmed"
    assert reconciled_intent(outcome).attempts[0].provider_ref == "ref_1"
    assert reconciled_intent(outcome).attempts[0].status == "paid"


def test_payment_reported_by_webhook_and_reconcile_counts_once() -> None:
    """Review Focus 3 across both paths that report the same payment."""
    # pialy: reconcile-attempt.test.ts > "applies a payment once when the webhook records it
    # while the rail is being asked"
    after_webhook = reconciled_intent(apply_provider_payment(an_intent(), a_payment()))

    outcome = reconcile_attempt(after_webhook, "att_1", paid(a_payment()), AFTER_EXPIRY)

    assert outcome.kind == "already_applied"
    assert outcome.intent == after_webhook
    assert outcome.refund_flags == ()


def test_new_payment_found_for_a_paid_intent_is_a_duplicate() -> None:
    """Reconcile reports a second payment like a webhook would: money to refund, not a replay."""
    second = a_payment(provider_event_id="evt_2")

    outcome = reconcile_attempt(paid_intent(), "att_1", paid(second), AFTER_EXPIRY)

    assert outcome.kind == "duplicate"
    assert outcome.refund_flags == (
        RefundFlag(reason="duplicate", amount=mxn(35000), provider_event_id="evt_2"),
    )
    assert reconciled_intent(outcome).paid_total == mxn(35000)


def test_late_payment_found_for_an_expired_attempt_confirms_late() -> None:
    """The payer paid after the attempt expired (ruling R6).

    pialy's version paid within validity and expected late; that case is confirmed here (spec
    7.6), see test_on_time_payment_for_a_stale_attempt_after_the_intent_expired_confirms.
    """
    # pialy: reconcile-attempt.test.ts > "applies a late payment found for an expired attempt
    # (late-payment watch)"
    intent = an_intent(status="expired", attempts=(an_attempt(status="expired"),))
    late = a_payment(received_at=EXPIRES_AT + timedelta(seconds=5))

    outcome = reconcile_attempt(intent, "att_1", paid(late), AFTER_EXPIRY)

    assert outcome.kind == "confirmed_late"
    assert reconciled_intent(outcome).late is True
    assert reconciled_intent(outcome).status == "paid"


def test_on_time_payment_for_a_stale_attempt_after_the_intent_expired_confirms() -> None:
    """Ruling R6: reconcile expired the intent, but the money was paid within validity."""
    stale = an_attempt(
        id="att_2",
        rail="codi_qr",
        provider_ref="ref_2",
        expires_at=EXPIRES_AT + timedelta(seconds=5),
    )
    intent = an_intent(attempts=(an_attempt(), stale))
    expired = reconciled_intent(reconcile_attempt(intent, "att_1", EXPIRED_UNPAID, AFTER_EXPIRY))
    assert expired.status == "expired"
    assert expired.attempts[1].status == "requested"
    on_time = a_payment(
        provider_ref="ref_2", rail="codi_qr", received_at=EXPIRES_AT + timedelta(seconds=2)
    )

    outcome = apply_provider_payment(expired, on_time)

    assert outcome.kind == "confirmed"
    assert reconciled_intent(outcome).late is False
    assert reconciled_intent(outcome).status == "paid"
    assert reconciled_intent(outcome).attempts[1].status == "paid"


def test_attempt_the_provider_never_knew_expires_after_its_validity() -> None:
    # pialy: reconcile-attempt.test.ts > "expires an attempt the rail never knew (no provider
    # ids) without calling the rail"
    intent = an_intent(attempts=(an_attempt(provider_ref=None, status="ambiguous"),))

    outcome = reconcile_attempt(intent, "att_1", NOT_FOUND, AFTER_EXPIRY)

    assert outcome.kind == "expired"
    assert reconciled_intent(outcome).status == "expired"


def test_never_expires_an_intent_that_already_succeeded() -> None:
    # pialy: domain.test.ts > expireAttempt > "never expires an intent that already succeeded"
    intent = paid_intent(an_attempt(id="att_2", rail="codi_qr", provider_ref="ref_2"))

    outcome = reconcile_attempt(intent, "att_2", EXPIRED_UNPAID, AFTER_EXPIRY)

    assert reconciled_intent(outcome).status == "paid"
    assert reconciled_intent(outcome).attempts[1].status == "expired"


@pytest.mark.parametrize("status", [PENDING, EXPIRED_UNPAID, NOT_FOUND])
def test_a_paid_attempt_is_never_downgraded(status: ProviderStatus) -> None:
    # pialy: reconcile-attempt.test.ts > "does not ask the rail about an attempt that is already
    # paid"
    # pialy: domain.test.ts > expireAttempt > "is a no-op for an attempt that is already
    # succeeded"
    outcome = reconcile_attempt(paid_intent(), "att_1", status, AFTER_EXPIRY)

    assert outcome.kind == "already_applied"
    assert outcome.intent == paid_intent()


@pytest.mark.parametrize("status", [PENDING, EXPIRED_UNPAID, NOT_FOUND])
def test_an_expired_attempt_with_nothing_paid_stays_expired(status: ProviderStatus) -> None:
    # pialy: reconcile-attempt.test.ts > "reports an expired attempt with nothing paid as
    # expired, and a rail failure without changing it"
    # pialy: domain.test.ts > expireAttempt > "is a no-op for an attempt that is already expired"
    intent = an_intent(status="expired", attempts=(an_attempt(status="expired"),))

    outcome = reconcile_attempt(intent, "att_1", status, AFTER_EXPIRY)

    assert outcome.kind == "expired"
    assert outcome.intent == intent


def test_unknown_attempt_fails_loudly() -> None:
    # pialy: reconcile-attempt.test.ts > "fails loudly for an attempt that does not exist"
    # pialy: domain.test.ts > expireAttempt > "throws AttemptIntentMismatchError when the
    # attempt belongs to another intent"
    with pytest.raises(ValueError, match="nope"):
        reconcile_attempt(an_intent(), "nope", PENDING, AFTER_EXPIRY)

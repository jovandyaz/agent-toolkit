from datetime import UTC, datetime, timedelta

import pytest

from agent_toolkit.payments_core import (
    Attempt,
    CurrencyMismatchError,
    Money,
    PaymentIntent,
    PaymentOutcome,
    ProviderPayment,
    RefundFlag,
    apply_provider_payment,
)

T = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)
EXPIRES_AT = T + timedelta(minutes=10)
WITHIN_VALIDITY = T + timedelta(minutes=5)


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
        "received_at": WITHIN_VALIDITY,
        "rail": "codi_push",
    }
    return ProviderPayment.model_validate(fields | overrides)


def an_open_amount_intent() -> PaymentIntent:
    """A QR the payer fills in, so partial payments and overpayments can happen."""
    return an_intent(attempts=(an_attempt(rail="codi_qr", amount=None),))


def applied_intent(outcome: PaymentOutcome) -> PaymentIntent:
    assert outcome.intent is not None
    return outcome.intent


def test_payment_within_validity_confirms() -> None:
    # pialy: domain.test.ts > recordPayment > "succeeds a processing intent and its attempt"
    # pialy: apply-provider-payment.test.ts > "applies a matched payment once"
    outcome = apply_provider_payment(an_intent(), a_payment())

    intent = applied_intent(outcome)
    assert outcome.kind == "confirmed"
    assert intent.status == "paid"
    assert intent.late is False
    assert intent.paid_total == mxn(35000)
    assert intent.applied_event_ids == frozenset({"evt_1"})
    assert intent.attempts[0].status == "paid"
    assert outcome.refund_flags == ()


def test_payment_at_the_expiry_instant_is_on_time() -> None:
    outcome = apply_provider_payment(an_intent(), a_payment(received_at=EXPIRES_AT))

    assert outcome.kind == "confirmed"
    assert applied_intent(outcome).late is False


def test_late_payment_confirms_and_flags_late() -> None:
    late = a_payment(received_at=EXPIRES_AT + timedelta(seconds=1))

    outcome = apply_provider_payment(an_intent(), late)

    intent = applied_intent(outcome)
    assert outcome.kind == "confirmed_late"
    assert intent.late is True
    assert intent.status == "paid"
    assert intent.paid_total == mxn(35000)


def test_on_time_payment_on_an_attempt_marked_expired_confirms() -> None:
    # pialy: domain.test.ts > recordPayment > "still emits PaymentSucceeded when the attempt
    # expired but the intent is processing"
    push = an_attempt(status="expired")
    qr = an_attempt(
        id="att_2", rail="codi_qr", provider_ref="ref_2", expires_at=EXPIRES_AT + timedelta(hours=1)
    )
    intent = an_intent(attempts=(push, qr))

    outcome = apply_provider_payment(intent, a_payment(received_at=WITHIN_VALIDITY))

    assert outcome.kind == "confirmed"
    assert applied_intent(outcome).late is False
    assert applied_intent(outcome).attempts[0].status == "paid"


def test_lateness_follows_the_attempt_that_was_paid() -> None:
    """Ruling R6: late means after the paid attempt's own expiry, even if another is open."""
    push = an_attempt()
    qr = an_attempt(
        id="att_2", rail="codi_qr", provider_ref="ref_2", expires_at=EXPIRES_AT + timedelta(hours=1)
    )
    intent = an_intent(attempts=(push, qr))

    outcome = apply_provider_payment(
        intent, a_payment(received_at=EXPIRES_AT + timedelta(minutes=1))
    )

    assert outcome.kind == "confirmed_late"
    assert applied_intent(outcome).late is True


def test_replayed_event_is_already_applied() -> None:
    """Review Focus 3: a provider event delivered twice is counted once."""
    # pialy: apply-provider-payment.test.ts > "reports a second delivery of the same bank
    # payment as duplicate and records nothing new"
    first = apply_provider_payment(an_intent(), a_payment())

    replay = apply_provider_payment(applied_intent(first), a_payment())

    assert replay.kind == "already_applied"
    assert replay.intent == first.intent
    assert applied_intent(replay).paid_total == mxn(35000)
    assert replay.refund_flags == ()


def test_replayed_partial_payment_is_not_counted_twice() -> None:
    """Review Focus 3 on a partially paid intent, where a second count would change the total."""
    partial = a_payment(amount=mxn(20000))
    first = apply_provider_payment(an_open_amount_intent(), partial)

    replay = apply_provider_payment(applied_intent(first), partial)

    assert replay.kind == "already_applied"
    assert applied_intent(replay).paid_total == mxn(20000)
    assert applied_intent(replay).status == "partially_paid"


def test_second_distinct_payment_on_paid_intent_flags_duplicate() -> None:
    # pialy: domain.test.ts > recordPayment > "leaves a succeeded intent unchanged and emits
    # DuplicatePaymentReceived"
    paid = applied_intent(apply_provider_payment(an_intent(), a_payment()))
    second = a_payment(provider_event_id="evt_2", amount=mxn(35000))

    outcome = apply_provider_payment(paid, second)

    intent = applied_intent(outcome)
    assert outcome.kind == "duplicate"
    assert outcome.refund_flags == (
        RefundFlag(reason="duplicate", amount=mxn(35000), provider_event_id="evt_2"),
    )
    assert intent.paid_total == mxn(35000)
    assert intent.status == "paid"
    assert intent.applied_event_ids == frozenset({"evt_1", "evt_2"})


def test_replayed_duplicate_is_flagged_for_refund_only_once() -> None:
    """Only the same `provider_event_id` is `already_applied`: a harmless replay, no refund."""
    paid = applied_intent(apply_provider_payment(an_intent(), a_payment()))
    second = a_payment(provider_event_id="evt_2")
    flagged = applied_intent(apply_provider_payment(paid, second))

    replay = apply_provider_payment(flagged, second)

    assert replay.kind == "already_applied"
    assert replay.refund_flags == ()
    assert replay.intent == flagged


def test_payment_without_intent_is_orphan() -> None:
    # pialy: apply-provider-payment.test.ts > "records an unmatched payment without applying it"
    outcome = apply_provider_payment(None, a_payment(amount=mxn(12345)))

    assert outcome.kind == "orphan"
    assert outcome.intent is None
    assert outcome.refund_flags == (
        RefundFlag(reason="orphan", amount=mxn(12345), provider_event_id="evt_1"),
    )


def test_partial_then_complete() -> None:
    first = apply_provider_payment(an_open_amount_intent(), a_payment(amount=mxn(20000)))

    assert first.kind == "partial"
    assert applied_intent(first).status == "partially_paid"
    assert applied_intent(first).paid_total == mxn(20000)
    assert first.refund_flags == ()

    second = apply_provider_payment(
        applied_intent(first), a_payment(provider_event_id="evt_2", amount=mxn(15000))
    )

    assert second.kind == "confirmed"
    assert applied_intent(second).status == "paid"
    assert applied_intent(second).paid_total == mxn(35000)
    assert second.refund_flags == ()


def test_overpayment_flags_excess() -> None:
    outcome = apply_provider_payment(an_open_amount_intent(), a_payment(amount=mxn(40000)))

    assert outcome.kind == "confirmed"
    assert applied_intent(outcome).paid_total == mxn(35000)
    assert outcome.refund_flags == (
        RefundFlag(reason="excess", amount=mxn(5000), provider_event_id="evt_1"),
    )


def test_payment_completing_a_partial_with_extra_flags_only_the_excess() -> None:
    first = apply_provider_payment(an_open_amount_intent(), a_payment(amount=mxn(20000)))

    second = apply_provider_payment(
        applied_intent(first), a_payment(provider_event_id="evt_2", amount=mxn(20000))
    )

    assert second.kind == "confirmed"
    assert applied_intent(second).paid_total == mxn(35000)
    assert second.refund_flags == (
        RefundFlag(reason="excess", amount=mxn(5000), provider_event_id="evt_2"),
    )


def test_late_partial_payment_marks_the_intent_late() -> None:
    late_partial = a_payment(amount=mxn(20000), received_at=EXPIRES_AT + timedelta(minutes=1))

    outcome = apply_provider_payment(an_open_amount_intent(), late_partial)

    assert outcome.kind == "partial"
    assert applied_intent(outcome).late is True


@pytest.mark.parametrize(
    "amounts",
    [
        (35000,),
        (20000, 15000),
        (40000,),
        (20000, 20000, 1),
        (35000, 35000, 100),
        (1, 2, 3, 34999),
    ],
)
def test_every_received_centavo_is_either_applied_or_flagged_for_refund(
    amounts: tuple[int, ...],
) -> None:
    intent = an_open_amount_intent()
    flagged = 0
    for number, amount in enumerate(amounts):
        payment = a_payment(provider_event_id=f"evt_{number}", amount=mxn(amount))
        outcome = apply_provider_payment(intent, payment)
        intent = applied_intent(outcome)
        flagged += sum(flag.amount.amount_minor for flag in outcome.refund_flags)

    assert intent.paid_total.amount_minor + flagged == sum(amounts)
    assert intent.paid_total.amount_minor <= intent.amount.amount_minor


def test_currency_mismatch_raises() -> None:
    usd = a_payment(amount=Money(amount_minor=35000, currency="USD"))

    with pytest.raises(CurrencyMismatchError):
        apply_provider_payment(an_intent(), usd)


def test_payment_for_a_request_the_intent_never_made_raises() -> None:
    """The caller looked up the wrong intent: applying the money there would misattribute it."""
    # pialy: domain.test.ts > recordPayment > "throws AttemptIntentMismatchError when the
    # attempt belongs to another intent"
    with pytest.raises(ValueError, match="ref_other"):
        apply_provider_payment(an_intent(), a_payment(provider_ref="ref_other"))


# Ported from pialy (cursor_hackathon_nov_2026, packages/payments/test). Each test names its
# source; cases already covered above carry the same kind of comment where they live.


def test_on_time_payment_on_an_expired_intent_confirms() -> None:
    """Diverges from pialy, which marked it late: spec 7.6 confirms a payment made in validity.

    Ruling R6: an expired intent does not make an on-time payment late.
    """
    # pialy: domain.test.ts > recordPayment > "succeeds a expired intent with a
    # LatePaymentReceived event"
    # pialy: apply-provider-payment.test.ts > "marks a payment for an expired intent as late"
    intent = an_intent(status="expired", attempts=(an_attempt(status="expired"),))

    outcome = apply_provider_payment(intent, a_payment(received_at=WITHIN_VALIDITY))

    assert outcome.kind == "confirmed"
    assert applied_intent(outcome).late is False
    assert applied_intent(outcome).status == "paid"
    assert applied_intent(outcome).attempts[0].status == "paid"


def test_applying_a_payment_only_marks_its_attempt_paid() -> None:
    # pialy: domain.test.ts > recordPayment > "keeps provider ids already on the attempt"
    outcome = apply_provider_payment(an_intent(), a_payment())

    assert applied_intent(outcome).attempts == (an_attempt(status="paid"),)


def test_payment_is_matched_to_the_attempt_with_its_provider_ref() -> None:
    # pialy: apply-provider-payment.test.ts > "prefers the rail transaction id, then the QR id,
    # over our attempt id"
    push = an_attempt()
    qr = an_attempt(id="att_2", rail="codi_qr", provider_ref="ref_2")

    outcome = apply_provider_payment(
        an_intent(attempts=(push, qr)), a_payment(provider_ref="ref_2")
    )

    assert [a.status for a in applied_intent(outcome).attempts] == ["requested", "paid"]


def test_duplicate_on_another_attempt_marks_that_attempt_paid_without_counting_it() -> None:
    # pialy: domain.test.ts > recordPayment > "marks a second paid attempt succeeded on a
    # duplicate without touching the intent"
    push = an_attempt()
    qr = an_attempt(id="att_2", rail="codi_qr", provider_ref="ref_2")
    paid = applied_intent(apply_provider_payment(an_intent(attempts=(push, qr)), a_payment()))

    outcome = apply_provider_payment(
        paid, a_payment(provider_event_id="evt_2", provider_ref="ref_2")
    )

    intent = applied_intent(outcome)
    assert outcome.kind == "duplicate"
    assert intent.attempts[1].status == "paid"
    assert intent.paid_total == mxn(35000)
    assert outcome.refund_flags == (
        RefundFlag(reason="duplicate", amount=mxn(35000), provider_event_id="evt_2"),
    )


def test_payment_differing_from_a_fixed_amount_request_is_still_accounted() -> None:
    """Deliberate change from pialy, which rejected it: money that arrived is never dropped."""
    # pialy: domain.test.ts > recordPayment > "throws AmountMismatchError when the amount
    # differs"
    outcome = apply_provider_payment(an_intent(), a_payment(amount=mxn(100)))

    assert outcome.kind == "partial"
    assert applied_intent(outcome).paid_total == mxn(100)

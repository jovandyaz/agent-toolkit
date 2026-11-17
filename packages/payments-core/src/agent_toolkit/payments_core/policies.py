"""Payment policies as pure functions: they never mutate their input and do no I/O."""

from datetime import datetime
from typing import Literal, assert_never

from pydantic import BaseModel

from agent_toolkit.payments_core.intent import (
    Attempt,
    IntentStatus,
    OutcomeKind,
    PaymentIntent,
    PaymentOutcome,
    ProviderPayment,
    ProviderStatus,
    RefundFlag,
    RefundReason,
)
from agent_toolkit.payments_core.money import CurrencyMismatchError, Money

RequestOutcome = Literal["not_sent", "rejected", "ambiguous"]


def apply_provider_payment(
    intent: PaymentIntent | None, payment: ProviderPayment
) -> PaymentOutcome:
    """Account for a payment the provider reported, at most once per `provider_event_id`.

    - No intent: `orphan`, the whole payment flagged for refund.
    - Same `provider_event_id` as an applied event: `already_applied`, a harmless replay;
      nothing changes and nothing is flagged.
    - Another payment for an intent already paid: `duplicate`, the whole payment flagged for
      refund; the event is recorded so its own replay is `already_applied`.
    - Otherwise the payment counts up to the amount still owed (`partial`, `confirmed` or
      `confirmed_late`) and anything beyond it is flagged as `excess`. It is late only when
      received after the `expires_at` of the attempt it paid, even if the intent had expired.

    The caller must read the intent and save the outcome in one unit of work that locks the
    intent (or rejects a concurrent change), or two payments applied at once can overwrite each
    other. An `orphan` has no intent to remember its event, so the caller must deduplicate
    orphan refund flags by `provider_event_id`.

    Raises `CurrencyMismatchError` for a payment in another currency and `ValueError` when no
    attempt of the intent has the payment's `provider_ref`.
    """
    if intent is None:
        return PaymentOutcome(
            kind="orphan", intent=None, refund_flags=(_refund("orphan", payment.amount, payment),)
        )
    attempt = _attempt_with_ref(intent, payment.provider_ref)
    return _apply(intent, attempt, payment)


def reconcile_attempt(
    intent: PaymentIntent, attempt_id: str, status: ProviderStatus, now: datetime
) -> PaymentOutcome:
    """Bring one attempt in line with what the provider reports about it at `now`.

    - `paid`: applied exactly as `apply_provider_payment` would; an attempt that has no
      `provider_ref` yet (an ambiguous request) takes the payment's.
    - `pending`: nothing changes (`pending`).
    - `expired_unpaid`: the attempt expires (`expired`).
    - `not_found`: an `ambiguous` attempt past its expiry never reached the provider and
      expires; anything else stays as it is, because only `expired_unpaid` proves that no
      money is coming for a request the provider accepted.

    The intent expires with its attempt only when nothing was paid and no other attempt is
    still open and within its validity. A paid attempt is never downgraded (`already_applied`).

    Raises `ValueError` for an unknown `attempt_id`, a naive `now`, or a payment that belongs
    to another request.
    """
    if now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    attempt = _attempt_with_id(intent, attempt_id)
    if status.payment is not None:
        return _apply_on_attempt(intent, attempt, status.payment)
    if status.kind == "expired_unpaid" or _never_reached_provider(attempt, status, now):
        return _expire(intent, attempt, now)
    return _unchanged(intent, attempt)


def may_retry_request(outcome: RequestOutcome) -> bool:
    """Retry a request to the provider only when the failure proves it was never processed.

    `not_sent`: it never left (for example, the connection was refused). `rejected`: the
    provider answered and refused it. `ambiguous`: no conclusive answer (for example, a
    timeout), so it may have been processed; reconcile it instead of sending it again.
    """
    match outcome:
        case "not_sent":
            return True
        case "rejected" | "ambiguous":
            return False
        case _:
            assert_never(outcome)


def _apply_on_attempt(
    intent: PaymentIntent, attempt: Attempt, payment: ProviderPayment
) -> PaymentOutcome:
    if attempt.provider_ref not in (None, payment.provider_ref):
        raise ValueError(
            f"attempt {attempt.id!r} has provider_ref {attempt.provider_ref!r}, "
            f"but the payment is for {payment.provider_ref!r}"
        )
    return _apply(intent, attempt, payment)


def _apply(intent: PaymentIntent, attempt: Attempt, payment: ProviderPayment) -> PaymentOutcome:
    if payment.amount.currency != intent.amount.currency:
        raise CurrencyMismatchError(intent.amount.currency, payment.amount.currency)
    if payment.provider_event_id in intent.applied_event_ids:
        return PaymentOutcome(kind="already_applied", intent=intent, refund_flags=())

    paid_attempt = _evolve(attempt, status="paid", provider_ref=payment.provider_ref)
    attempts = tuple(paid_attempt if a.id == attempt.id else a for a in intent.attempts)
    applied_event_ids = intent.applied_event_ids | {payment.provider_event_id}

    if intent.status == "paid":
        updated = _evolve(intent, attempts=attempts, applied_event_ids=applied_event_ids)
        duplicate = _refund("duplicate", payment.amount, payment)
        return PaymentOutcome(kind="duplicate", intent=updated, refund_flags=(duplicate,))

    owed = intent.amount - intent.paid_total
    counted = min(payment.amount, owed, key=lambda money: money.amount_minor)
    excess = payment.amount - counted
    paid_total = intent.paid_total + counted
    late = _is_late(attempt, payment)
    status: IntentStatus = "paid" if paid_total == intent.amount else "partially_paid"
    updated = _evolve(
        intent,
        attempts=attempts,
        applied_event_ids=applied_event_ids,
        paid_total=paid_total,
        status=status,
        late=intent.late or late,
    )
    refund_flags = (_refund("excess", excess, payment),) if excess.amount_minor else ()
    return PaymentOutcome(kind=_kind(status, late), intent=updated, refund_flags=refund_flags)


def _kind(status: IntentStatus, late: bool) -> OutcomeKind:
    if status == "partially_paid":
        return "partial"
    return "confirmed_late" if late else "confirmed"


def _is_late(attempt: Attempt, payment: ProviderPayment) -> bool:
    """Late only when received after the paid attempt's own `expires_at` (spec 7.6, ruling R6).

    An expired intent does not make an on-time payment late: the payer paid within validity.
    """
    return payment.received_at > attempt.expires_at


def _never_reached_provider(attempt: Attempt, status: ProviderStatus, now: datetime) -> bool:
    return (
        status.kind == "not_found" and attempt.status == "ambiguous" and now >= attempt.expires_at
    )


def _expire(intent: PaymentIntent, attempt: Attempt, now: datetime) -> PaymentOutcome:
    if attempt.status == "paid":
        return _unchanged(intent, attempt)
    expired = _evolve(attempt, status="expired")
    attempts = tuple(expired if a.id == attempt.id else a for a in intent.attempts)
    none_open = not any(_is_open(a, now) for a in attempts)
    status: IntentStatus = "expired" if intent.status == "pending" and none_open else intent.status
    updated = _evolve(intent, attempts=attempts, status=status)
    return PaymentOutcome(kind="expired", intent=updated, refund_flags=())


def _is_open(attempt: Attempt, now: datetime) -> bool:
    return attempt.status in ("requested", "ambiguous") and now < attempt.expires_at


def _unchanged(intent: PaymentIntent, attempt: Attempt) -> PaymentOutcome:
    return PaymentOutcome(kind=_kind_when_unchanged(attempt), intent=intent, refund_flags=())


def _kind_when_unchanged(attempt: Attempt) -> OutcomeKind:
    match attempt.status:
        case "paid":
            return "already_applied"
        case "expired":
            return "expired"
        case "requested" | "ambiguous":
            return "pending"
        case _:
            assert_never(attempt.status)


def _attempt_with_id(intent: PaymentIntent, attempt_id: str) -> Attempt:
    for attempt in intent.attempts:
        if attempt.id == attempt_id:
            return attempt
    raise ValueError(f"intent {intent.id!r} has no attempt {attempt_id!r}")


def _attempt_with_ref(intent: PaymentIntent, provider_ref: str) -> Attempt:
    for attempt in intent.attempts:
        if attempt.provider_ref == provider_ref:
            return attempt
    raise ValueError(f"intent {intent.id!r} has no attempt with provider_ref {provider_ref!r}")


def _refund(reason: RefundReason, amount: Money, payment: ProviderPayment) -> RefundFlag:
    return RefundFlag(reason=reason, amount=amount, provider_event_id=payment.provider_event_id)


def _evolve[M: BaseModel](model: M, **changes: object) -> M:
    """Copy `model` with `changes`, validating the result.

    `model_copy(update=...)` skips validation, which would let a policy bug produce an intent
    that breaks its invariants; rebuilding through `model_validate` re-checks them.
    """
    return type(model).model_validate({**dict(model), **changes})

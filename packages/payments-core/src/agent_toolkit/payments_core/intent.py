"""Payment intents, their attempts on a rail, and what providers report about them."""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from agent_toolkit.payments_core.money import CurrencyMismatchError, Money

IntentStatus = Literal["pending", "partially_paid", "paid", "expired"]
AttemptStatus = Literal["requested", "ambiguous", "paid", "expired"]
RefundReason = Literal["duplicate", "orphan", "excess"]
ProviderStatusKind = Literal["paid", "pending", "expired_unpaid", "not_found"]
# `already_applied`: the same provider event again, a harmless replay with nothing to refund.
# `duplicate`: a different payment for an intent already paid, flagged for refund.
# `pending` and `expired` are reconcile results: nothing changed yet, or the attempt expired.
OutcomeKind = Literal[
    "confirmed",
    "confirmed_late",
    "partial",
    "already_applied",
    "duplicate",
    "orphan",
    "pending",
    "expired",
]

Identifier = Annotated[str, Field(min_length=1)]


def _require_positive(money: Money, field: str) -> None:
    if money.amount_minor <= 0:
        raise ValueError(f"{field} must be positive, got {money}")


class Attempt(BaseModel):
    """One request to pay sent over a rail. `amount=None` means the payer sets it (open QR)."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: Identifier
    rail: Identifier
    provider_ref: Identifier | None
    amount: Money | None
    created_at: AwareDatetime
    expires_at: AwareDatetime
    status: AttemptStatus

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("an attempt must expire after it is created")
        if self.amount is not None:
            _require_positive(self.amount, "attempt amount")
        return self


class PaymentIntent(BaseModel):
    """What a principal asked to collect, the attempts made, and the money applied to it.

    Invariants: every amount is in `amount.currency`; `paid_total` never exceeds `amount`
    (money beyond it is flagged for refund, never applied); `status` agrees with `paid_total`;
    `applied_event_ids` holds every provider event already accounted for.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: Identifier
    principal_id: Identifier
    reference: Identifier
    amount: Money
    idempotency_key: Identifier
    status: IntentStatus
    attempts: tuple[Attempt, ...]
    paid_total: Money
    late: bool
    applied_event_ids: frozenset[str]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_positive(self.amount, "intent amount")
        currency = self.amount.currency
        amounts = [self.paid_total, *(a.amount for a in self.attempts if a.amount is not None)]
        for money in amounts:
            if money.currency != currency:
                raise CurrencyMismatchError(currency, money.currency)
        if len({a.id for a in self.attempts}) != len(self.attempts):
            raise ValueError("attempt ids must be unique within an intent")
        refs = [a.provider_ref for a in self.attempts if a.provider_ref is not None]
        if len(set(refs)) != len(refs):
            raise ValueError("each provider_ref must belong to a single attempt")
        if self.paid_total.amount_minor > self.amount.amount_minor:
            raise ValueError("paid_total cannot exceed the intent amount")
        if self.status not in _statuses_for(self.paid_total, self.amount):
            raise ValueError(f"status {self.status!r} disagrees with paid_total {self.paid_total}")
        return self


def _statuses_for(paid_total: Money, amount: Money) -> frozenset[IntentStatus]:
    if paid_total == amount:
        return frozenset({"paid"})
    if paid_total.amount_minor > 0:
        return frozenset({"partially_paid"})
    return frozenset({"pending", "expired"})


class ProviderPayment(BaseModel):
    """Money the provider reports as received for the request `provider_ref`.

    `provider_event_id` identifies this payment at the provider; it is the idempotency key.
    The webhook and the provider query must report the same `provider_event_id` for the same
    money, or reconciling a payment a webhook already applied flags a false `duplicate`.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    provider_event_id: Identifier
    provider_ref: Identifier
    amount: Money
    received_at: AwareDatetime
    rail: Identifier

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_positive(self.amount, "payment amount")
        return self


class RefundFlag(BaseModel):
    """Money received that must go back to the payer. Rails like CoDi have no refund API."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    reason: RefundReason
    amount: Money
    provider_event_id: Identifier

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_positive(self.amount, "refund amount")
        return self


class ProviderStatus(BaseModel):
    """What the provider says about one request; `payment` is present exactly when `paid`."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: ProviderStatusKind
    payment: ProviderPayment | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.kind == "paid") != (self.payment is not None):
            raise ValueError("a provider status carries a payment exactly when it is paid")
        return self


class PaymentOutcome(BaseModel):
    """Result of a payment policy: the new intent (None only for an orphan) and refunds to make."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: OutcomeKind
    intent: PaymentIntent | None
    refund_flags: tuple[RefundFlag, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.kind == "orphan") != (self.intent is None):
            raise ValueError("an outcome has no intent exactly when it is an orphan")
        return self

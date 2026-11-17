"""Pure check of a closed mandate (one charge) against the open mandate that should cover it."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent_toolkit.mandates.models import Amount, ClosedMandate, OpenMandate, require_aware

Violation = Literal[
    "expired",
    "revoked",
    "mandate_mismatch",
    "account_not_allowed",
    "rail_not_allowed",
    "currency_mismatch",
    "amount_out_of_range",
    "budget_exceeded",
    "outside_window",
]
ApprovalReason = Literal["payer_not_allowed", "open_amount"]


class Verdict(BaseModel):
    """Every violation found (in `Violation` order) and every reason a human must approve."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    violations: tuple[Violation, ...]
    approval_reasons: tuple[ApprovalReason, ...]

    @property
    def ok(self) -> bool:
        """True when the charge fits the mandate and needs no approval."""
        return not self.violations and not self.approval_reasons


def reserve_amount(closed: ClosedMandate) -> Amount | None:
    """What the charge takes from the budget: the amount, the increase's delta, or None if open."""
    if closed.amount is None:
        return None
    if closed.previous_amount is None:
        return closed.amount
    delta = closed.amount.minor - closed.previous_amount.minor
    return Amount(minor=delta, currency=closed.amount.currency)


def check(closed: ClosedMandate, open: OpenMandate, consumed: Amount, now: datetime) -> Verdict:
    """Judge whether `closed` fits `open` at `now`, given the budget `consumed` this period.

    `consumed` comes from the caller (see `consumed_in_period`) and already includes what a
    pending link reserved before an increase. All violations are collected. An open amount
    needs approval and skips the range and budget checks, since nothing bounds it. Raises
    ValueError when `now` is naive.
    """
    require_aware(now, "now")
    return Verdict(
        violations=_violations(closed, open, consumed, now),
        approval_reasons=_approval_reasons(closed, open),
    )


def _violations(
    closed: ClosedMandate, mandate: OpenMandate, consumed: Amount, now: datetime
) -> tuple[Violation, ...]:
    charge_amounts = [a for a in (closed.amount, closed.previous_amount) if a is not None]
    charge_currency_ok = all(a.currency == mandate.currency for a in charge_amounts)
    consumed_currency_ok = consumed.currency == mandate.currency
    reserve = reserve_amount(closed)

    found: list[Violation] = []
    if now >= mandate.expires_at:
        found.append("expired")
    if mandate.revoked_at is not None and now >= mandate.revoked_at:
        found.append("revoked")
    if closed.open_mandate_id != mandate.id:
        found.append("mandate_mismatch")
    if closed.receiving_account not in mandate.receiving_accounts:
        found.append("account_not_allowed")
    if closed.rail not in mandate.allowed_rails:
        found.append("rail_not_allowed")
    if not (charge_currency_ok and consumed_currency_ok):
        found.append("currency_mismatch")
    # Amounts in another currency cannot be bounded; currency_mismatch already rejects them.
    if (
        closed.amount is not None
        and charge_currency_ok
        and not mandate.amount_range.contains(closed.amount)
    ):
        found.append("amount_out_of_range")
    if (
        reserve is not None
        and charge_currency_ok
        and consumed_currency_ok
        and consumed.minor + reserve.minor > mandate.budget.limit.minor
    ):
        found.append("budget_exceeded")
    window = mandate.daily_window
    if window is not None and not window.contains(now.astimezone(mandate.zone).time()):
        found.append("outside_window")
    return tuple(found)


def _approval_reasons(closed: ClosedMandate, mandate: OpenMandate) -> tuple[ApprovalReason, ...]:
    found: list[ApprovalReason] = []
    if closed.payer not in mandate.allowed_payers:
        found.append("payer_not_allowed")
    if closed.amount is None:
        found.append("open_amount")
    return tuple(found)

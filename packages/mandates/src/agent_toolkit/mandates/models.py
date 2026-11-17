"""What an owner authorized an agent to do (open mandate) and one concrete charge (closed mandate).

Adapted from AP2 with the money flowing the other way: the owner authorizes the agent to ISSUE
payment requests to customers (CoDi), and each payer approves separately in their banking app.
"""

import hashlib
import json
from datetime import datetime, time
from functools import cache
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, available_timezones

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1)]
BudgetPeriod = Literal["day", "week", "month", "total"]
ChargeKind = Literal["new_charge", "amount_increase"]


@cache
def _known_time_zones() -> frozenset[str]:
    """IANA keys from the system database and the `tzdata` package, read once per process."""
    return frozenset(available_timezones())


def require_aware(moment: datetime, name: str) -> None:
    """Raise ValueError unless `moment` carries a time zone."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got {moment!r}")


class Amount(BaseModel):
    """A non-negative amount in minor units (centavos for MXN) of one ISO 4217 currency."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    minor: Annotated[int, Field(ge=0)]
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class AmountRange(BaseModel):
    """Inclusive bounds for one charge, both in the same currency."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    min: Amount
    max: Amount

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min.currency != self.max.currency:
            raise ValueError("amount_range bounds must share one currency")
        if self.min.minor > self.max.minor:
            raise ValueError("amount_range min cannot exceed max")
        return self

    def contains(self, amount: Amount) -> bool:
        """Whether `amount` lies within the bounds. Raises ValueError on another currency."""
        if amount.currency != self.min.currency:
            raise ValueError(f"cannot compare {amount.currency} with {self.min.currency}")
        return self.min.minor <= amount.minor <= self.max.minor


class Budget(BaseModel):
    """How much the agent may request per period, counted in the mandate's time zone."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    limit: Amount
    period: BudgetPeriod


class DailyWindow(BaseModel):
    """Local wall-clock hours when charges may be issued: `start <= t < end`, same day."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    start: time
    end: time

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise ValueError("daily window times carry no tzinfo: the mandate tz applies")
        if self.start >= self.end:
            raise ValueError("daily window start must come before its end (no wrap past midnight)")
        return self

    def contains(self, local_time: time) -> bool:
        """Whether `local_time` (wall clock in the mandate tz) falls inside the window."""
        return self.start <= local_time < self.end


class OpenMandate(BaseModel):
    """What a principal authorized an agent to charge for, within which limits and until when.

    Every amount is in one currency. `receiving_accounts` only lists accounts of the credential
    holder. An empty `allowed_payers` means every payer needs approval.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: Identifier
    principal_id: Identifier
    agent_id: Identifier
    purpose: Identifier
    amount_range: AmountRange
    budget: Budget
    receiving_accounts: frozenset[Identifier]
    allowed_payers: frozenset[Identifier]
    allowed_rails: frozenset[Identifier]
    daily_window: DailyWindow | None
    tz: str
    expires_at: AwareDatetime
    revoked_at: AwareDatetime | None

    @field_validator("tz")
    @classmethod
    def _known_time_zone(cls, value: str) -> str:
        # Exact membership, not ZoneInfo(value): a case-insensitive filesystem (macOS) would load
        # "america/mexico_city", which Linux rejects.
        if value not in _known_time_zones():
            raise ValueError(f"unknown time zone {value!r}")
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.receiving_accounts:
            raise ValueError("receiving_accounts cannot be empty")
        if self.budget.limit.currency != self.currency:
            raise ValueError("budget and amount_range must share one currency")
        return self

    @property
    def currency(self) -> str:
        """The single currency of every amount in this mandate."""
        return self.amount_range.min.currency

    @property
    def zone(self) -> ZoneInfo:
        """The mandate's time zone, for daily windows and budget periods."""
        return ZoneInfo(self.tz)


class ClosedMandate(BaseModel):
    """One concrete charge the agent wants to issue under an open mandate.

    `agent_id` is the agent acting; `charge_ref` is the charge's key in the consumption ledger
    (`ConsumptionEntry.charge_ref`). `amount=None` is an open-amount request (the payer sets
    it). An `amount_increase` raises the pending link `charge_ref` from `previous_amount` to
    `amount`, both present, same currency, and strictly greater; a `new_charge` has no
    `previous_amount`. Fields that come from tool arguments are untrusted: `check` compares
    `agent_id` and `previous_amount` against the mandate and the ledger.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    open_mandate_id: Identifier
    agent_id: Identifier
    kind: ChargeKind
    charge_ref: Identifier
    payer: Identifier
    receiving_account: Identifier
    rail: Identifier
    amount: Amount | None
    previous_amount: Amount | None
    reference: Identifier

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, value in (("amount", self.amount), ("previous_amount", self.previous_amount)):
            if value is not None and value.minor <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.kind == "new_charge":
            if self.previous_amount is not None:
                raise ValueError("a new_charge has no previous_amount")
            return self
        if self.previous_amount is None or self.amount is None:
            raise ValueError("an amount_increase needs both previous_amount and amount")
        if self.amount.currency != self.previous_amount.currency:
            raise ValueError("amount and previous_amount must share one currency")
        if self.amount.minor <= self.previous_amount.minor:
            raise ValueError("an amount_increase needs amount greater than previous_amount")
        return self

    @property
    def content_hash(self) -> str:
        """SHA-256 hex of the canonical JSON: `model_dump(mode="json")`, sorted keys, no spaces."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

"""Budget consumption: what the agent requested in the current period, from a reserve ledger.

Every charge issued reserves what it requested (a link increase reserves only its delta, under
the same `charge_ref`). A charge that expired unpaid is released only once the provider confirms
it, and the release frees the whole ref. Payments never consume: they are income, not requests.
"""

from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Literal, Self, assert_never
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from agent_toolkit.mandates.models import (
    Amount,
    BudgetPeriod,
    Identifier,
    OpenMandate,
    require_aware,
)

EntryKind = Literal["reserve", "release"]
RefStatus = Literal["unknown", "reserved", "released"]


class ConsumptionEntry(BaseModel):
    """One ledger line: budget reserved by a charge, or the release of everything it reserved."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    charge_ref: Identifier
    kind: EntryKind
    amount: Amount
    at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.amount.minor <= 0:
            raise ValueError(f"a consumption entry amount must be positive, got {self.amount}")
        return self


class RefState(BaseModel):
    """What the ledger knows about one `charge_ref`.

    `unknown`: never reserved. `reserved`: `reserved` holds the unreleased total. `released`:
    its reserves were released, so the ref is spent and cannot be reused.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    status: RefStatus
    reserved: Amount | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.status == "reserved") != (self.reserved is not None):
            raise ValueError("a ref state carries an amount exactly when it is reserved")
        return self


def consumed_in_period(
    entries: Iterable[ConsumptionEntry], mandate: OpenMandate, now: datetime
) -> Amount:
    """Sum the unreleased reserves whose `at` falls in the budget period containing `now`.

    `entries` must be this mandate's ledger only. Periods are judged in the mandate's time
    zone: a day runs from local midnight, a week from Monday, a month from day 1; `total`
    ignores time. A released ref contributes nothing, in any period. Raises ValueError when
    `now` is naive, an entry is in another currency, or the ledger is inconsistent (see
    `ref_state`).
    """
    require_aware(now, "now")
    ledger = list(entries)
    for entry in ledger:
        if entry.amount.currency != mandate.currency:
            raise ValueError(
                f"entry {entry.charge_ref!r} currency {entry.amount.currency} differs from "
                f"the mandate currency {mandate.currency}"
            )
    unreleased, _ = _validated_ledger(ledger)
    zone = mandate.zone
    period = mandate.budget.period
    current = _period_start(now, period, zone)
    total = sum(
        entry.amount.minor
        for ref_reserves in unreleased.values()
        for entry in ref_reserves
        if _period_start(entry.at, period, zone) == current
    )
    return Amount(minor=total, currency=mandate.currency)


def ref_state(entries: Iterable[ConsumptionEntry], charge_ref: str) -> RefState:
    """Whether `charge_ref` is unknown, still reserved (and for how much), or released.

    `entries` must be this mandate's ledger only; `check` needs the result to reject a reused
    ref and an increase whose claimed `previous_amount` differs from the reserve. Raises
    ValueError when the ledger is inconsistent anywhere: a release with no reserve, a second
    release of a ref, a release whose amount differs from everything reserved for its ref
    (including a reserve recorded after the release), or a ref in two currencies.
    """
    unreleased, released = _validated_ledger(entries)
    if charge_ref in released:
        return RefState(status="released", reserved=None)
    ref_reserves = unreleased.get(charge_ref)
    if ref_reserves is None:
        return RefState(status="unknown", reserved=None)
    total = sum(entry.amount.minor for entry in ref_reserves)
    amount = Amount(minor=total, currency=ref_reserves[0].amount.currency)
    return RefState(status="reserved", reserved=amount)


def _validated_ledger(
    entries: Iterable[ConsumptionEntry],
) -> tuple[dict[str, list[ConsumptionEntry]], frozenset[str]]:
    """Validate the whole ledger; return the unreleased reserves by ref and the released refs."""
    reserves: defaultdict[str, list[ConsumptionEntry]] = defaultdict(list)
    releases: dict[str, ConsumptionEntry] = {}
    for entry in entries:
        if entry.kind == "reserve":
            reserves[entry.charge_ref].append(entry)
        elif entry.charge_ref in releases:
            raise ValueError(f"charge {entry.charge_ref!r} released twice")
        else:
            releases[entry.charge_ref] = entry

    for ref in releases:
        if ref not in reserves:
            raise ValueError(f"release of charge {ref!r} with no reserve")
    for ref, ref_reserves in reserves.items():
        released_entry = releases.get(ref)
        currencies = {entry.amount.currency for entry in ref_reserves}
        if released_entry is not None:
            currencies.add(released_entry.amount.currency)
        if len(currencies) > 1:
            raise ValueError(f"charge {ref!r} mixes currencies {sorted(currencies)}")
        reserved = sum(entry.amount.minor for entry in ref_reserves)
        if released_entry is not None and released_entry.amount.minor != reserved:
            raise ValueError(
                f"release of charge {ref!r} is {released_entry.amount.minor}, "
                f"but {reserved} was reserved"
            )
    unreleased = {ref: kept for ref, kept in reserves.items() if ref not in releases}
    return unreleased, frozenset(releases)


def _period_start(moment: datetime, period: BudgetPeriod, zone: ZoneInfo) -> date | None:
    """Local date the period containing `moment` starts on; None for `total` (one period)."""
    day = moment.astimezone(zone).date()
    match period:
        case "day":
            return day
        case "week":
            return day - timedelta(days=day.weekday())
        case "month":
            return day.replace(day=1)
        case "total":
            return None
        case _:
            assert_never(period)

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


def consumed_in_period(
    entries: Iterable[ConsumptionEntry], mandate: OpenMandate, now: datetime
) -> Amount:
    """Sum the unreleased reserves whose `at` falls in the budget period containing `now`.

    Periods are judged in the mandate's time zone: a day runs from local midnight, a week from
    Monday, a month from day 1; `total` ignores time. A released ref contributes nothing, in any
    period. Raises ValueError when `now` is naive, an entry is in another currency, or the
    ledger is inconsistent (see `reserved_for`).
    """
    require_aware(now, "now")
    ledger = list(entries)
    for entry in ledger:
        if entry.amount.currency != mandate.currency:
            raise ValueError(
                f"entry {entry.charge_ref!r} currency {entry.amount.currency} differs from "
                f"the mandate currency {mandate.currency}"
            )
    zone = mandate.zone
    period = mandate.budget.period
    current = _period_start(now, period, zone)
    total = sum(
        entry.amount.minor
        for ref_reserves in _unreleased_reserves(ledger).values()
        for entry in ref_reserves
        if _period_start(entry.at, period, zone) == current
    )
    return Amount(minor=total, currency=mandate.currency)


def reserved_for(entries: Iterable[ConsumptionEntry], charge_ref: str) -> Amount | None:
    """Total still reserved for `charge_ref`, or None if it has no reserve or was released.

    `check` compares it with an increase's claimed `previous_amount`. Raises ValueError when the
    ledger is inconsistent anywhere: a release with no reserve, a second release of a ref, a
    release whose amount differs from everything reserved for its ref, or a ref in two
    currencies.
    """
    ref_reserves = _unreleased_reserves(entries).get(charge_ref)
    if ref_reserves is None:
        return None
    total = sum(entry.amount.minor for entry in ref_reserves)
    return Amount(minor=total, currency=ref_reserves[0].amount.currency)


def _unreleased_reserves(
    entries: Iterable[ConsumptionEntry],
) -> dict[str, list[ConsumptionEntry]]:
    """Validate the whole ledger and return the reserves of every ref not released, by ref."""
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
    return {ref: ref_reserves for ref, ref_reserves in reserves.items() if ref not in releases}


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

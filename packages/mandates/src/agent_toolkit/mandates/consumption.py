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
    ledger is inconsistent: a release with no reserve, a second release of a ref, or a release
    whose amount differs from everything reserved for its ref.
    """
    require_aware(now, "now")
    reserves: defaultdict[str, list[ConsumptionEntry]] = defaultdict(list)
    releases: dict[str, ConsumptionEntry] = {}
    for entry in entries:
        if entry.amount.currency != mandate.currency:
            raise ValueError(
                f"entry {entry.charge_ref!r} currency {entry.amount.currency} differs from "
                f"the mandate currency {mandate.currency}"
            )
        if entry.kind == "reserve":
            reserves[entry.charge_ref].append(entry)
        elif entry.charge_ref in releases:
            raise ValueError(f"charge {entry.charge_ref!r} released twice")
        else:
            releases[entry.charge_ref] = entry

    for ref, released in releases.items():
        if ref not in reserves:
            raise ValueError(f"release of charge {ref!r} with no reserve")
        reserved = sum(entry.amount.minor for entry in reserves[ref])
        if released.amount.minor != reserved:
            raise ValueError(
                f"release of charge {ref!r} is {released.amount.minor}, but {reserved} was reserved"
            )

    zone = mandate.zone
    period = mandate.budget.period
    current = _period_start(now, period, zone)
    total = sum(
        entry.amount.minor
        for ref, ref_reserves in reserves.items()
        if ref not in releases
        for entry in ref_reserves
        if _period_start(entry.at, period, zone) == current
    )
    return Amount(minor=total, currency=mandate.currency)


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

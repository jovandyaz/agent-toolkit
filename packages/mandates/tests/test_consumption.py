"""`consumed_in_period`: unreleased reserves in the current budget period, in the mandate tz."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    BudgetPeriod,
    ConsumptionEntry,
    OpenMandate,
    consumed_in_period,
)

MEXICO_CITY = ZoneInfo("America/Mexico_City")
NOW = datetime(2026, 12, 2, 12, 0, tzinfo=MEXICO_CITY)  # Wednesday


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


def local(day: int, hour: int, minute: int = 0, month: int = 12, year: int = 2026) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MEXICO_CITY)


def mandate(period: BudgetPeriod = "day") -> OpenMandate:
    return OpenMandate(
        id="om_1",
        principal_id="owner_1",
        agent_id="agent_1",
        purpose="Collect payments from customers of the shop",
        amount_range=AmountRange(min=mxn(100), max=mxn(500000)),
        budget=Budget(limit=mxn(1000000), period=period),
        receiving_accounts=frozenset({"acct-own"}),
        allowed_payers=frozenset({"cust-juan"}),
        allowed_rails=frozenset({"codi_push", "codi_link"}),
        daily_window=None,
        tz="America/Mexico_City",
        expires_at=NOW + timedelta(days=365),
        revoked_at=None,
    )


def reserve(ref: str, minor: int, at: datetime = NOW) -> ConsumptionEntry:
    return ConsumptionEntry(charge_ref=ref, kind="reserve", amount=mxn(minor), at=at)


def release(ref: str, minor: int, at: datetime = NOW) -> ConsumptionEntry:
    return ConsumptionEntry(charge_ref=ref, kind="release", amount=mxn(minor), at=at)


def test_nothing_consumed_is_zero_in_the_mandate_currency() -> None:
    assert consumed_in_period([], mandate(), NOW) == mxn(0)


def test_reserves_add_up() -> None:
    entries = [reserve("ch_1", 30000), reserve("ch_2", 20000)]

    assert consumed_in_period(entries, mandate(), NOW) == mxn(50000)


def test_release_only_subtracts_matching_reserve() -> None:
    entries = [reserve("ch_1", 30000), release("ch_1", 30000)]
    assert consumed_in_period(entries, mandate(), NOW) == mxn(0)

    entries = [reserve("ch_1", 30000), reserve("ch_2", 20000), release("ch_1", 30000)]
    assert consumed_in_period(entries, mandate(), NOW) == mxn(20000)

    with pytest.raises(ValueError, match="no reserve"):
        consumed_in_period([reserve("ch_1", 30000), release("ch_2", 30000)], mandate(), NOW)


def test_release_frees_every_reserve_of_an_increased_link() -> None:
    increased = [reserve("link_1", 40000), reserve("link_1", 10000)]
    assert consumed_in_period(increased, mandate(), NOW) == mxn(50000)

    released = [*increased, release("link_1", 50000)]
    assert consumed_in_period(released, mandate(), NOW) == mxn(0)


def test_release_order_in_the_ledger_does_not_matter() -> None:
    entries = [release("ch_1", 30000), reserve("ch_1", 30000)]

    assert consumed_in_period(entries, mandate(), NOW) == mxn(0)


@pytest.mark.parametrize("released", [40000, 10000, 50001])
def test_release_must_equal_everything_reserved_for_the_ref(released: int) -> None:
    entries = [reserve("link_1", 40000), reserve("link_1", 10000), release("link_1", released)]

    with pytest.raises(ValueError, match="reserved"):
        consumed_in_period(entries, mandate(), NOW)


def test_second_release_of_a_ref_is_rejected() -> None:
    entries = [reserve("ch_1", 30000), release("ch_1", 30000), release("ch_1", 30000)]

    with pytest.raises(ValueError, match="released twice"):
        consumed_in_period(entries, mandate(), NOW)


@pytest.mark.parametrize("kind", ["reserve", "release"])
def test_entry_in_another_currency_is_rejected(kind: str) -> None:
    usd = ConsumptionEntry.model_validate(
        {"charge_ref": "ch_1", "kind": kind, "amount": Amount(minor=1, currency="USD"), "at": NOW}
    )

    with pytest.raises(ValueError, match="currency"):
        consumed_in_period([usd], mandate(), NOW)


def test_entries_may_be_a_one_shot_iterator() -> None:
    entries = iter([reserve("ch_1", 30000), release("ch_1", 30000), reserve("ch_2", 5000)])

    assert consumed_in_period(entries, mandate(), NOW) == mxn(5000)


def test_daily_period_uses_mandate_timezone() -> None:
    # 23:30 on Dec 1 local is 05:30 UTC on Dec 2, the same UTC day as 00:30 on Dec 2 local.
    late = reserve("ch_1", 30000, at=local(1, 23, 30))
    after_midnight = local(2, 0, 30)

    assert consumed_in_period([late], mandate("day"), after_midnight) == mxn(0)
    early = reserve("ch_2", 20000, at=local(2, 0, 10))
    assert consumed_in_period([late, early], mandate("day"), after_midnight) == mxn(20000)


def test_daily_period_ignores_the_utc_date() -> None:
    # 18:30 on Dec 1 local is 00:30 UTC on Dec 2: same UTC day as NOW, a different local day.
    evening = reserve("ch_1", 30000, at=datetime(2026, 12, 2, 0, 30, tzinfo=UTC))

    assert consumed_in_period([evening], mandate("day"), NOW) == mxn(0)


def test_released_reserves_from_another_period_do_not_count() -> None:
    entries = [reserve("ch_1", 30000, at=local(1, 12)), release("ch_1", 30000)]

    assert consumed_in_period(entries, mandate("day"), NOW) == mxn(0)


def test_weekly_period_starts_on_monday() -> None:
    sunday = reserve("ch_1", 10000, at=local(29, 23, 59, month=11))
    monday = reserve("ch_2", 20000, at=local(30, 0, 0, month=11))
    sunday_next = reserve("ch_3", 40000, at=local(6, 23, 59))

    entries = [sunday, monday, sunday_next]

    assert consumed_in_period(entries, mandate("week"), NOW) == mxn(60000)


def test_monthly_period_starts_on_day_one() -> None:
    last_of_november = reserve("ch_1", 10000, at=local(30, 23, 30, month=11))
    first_of_december = reserve("ch_2", 20000, at=local(1, 0, 0))

    entries = [last_of_november, first_of_december]

    assert consumed_in_period(entries, mandate("month"), local(1, 0, 30)) == mxn(20000)


def test_total_period_ignores_time() -> None:
    entries = [
        reserve("ch_1", 10000, at=local(15, 9, month=1, year=2025)),
        reserve("ch_2", 20000, at=local(30, 23, 59, month=11)),
        reserve("ch_3", 30000, at=NOW),
        release("ch_2", 20000, at=NOW),
    ]

    assert consumed_in_period(entries, mandate("total"), NOW) == mxn(40000)


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="aware"):
        consumed_in_period([], mandate(), datetime(2026, 12, 2, 12, 0))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"amount": mxn(0)}, "positive"),
        ({"at": datetime(2026, 12, 2, 12, 0)}, "timezone"),
        ({"charge_ref": ""}, "at least 1 character"),
        ({"kind": "refund"}, "kind"),
    ],
)
def test_entry_rejects_invalid_fields(overrides: dict[str, object], message: str) -> None:
    fields: dict[str, object] = {
        "charge_ref": "ch_1",
        "kind": "reserve",
        "amount": mxn(1),
        "at": NOW,
    }

    with pytest.raises(ValidationError, match=message):
        ConsumptionEntry.model_validate(fields | overrides)

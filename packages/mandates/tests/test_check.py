"""`check`: does a concrete charge (closed mandate) fit what the owner authorized (open mandate)?"""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    DailyWindow,
    OpenMandate,
    Verdict,
    check,
    reserve_amount,
)

MEXICO_CITY = ZoneInfo("America/Mexico_City")
NOW = datetime(2026, 12, 1, 12, 0, tzinfo=MEXICO_CITY)


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


def usd(minor: int) -> Amount:
    return Amount(minor=minor, currency="USD")


def mandate(**overrides: object) -> OpenMandate:
    fields: dict[str, object] = {
        "id": "om_1",
        "principal_id": "owner_1",
        "agent_id": "agent_1",
        "purpose": "Collect payments from customers of the shop",
        "amount_range": AmountRange(min=mxn(100), max=mxn(500000)),
        "budget": Budget(limit=mxn(1000000), period="day"),
        "receiving_accounts": frozenset({"acct-own"}),
        "allowed_payers": frozenset({"cust-juan"}),
        "allowed_rails": frozenset({"codi_push", "codi_link"}),
        "daily_window": None,
        "tz": "America/Mexico_City",
        "expires_at": NOW + timedelta(days=30),
        "revoked_at": None,
    }
    return OpenMandate.model_validate(fields | overrides)


def charge(**overrides: object) -> ClosedMandate:
    fields: dict[str, object] = {
        "open_mandate_id": "om_1",
        "kind": "new_charge",
        "payer": "cust-juan",
        "receiving_account": "acct-own",
        "rail": "codi_push",
        "amount": mxn(35000),
        "previous_amount": None,
        "reference": "R-1001",
    }
    return ClosedMandate.model_validate(fields | overrides)


def increase(previous: int, new: int) -> ClosedMandate:
    return charge(
        kind="amount_increase", rail="codi_link", previous_amount=mxn(previous), amount=mxn(new)
    )


OK = Verdict(violations=(), approval_reasons=())


def test_in_mandate_charge_is_ok() -> None:
    verdict = check(charge(), mandate(), consumed=mxn(0), now=NOW)

    assert verdict == OK
    assert verdict.ok


def test_foreign_receiving_account_is_violation() -> None:
    verdict = check(charge(receiving_account="acct-other"), mandate(), mxn(0), NOW)

    assert verdict.violations == ("account_not_allowed",)
    assert not verdict.ok


def test_rail_outside_mandate_is_violation() -> None:
    verdict = check(charge(rail="spei"), mandate(), mxn(0), NOW)

    assert verdict.violations == ("rail_not_allowed",)


def test_unknown_payer_needs_approval() -> None:
    verdict = check(charge(payer="cust-ana"), mandate(), mxn(0), NOW)

    assert verdict.approval_reasons == ("payer_not_allowed",)
    assert verdict.violations == ()
    assert not verdict.ok


def test_empty_allowed_payers_means_every_payer_needs_approval() -> None:
    verdict = check(charge(), mandate(allowed_payers=frozenset()), mxn(0), NOW)

    assert verdict.approval_reasons == ("payer_not_allowed",)
    assert verdict.violations == ()


def test_open_amount_always_needs_approval() -> None:
    verdict = check(charge(amount=None), mandate(), mxn(0), NOW)

    assert verdict.approval_reasons == ("open_amount",)
    assert verdict.violations == ()
    assert not verdict.ok


def test_open_amount_skips_range_and_budget_but_not_the_rest() -> None:
    spent = mandate().budget.limit

    verdict = check(charge(amount=None, rail="spei", payer="cust-ana"), mandate(), spent, NOW)

    assert verdict.violations == ("rail_not_allowed",)
    assert verdict.approval_reasons == ("payer_not_allowed", "open_amount")


def test_open_amount_reserves_nothing() -> None:
    assert reserve_amount(charge(amount=None)) is None


def test_new_charge_reserves_its_amount() -> None:
    assert reserve_amount(charge(amount=mxn(35000))) == mxn(35000)


def test_increase_consumes_delta_and_total_must_fit_range() -> None:
    within = increase(previous=40000, new=50000)
    assert reserve_amount(within) == mxn(10000)
    assert check(within, mandate(), mxn(0), NOW) == OK

    beyond = increase(previous=40000, new=600000)
    assert check(beyond, mandate(), mxn(0), NOW).violations == ("amount_out_of_range",)


def test_increase_charges_only_the_delta_against_the_budget() -> None:
    # 990000 consumed already includes the link's previous 40000; the delta of 10000 fills it.
    assert check(increase(previous=40000, new=50000), mandate(), mxn(990000), NOW) == OK
    over = check(increase(previous=40000, new=50001), mandate(), mxn(990000), NOW)
    assert over.violations == ("budget_exceeded",)


@pytest.mark.parametrize(
    ("minor", "fits"), [(99, False), (100, True), (500000, True), (500001, False)]
)
def test_amount_range_bounds_are_inclusive(minor: int, fits: bool) -> None:
    verdict = check(charge(amount=mxn(minor)), mandate(), mxn(0), NOW)

    assert verdict.violations == (() if fits else ("amount_out_of_range",))


def test_budget_counts_requested_amount() -> None:
    verdict = check(charge(amount=mxn(20000)), mandate(), consumed=mxn(990000), now=NOW)

    assert verdict.violations == ("budget_exceeded",)


def test_budget_may_be_filled_exactly() -> None:
    assert check(charge(amount=mxn(20000)), mandate(), mxn(980000), NOW) == OK


def test_expired_revoked_and_mismatch() -> None:
    assert check(charge(), mandate(expires_at=NOW), mxn(0), NOW).violations == ("expired",)
    assert check(charge(), mandate(expires_at=NOW + timedelta(seconds=1)), mxn(0), NOW) == OK

    assert check(charge(), mandate(revoked_at=NOW), mxn(0), NOW).violations == ("revoked",)
    revoked_later = mandate(revoked_at=NOW + timedelta(seconds=1))
    assert check(charge(), revoked_later, mxn(0), NOW) == OK

    other = check(charge(open_mandate_id="om_2"), mandate(), mxn(0), NOW)
    assert other.violations == ("mandate_mismatch",)


def test_every_violation_is_reported_not_only_the_first() -> None:
    bad = charge(
        open_mandate_id="om_2",
        receiving_account="acct-other",
        rail="spei",
        payer="cust-ana",
        amount=mxn(600000),
    )
    window = DailyWindow(start=time(7, 0), end=time(22, 0))
    late_night = datetime(2026, 12, 1, 23, 0, tzinfo=MEXICO_CITY)
    gone = mandate(
        daily_window=window, expires_at=late_night, revoked_at=late_night - timedelta(hours=1)
    )

    verdict = check(bad, gone, consumed=mxn(500000), now=late_night)

    assert verdict.violations == (
        "expired",
        "revoked",
        "mandate_mismatch",
        "account_not_allowed",
        "rail_not_allowed",
        "amount_out_of_range",
        "budget_exceeded",
        "outside_window",
    )
    assert verdict.approval_reasons == ("payer_not_allowed",)


def test_outside_daily_window() -> None:
    windowed = mandate(daily_window=DailyWindow(start=time(7, 0), end=time(22, 0)))

    def at(hour: int, minute: int) -> datetime:
        return datetime(2026, 12, 1, hour, minute, tzinfo=MEXICO_CITY)

    assert check(charge(), windowed, mxn(0), at(21, 59)) == OK
    assert check(charge(), windowed, mxn(0), at(22, 1)).violations == ("outside_window",)
    assert check(charge(), windowed, mxn(0), at(22, 0)).violations == ("outside_window",)
    assert check(charge(), windowed, mxn(0), at(7, 0)) == OK
    assert check(charge(), windowed, mxn(0), at(6, 59)).violations == ("outside_window",)


def test_daily_window_is_judged_in_the_mandate_timezone() -> None:
    windowed = mandate(daily_window=DailyWindow(start=time(7, 0), end=time(22, 0)))
    # 03:59 UTC on Dec 2 is 21:59 on Dec 1 in Mexico City (UTC-6); 04:01 UTC is 22:01 there.
    inside = datetime(2026, 12, 2, 3, 59, tzinfo=UTC)
    outside = datetime(2026, 12, 2, 4, 1, tzinfo=UTC)

    assert check(charge(), windowed, mxn(0), inside) == OK
    assert check(charge(), windowed, mxn(0), outside).violations == ("outside_window",)


def test_charge_in_another_currency_is_mismatch_not_range_or_budget() -> None:
    verdict = check(charge(amount=usd(600000)), mandate(), mxn(990000), NOW)

    assert verdict.violations == ("currency_mismatch",)


def test_increase_in_another_currency_is_mismatch() -> None:
    usd_increase = charge(kind="amount_increase", previous_amount=usd(100), amount=usd(200))

    verdict = check(usd_increase, mandate(), mxn(0), NOW)

    assert verdict.violations == ("currency_mismatch",)


def test_consumed_in_another_currency_is_mismatch() -> None:
    verdict = check(charge(), mandate(), consumed=usd(0), now=NOW)

    assert verdict.violations == ("currency_mismatch",)


def test_open_amount_still_flags_consumed_in_another_currency() -> None:
    verdict = check(charge(amount=None), mandate(), consumed=usd(0), now=NOW)

    assert verdict.violations == ("currency_mismatch",)
    assert verdict.approval_reasons == ("open_amount",)


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="aware"):
        check(charge(), mandate(), mxn(0), datetime(2026, 12, 1, 12, 0))

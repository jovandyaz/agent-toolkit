"""Invariants the models enforce, so no caller can build an inconsistent mandate or charge."""

from datetime import UTC, datetime, time, timedelta

import pytest
from pydantic import ValidationError

from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    DailyWindow,
    OpenMandate,
)

T = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


def usd(minor: int) -> Amount:
    return Amount(minor=minor, currency="USD")


def mandate_fields(**overrides: object) -> dict[str, object]:
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
        "expires_at": T + timedelta(days=30),
        "revoked_at": None,
    }
    return fields | overrides


def charge_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "open_mandate_id": "om_1",
        "agent_id": "agent_1",
        "kind": "new_charge",
        "charge_ref": "ch_1",
        "payer": "cust-juan",
        "receiving_account": "acct-own",
        "rail": "codi_push",
        "amount": mxn(35000),
        "previous_amount": None,
        "reference": "R-1001",
    }
    return fields | overrides


@pytest.mark.parametrize(
    "fields",
    [
        {"minor": -1, "currency": "MXN"},
        {"minor": 1.5, "currency": "MXN"},
        {"minor": "100", "currency": "MXN"},
        {"minor": True, "currency": "MXN"},
        {"minor": 100, "currency": "mxn"},
        {"minor": 100, "currency": "PESOS"},
    ],
)
def test_amount_is_non_negative_integer_minor_units_of_an_iso_code(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Amount.model_validate(fields)


def test_a_consistent_mandate_is_accepted() -> None:
    mandate = OpenMandate.model_validate(mandate_fields())

    assert mandate.currency == "MXN"


def test_amount_range_rejects_min_above_max() -> None:
    with pytest.raises(ValidationError, match="min"):
        AmountRange(min=mxn(500001), max=mxn(500000))


def test_amount_range_rejects_mixed_currencies() -> None:
    with pytest.raises(ValidationError, match="currency"):
        AmountRange(min=mxn(100), max=usd(500000))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"receiving_accounts": frozenset()}, "receiving_accounts"),
        ({"budget": Budget(limit=usd(1000000), period="day")}, "currency"),
        ({"tz": "Mars/Olympus"}, "time zone"),
        ({"tz": ""}, "time zone"),
        ({"tz": "../etc/passwd"}, "time zone"),
        ({"expires_at": datetime(2027, 1, 1)}, "timezone"),
        ({"revoked_at": datetime(2026, 12, 2)}, "timezone"),
        ({"receiving_accounts": frozenset({""})}, "at least 1 character"),
        ({"allowed_rails": {"codi_push"}}, "frozenset"),
    ],
)
def test_mandate_rejects_invalid_constraints(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        OpenMandate.model_validate(mandate_fields(**overrides))


@pytest.mark.parametrize(
    "tz", ["america/mexico_city", "AMERICA/MEXICO_CITY", "America/Mexico_city"]
)
def test_mandate_rejects_a_wrong_case_time_zone_on_every_os(tz: str) -> None:
    # Case-insensitive filesystems (macOS) would load these; Linux would not.
    with pytest.raises(ValidationError, match="time zone"):
        OpenMandate.model_validate(mandate_fields(tz=tz))


@pytest.mark.parametrize(
    ("start", "end"),
    [(time(22, 0), time(7, 0)), (time(7, 0), time(7, 0))],
)
def test_daily_window_must_start_before_it_ends(start: time, end: time) -> None:
    with pytest.raises(ValidationError, match="start"):
        DailyWindow(start=start, end=end)


def test_daily_window_times_are_local_wall_clock_times() -> None:
    with pytest.raises(ValidationError, match="tz"):
        DailyWindow(start=time(7, 0, tzinfo=UTC), end=time(22, 0))


def test_a_consistent_charge_is_accepted() -> None:
    closed = ClosedMandate.model_validate(charge_fields())

    assert closed.kind == "new_charge"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"previous_amount": mxn(100)}, "new_charge"),
        ({"kind": "amount_increase"}, "previous_amount"),
        ({"kind": "amount_increase", "previous_amount": mxn(100), "amount": None}, "amount"),
        ({"kind": "amount_increase", "previous_amount": mxn(35000)}, "greater"),
        ({"kind": "amount_increase", "previous_amount": mxn(40000)}, "greater"),
        ({"kind": "amount_increase", "previous_amount": usd(100)}, "currency"),
        ({"amount": mxn(0)}, "positive"),
        ({"kind": "amount_increase", "previous_amount": mxn(0)}, "positive"),
        ({"payer": ""}, "at least 1 character"),
        ({"agent_id": ""}, "at least 1 character"),
        ({"charge_ref": ""}, "at least 1 character"),
        ({"kind": "refund"}, "kind"),
    ],
)
def test_charge_rejects_inconsistent_amounts(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ClosedMandate.model_validate(charge_fields(**overrides))


def test_models_are_frozen() -> None:
    closed = ClosedMandate.model_validate(charge_fields())

    with pytest.raises(ValidationError):
        closed.amount = mxn(1)  # type: ignore[misc]


@pytest.mark.parametrize("field", ["agent_id", "charge_ref"])
def test_charge_names_its_agent_and_ledger_ref(field: str) -> None:
    fields = charge_fields()
    del fields[field]

    with pytest.raises(ValidationError, match=field):
        ClosedMandate.model_validate(fields)

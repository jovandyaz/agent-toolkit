import pytest
from pydantic import ValidationError

from agent_toolkit.payments_core import CurrencyMismatchError, Money


def mxn(amount_minor: int) -> Money:
    return Money(amount_minor=amount_minor, currency="MXN")


def test_money_rejects_float_and_bad_currency() -> None:
    with pytest.raises(ValidationError):
        Money(amount_minor=1.5, currency="MXN")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Money(amount_minor=100, currency="mxn")


# pialy: domain.test.ts > money > "rejects %s" (1.5, -1, NaN, Infinity). Its
# MAX_SAFE_INTEGER + 1 case does not apply: Python integers are exact at any size.
@pytest.mark.parametrize("amount_minor", [1.0, True, -1, float("nan"), float("inf")])
def test_money_rejects_non_integer_or_negative_amounts(amount_minor: object) -> None:
    with pytest.raises(ValidationError):
        Money(amount_minor=amount_minor, currency="MXN")  # type: ignore[arg-type]


@pytest.mark.parametrize("currency", ["MX", "MXNN", "M1N", " MXN", "MXN\n", ""])
def test_money_requires_a_three_letter_uppercase_currency(currency: str) -> None:
    with pytest.raises(ValidationError):
        Money(amount_minor=1, currency=currency)


def test_money_allows_zero() -> None:
    # pialy: domain.test.ts > money > "builds MXN money from integer minor units"
    assert mxn(0).amount_minor == 0


def test_money_add_requires_same_currency() -> None:
    assert mxn(100) + mxn(1) == mxn(101)
    with pytest.raises(CurrencyMismatchError):
        mxn(1) + Money(amount_minor=1, currency="USD")


def test_money_subtract_requires_same_currency_and_non_negative_result() -> None:
    assert mxn(100) - mxn(1) == mxn(99)
    assert mxn(1) - mxn(1) == mxn(0)
    with pytest.raises(CurrencyMismatchError):
        mxn(1) - Money(amount_minor=1, currency="USD")
    with pytest.raises(ValueError, match="negative"):
        mxn(1) - mxn(2)


def test_money_is_immutable() -> None:
    money = mxn(1)
    with pytest.raises(ValidationError):
        money.amount_minor = 2  # type: ignore[misc]


def test_currency_mismatch_error_is_a_value_error_with_details() -> None:
    # pialy: domain.test.ts > domain errors > "are Error subclasses with a stable name"
    with pytest.raises(CurrencyMismatchError) as caught:
        mxn(1) + Money(amount_minor=1, currency="USD")

    assert isinstance(caught.value, ValueError)
    assert (caught.value.expected, caught.value.received) == ("MXN", "USD")

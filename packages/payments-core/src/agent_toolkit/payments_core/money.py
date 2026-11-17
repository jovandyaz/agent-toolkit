"""Money as an integer amount in minor units plus an ISO 4217 currency code. Never a float."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class CurrencyMismatchError(ValueError):
    """Two amounts in different currencies were combined."""

    def __init__(self, expected: str, received: str) -> None:
        super().__init__(f"currency mismatch: expected {expected}, got {received}")
        self.expected = expected
        self.received = received


class Money(BaseModel):
    """A non-negative amount in minor units (centavos for MXN) of one currency."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    amount_minor: Annotated[int, Field(ge=0)]
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]

    def __add__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        return Money(amount_minor=self.amount_minor + other.amount_minor, currency=self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        difference = self.amount_minor - other.amount_minor
        if difference < 0:
            raise ValueError(f"money cannot be negative: {self} - {other}")
        return Money(amount_minor=difference, currency=self.currency)

    def _require_same_currency(self, other: "Money") -> None:
        if other.currency != self.currency:
            raise CurrencyMismatchError(self.currency, other.currency)

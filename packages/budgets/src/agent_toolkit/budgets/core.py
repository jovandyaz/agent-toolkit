"""Run budgets: caps on iterations, tool calls, tokens, USD cost and consecutive failures.

Pure functions over immutable state, so any runtime can drive them (OWASP Agentic ASI08,
runaway agents). `charge` returns the new state after a usage; `check` lists the limits that
state exceeds. A limit is exceeded only when the value is strictly greater than it.
"""

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

ExceededKind = Literal["iterations", "tool_calls", "tokens", "cost", "consecutive_failures"]

_PER_MTOK = Decimal(1_000_000)

Count = Annotated[int, Field(ge=0)]
UsdAmount = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class BudgetExceededError(Exception):
    """Raised by runtime adapters when a run exceeds its limits; `kinds` lists which ones."""

    def __init__(self, kinds: tuple[ExceededKind, ...]) -> None:
        super().__init__(f"budget exceeded: {', '.join(kinds)}")
        self.kinds = kinds


class BudgetLimits(BaseModel):
    """Caps for one run; `None` means unlimited. `max_tokens` bounds input plus output tokens."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    max_iterations: Count | None = None
    max_tool_calls: Count | None = None
    max_tokens: Count | None = None
    max_cost_usd: UsdAmount | None = None
    max_consecutive_failures: Count | None = None


class Pricing(BaseModel):
    """USD price per million tokens."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    input_per_mtok_usd: UsdAmount
    output_per_mtok_usd: UsdAmount


class Usage(BaseModel):
    """What one step consumed.

    `failed` True counts a failure, False resets the streak, None leaves it.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    iterations: Count = 0
    tool_calls: Count = 0
    input_tokens: Count = 0
    output_tokens: Count = 0
    failed: bool | None = None


class BudgetState(BaseModel):
    """Totals consumed so far by a run."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    iterations: Count
    tool_calls: Count
    input_tokens: Count
    output_tokens: Count
    cost_usd: UsdAmount
    consecutive_failures: Count

    @classmethod
    def empty(cls) -> Self:
        """A run that has consumed nothing."""
        return cls(
            iterations=0,
            tool_calls=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal(0),
            consecutive_failures=0,
        )


def charge(state: BudgetState, usage: Usage, pricing: Pricing | None) -> BudgetState:
    """Return `state` plus `usage`.

    Cost is exact `Decimal` arithmetic, never rounded: tokens / 1_000_000 times the per-million
    price. With `pricing=None` cost stays unchanged, so a cost limit without pricing never trips.
    """
    cost = state.cost_usd
    if pricing is not None:
        cost += (
            Decimal(usage.input_tokens) / _PER_MTOK * pricing.input_per_mtok_usd
            + Decimal(usage.output_tokens) / _PER_MTOK * pricing.output_per_mtok_usd
        )
    if usage.failed is None:
        failures = state.consecutive_failures
    elif usage.failed:
        failures = state.consecutive_failures + 1
    else:
        failures = 0
    return BudgetState(
        iterations=state.iterations + usage.iterations,
        tool_calls=state.tool_calls + usage.tool_calls,
        input_tokens=state.input_tokens + usage.input_tokens,
        output_tokens=state.output_tokens + usage.output_tokens,
        cost_usd=cost,
        consecutive_failures=failures,
    )


def check(state: BudgetState, limits: BudgetLimits) -> tuple[ExceededKind, ...]:
    """The exceeded limits, in fixed order (iterations, tool_calls, tokens, cost,
    consecutive_failures). Empty means the run is within budget."""
    exceeded: list[ExceededKind] = []
    if limits.max_iterations is not None and state.iterations > limits.max_iterations:
        exceeded.append("iterations")
    if limits.max_tool_calls is not None and state.tool_calls > limits.max_tool_calls:
        exceeded.append("tool_calls")
    if (
        limits.max_tokens is not None
        and state.input_tokens + state.output_tokens > limits.max_tokens
    ):
        exceeded.append("tokens")
    if limits.max_cost_usd is not None and state.cost_usd > limits.max_cost_usd:
        exceeded.append("cost")
    if (
        limits.max_consecutive_failures is not None
        and state.consecutive_failures > limits.max_consecutive_failures
    ):
        exceeded.append("consecutive_failures")
    return tuple(exceeded)

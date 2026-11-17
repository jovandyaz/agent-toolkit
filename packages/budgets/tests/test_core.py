from decimal import Decimal

import pytest
from pydantic import ValidationError

from agent_toolkit.budgets import BudgetLimits, BudgetState, Pricing, Usage, charge, check

PRICING = Pricing(input_per_mtok_usd=Decimal("3"), output_per_mtok_usd=Decimal("15"))


def test_empty_state_is_zero() -> None:
    state = BudgetState.empty()
    assert (state.iterations, state.tool_calls, state.input_tokens, state.output_tokens) == (0,) * 4
    assert state.cost_usd == Decimal(0)
    assert state.consecutive_failures == 0


def test_limits_default_to_unlimited() -> None:
    huge = BudgetState(
        iterations=10**9,
        tool_calls=10**9,
        input_tokens=10**9,
        output_tokens=10**9,
        cost_usd=Decimal(10**9),
        consecutive_failures=10**9,
    )
    assert check(huge, BudgetLimits()) == ()


def test_charge_accumulates_counts_and_tokens() -> None:
    state = charge(BudgetState.empty(), Usage(iterations=1, input_tokens=10, output_tokens=5), None)
    state = charge(state, Usage(tool_calls=2, input_tokens=1), None)
    assert (state.iterations, state.tool_calls) == (1, 2)
    assert (state.input_tokens, state.output_tokens) == (11, 5)


def test_charge_does_not_mutate_the_previous_state() -> None:
    before = BudgetState.empty()
    charge(before, Usage(iterations=1), None)
    assert before == BudgetState.empty()


def test_cost_is_exact_decimal_per_million_tokens() -> None:
    state = charge(
        BudgetState.empty(), Usage(input_tokens=1_234_567, output_tokens=7_654_321), PRICING
    )
    assert (
        state.cost_usd == Decimal(1_234_567) / 1_000_000 * 3 + Decimal(7_654_321) / 1_000_000 * 15
    )
    assert state.cost_usd == Decimal("118.518516")


def test_cost_accumulates_across_charges() -> None:
    state = charge(BudgetState.empty(), Usage(input_tokens=500_000), PRICING)
    state = charge(state, Usage(output_tokens=100_000), PRICING)
    assert state.cost_usd == Decimal("3.0")


def test_no_pricing_leaves_cost_unchanged() -> None:
    state = charge(BudgetState.empty(), Usage(input_tokens=1_000_000), None)
    assert state.input_tokens == 1_000_000
    assert state.cost_usd == Decimal(0)
    assert check(state, BudgetLimits(max_cost_usd=Decimal(0))) == ()


def test_failed_true_increments_and_false_resets() -> None:
    state = charge(BudgetState.empty(), Usage(failed=True), None)
    state = charge(state, Usage(failed=True), None)
    assert state.consecutive_failures == 2
    assert charge(state, Usage(failed=False), None).consecutive_failures == 0


def test_failed_none_leaves_consecutive_failures() -> None:
    state = charge(BudgetState.empty(), Usage(failed=True), None)
    assert charge(state, Usage(iterations=1), None).consecutive_failures == 1


@pytest.mark.parametrize(
    ("at_limit", "over", "limits", "kind"),
    [
        ({"iterations": 3}, {"iterations": 4}, {"max_iterations": 3}, "iterations"),
        ({"tool_calls": 3}, {"tool_calls": 4}, {"max_tool_calls": 3}, "tool_calls"),
        (
            {"input_tokens": 2, "output_tokens": 1},
            {"input_tokens": 3, "output_tokens": 1},
            {"max_tokens": 3},
            "tokens",
        ),
        (
            {"cost_usd": Decimal("1.5")},
            {"cost_usd": Decimal("1.500001")},
            {"max_cost_usd": Decimal("1.5")},
            "cost",
        ),
        (
            {"consecutive_failures": 3},
            {"consecutive_failures": 4},
            {"max_consecutive_failures": 3},
            "consecutive_failures",
        ),
    ],
)
def test_limit_reached_exactly_is_not_exceeded_but_plus_one_is(
    at_limit: dict[str, int | Decimal],
    over: dict[str, int | Decimal],
    limits: dict[str, int | Decimal],
    kind: str,
) -> None:
    budget = BudgetLimits.model_validate(limits)
    assert check(BudgetState.empty().model_copy(update=at_limit), budget) == ()
    assert check(BudgetState.empty().model_copy(update=over), budget) == (kind,)


def test_several_kinds_reported_together_in_fixed_order() -> None:
    state = BudgetState(
        iterations=9,
        tool_calls=9,
        input_tokens=9,
        output_tokens=9,
        cost_usd=Decimal(9),
        consecutive_failures=9,
    )
    limits = BudgetLimits(
        max_consecutive_failures=1,
        max_cost_usd=Decimal(1),
        max_tokens=1,
        max_tool_calls=1,
        max_iterations=1,
    )
    assert check(state, limits) == (
        "iterations",
        "tool_calls",
        "tokens",
        "cost",
        "consecutive_failures",
    )


def test_models_reject_negative_values_and_floats() -> None:
    with pytest.raises(ValidationError):
        Usage(iterations=-1)
    with pytest.raises(ValidationError):
        BudgetLimits(max_tokens=-1)
    with pytest.raises(ValidationError):
        BudgetLimits(max_cost_usd=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Pricing(input_per_mtok_usd=Decimal(-1), output_per_mtok_usd=Decimal(1))


def test_models_are_frozen_and_forbid_extras() -> None:
    limits = BudgetLimits(max_iterations=1)
    with pytest.raises(ValidationError):
        limits.max_iterations = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        BudgetLimits(max_steps=1)  # type: ignore[call-arg]

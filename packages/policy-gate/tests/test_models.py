"""Gate models: tool specs, context and the decision union."""

from datetime import datetime

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.mandates import ClosedMandate
from agent_toolkit.policy_gate import (
    Allow,
    Decision,
    Deny,
    Effects,
    GateConfig,
    GateContext,
    RequireApproval,
    SessionProfile,
    ToolCall,
    ToolSpec,
)


class Args(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str


def build(args: BaseModel, call_id: str) -> ClosedMandate:
    raise AssertionError("never called by these tests")


def spec(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "name": "lookup_customer",
        "risk": "low",
        "effects": Effects(reads_sensitive=True, changes_state=False, external_comm=False),
        "reversible": True,
        "financial": False,
        "args_model": Args,
        "to_closed_mandate": None,
    }
    return fields | overrides


def test_financial_tool_needs_a_closed_mandate_builder() -> None:
    assert ToolSpec.model_validate(spec(financial=True, to_closed_mandate=build)).financial

    with pytest.raises(ValidationError, match="needs to_closed_mandate"):
        ToolSpec.model_validate(spec(financial=True))


def test_builder_on_a_non_financial_tool_is_rejected() -> None:
    # A builder there would never run: the tool most likely moves money and lacks financial=True.
    with pytest.raises(ValidationError, match="only a financial tool"):
        ToolSpec.model_validate(spec(to_closed_mandate=build))


def test_args_model_must_be_a_pydantic_model() -> None:
    with pytest.raises(ValidationError):
        ToolSpec.model_validate(spec(args_model=dict))


def test_tool_call_keeps_a_snapshot_of_its_args() -> None:
    raw: dict[str, object] = {"customer_id": "cust-juan"}
    call = ToolCall(id="call_1", name="lookup_customer", args=raw)
    raw["customer_id"] = "cust-ana"

    assert call.args == {"customer_id": "cust-juan"}


def test_context_rejects_a_naive_now() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        GateContext(
            session=SessionProfile(
                agent_id="agent_1",
                principal_id="owner_1",
                untrusted_input=False,
                sensitive_access=False,
                state_change=False,
            ),
            mandate=None,
            ledger=None,
            budget_state=BudgetState.empty(),
            budget_limits=BudgetLimits(),
            now=datetime(2026, 12, 1, 12, 0),
            config=GateConfig(),
        )


def test_default_config_asks_for_approval_on_mandate_violations() -> None:
    assert GateConfig().on_mandate_violation == "require_approval"


def test_decisions_carry_at_least_one_reason() -> None:
    with pytest.raises(ValidationError):
        Deny(reasons=())
    with pytest.raises(ValidationError):
        RequireApproval(reasons=(), args=Args(customer_id="c"), closed=None)


def test_decision_is_discriminated_by_kind() -> None:
    decision = TypeAdapter(Decision).validate_json('{"kind": "deny", "reasons": ["tool_mismatch"]}')

    assert decision == Deny(reasons=("tool_mismatch",))


def test_dumped_decision_keeps_the_parsed_args() -> None:
    allow = Allow(args=Args(customer_id="cust-juan"), closed=None)

    assert allow.model_dump() == {
        "kind": "allow",
        "args": {"customer_id": "cust-juan"},
        "closed": None,
    }

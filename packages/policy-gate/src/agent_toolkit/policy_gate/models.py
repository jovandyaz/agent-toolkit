"""What the gate judges (a tool, the session, one proposed call) and what it decides."""

from collections.abc import Callable, Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    model_validator,
)

from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.mandates import Amount, ClosedMandate, OpenMandate, RefState

Identifier = Annotated[str, Field(min_length=1)]
Risk = Literal["low", "medium", "high"]
OnMandateViolation = Literal["deny", "require_approval"]
Reasons = Annotated[tuple[Identifier, ...], Field(min_length=1)]


class Effects(BaseModel):
    """What running a tool does to the world (OpenAI's tool-risk criteria)."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    reads_sensitive: bool
    changes_state: bool
    external_comm: bool


class ToolSpec(BaseModel):
    """A tool the agent may call, its risk, and how to validate what the model asks of it.

    `args_model` parses the model's arguments. It should be frozen, strict and forbid extra
    keys; `decide` validates strictly and rejects unknown keys whatever its config says. Give
    it JSON-shaped fields (str, int, bool, None, list, dict, Literal, nested models): strict
    validation rejects a string for a date, Decimal or Enum field and a list for a tuple.

    A `financial` tool issues payment requests, so it needs `to_closed_mandate`: it turns the
    parsed arguments (an `args_model` instance) into the concrete charge that `mandates.check`
    judges. The application fills `agent_id` from the session and mints a new charge's
    `charge_ref` server-side; neither comes from the arguments. Only a financial tool takes a
    builder: on any other tool it would never run, which most likely means a money tool is
    missing `financial=True`.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Identifier
    risk: Risk
    effects: Effects
    reversible: bool
    financial: bool
    args_model: type[BaseModel]
    to_closed_mandate: Callable[[BaseModel], ClosedMandate] | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.financial and self.to_closed_mandate is None:
            raise ValueError(f"financial tool {self.name!r} needs to_closed_mandate")
        if not self.financial and self.to_closed_mandate is not None:
            raise ValueError(f"only a financial tool takes to_closed_mandate, not {self.name!r}")
        return self


class SessionProfile(BaseModel):
    """Who acts, for whom, and the session's Rule of Two flags (Meta, Agents Rule of Two).

    A session with untrusted input, access to sensitive data and the power to change state
    holds all three; then every state change needs a verified mandate or a human.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    agent_id: Identifier
    principal_id: Identifier
    untrusted_input: bool
    sensitive_access: bool
    state_change: bool


class ToolCall(BaseModel):
    """One tool call as the model proposed it. Untrusted: `decide` validates it all."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: str
    name: str
    args: Mapping[str, object]


class GateConfig(BaseModel):
    """What a mandate violation that a human may override leads to (see `decide`)."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    on_mandate_violation: OnMandateViolation = "require_approval"


class GateContext(BaseModel):
    """Everything besides the call that `decide` needs, all from trusted application state.

    `consumed` and `ref_state` come from the mandate's ledger (see `decide`); a financial call
    under a mandate needs both.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    session: SessionProfile
    mandate: OpenMandate | None
    consumed: Amount | None
    ref_state: RefState | None
    budget_state: BudgetState
    budget_limits: BudgetLimits
    now: AwareDatetime
    config: GateConfig


class Allow(BaseModel):
    """Run the tool with `args`, and for a financial tool issue exactly `closed`."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["allow"] = "allow"
    args: SerializeAsAny[BaseModel]
    closed: ClosedMandate | None


class RequireApproval(BaseModel):
    """Run the tool only once a human approves this exact call, for every reason listed."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["require_approval"] = "require_approval"
    reasons: Reasons
    args: SerializeAsAny[BaseModel]
    closed: ClosedMandate | None


class Deny(BaseModel):
    """Do not run the tool; no approval can override it."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["deny"] = "deny"
    reasons: Reasons


Decision = Annotated[Allow | RequireApproval | Deny, Field(discriminator="kind")]

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
from agent_toolkit.mandates import ClosedMandate, ConsumptionEntry, OpenMandate

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


class BuildContext(BaseModel):
    """What a closed-mandate builder knows about the call besides its arguments.

    `thread_id` is the conversation the call belongs to, `message_id` the id of the model's
    message that proposed it (one per turn) and `call_id` the model's id for the call. All come
    from the runtime, never from tool arguments, and together identify the call: a call id
    alone can repeat in another message (some providers number calls `call_0`, `call_1` in
    every message) or another thread. Fields may be added; builders read them by name.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    thread_id: Identifier
    message_id: Identifier
    call_id: Identifier


class ToolSpec(BaseModel):
    """A tool the agent may call, its risk, and how to validate what the model asks of it.

    `args_model` parses the model's arguments. It should be frozen, strict and forbid extra
    keys; `decide` validates strictly and rejects unknown keys whatever its config says. Give
    it JSON-shaped fields (str, int, bool, None, list, dict, Literal, nested models): strict
    validation rejects a string for a date, Decimal or Enum field and a list for a tuple.

    A `financial` tool issues payment requests, so it needs `to_closed_mandate(args, ctx)`: it
    turns the parsed arguments (an `args_model` instance) of one tool call, identified by `ctx`
    (a `BuildContext`), into the concrete charge that `mandates.check` judges. The application
    fills `agent_id` from the session and mints a new charge's `charge_ref` server-side,
    deterministically from `ctx.thread_id`, `ctx.message_id` and `ctx.call_id` together (for
    example a hash of all three); neither ever comes from the arguments. The builder must be
    deterministic for a given (args, ctx) and free of side effects: it may run more than once
    per call (the LangGraph adapter also re-runs `decide` on resume), and an approval binds the
    charge it showed.
    For an `amount_increase` it loads the payer, receiving account and rail from the charge the
    application stored under the `charge_ref` the arguments name; only the amounts come from the
    arguments (`mandates.check` binds `previous_amount` to the ledger, nothing binds the rest).
    Only a financial tool takes a builder: on any other tool it would never run, which most
    likely means a money tool is missing `financial=True`.

    The spec is trusted configuration. The gate cannot tell that a tool moves money: one
    declared `financial=False` skips the mandate entirely and is judged only by its risk,
    effects and the session.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Identifier
    risk: Risk
    effects: Effects
    reversible: bool
    financial: bool
    args_model: type[BaseModel]
    to_closed_mandate: Callable[[BaseModel, BuildContext], ClosedMandate] | None

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
    or communicate externally (`state_change`) holds all three; then every tool that changes
    state or communicates externally needs a verified mandate or a human.
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

    `ledger` holds every entry of this mandate's consumption ledger and nothing else (see
    `decide`); a financial call under a mandate needs it, even when it is empty.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    session: SessionProfile
    mandate: OpenMandate | None
    ledger: tuple[ConsumptionEntry, ...] | None
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

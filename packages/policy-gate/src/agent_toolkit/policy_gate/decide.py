"""The gate's decision for one tool call: a pure function of the call, the tool and the context."""

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from agent_toolkit import budgets, mandates
from agent_toolkit.mandates import Amount, ClosedMandate, OpenMandate, RefState, Violation
from agent_toolkit.policy_gate.models import (
    Allow,
    Decision,
    Deny,
    GateContext,
    RequireApproval,
    SessionProfile,
    ToolCall,
    ToolSpec,
)

# Mandate violations a human may override when `GateConfig.on_mandate_violation` allows it. Any
# other violation, including one `mandates` adds later, is always denied: it pays an account
# that is not the credential holder's (rule 7.3(c)), runs under another mandate, agent or
# currency, outside the mandate's life, or contradicts the ledger.
_OVERRIDABLE: frozenset[Violation] = frozenset(
    {"rail_not_allowed", "amount_out_of_range", "budget_exceeded", "outside_window"}
)


@dataclass(frozen=True)
class _MandateInputs:
    mandate: OpenMandate
    consumed: Amount
    ref_state: RefState


@dataclass(frozen=True)
class _Charge:
    closed: ClosedMandate | None
    approvals: tuple[str, ...]
    verified: bool


def decide(call: ToolCall, tool: ToolSpec, ctx: GateContext) -> Decision:
    """Decide whether `call` may run, before it runs.

    Rules, in order. The first `Deny` wins and approval reasons accumulate, so `Deny` takes
    precedence over `RequireApproval`:

    1. `call.name != tool.name`: deny `tool_mismatch`.
    2. Arguments that fail `tool.args_model` in strict mode, unknown keys included at every
       depth whatever the model's own config: deny `invalid_args:…`.
    3. An exceeded run budget: deny `budget_exceeded:<kind>`.
    4. A financial tool without a mandate needs approval (`no_mandate`). With one, it builds
       the charge with `tool.to_closed_mandate`; a `ValueError` (pydantic's `ValidationError`
       included) denies `invalid_args:closed…`, and a charge for another agent than
       `ctx.session.agent_id` denies `agent_mismatch`. Then `mandates.check` judges it:
       violations outside `rail_not_allowed`, `amount_out_of_range`, `budget_exceeded` and
       `outside_window` are always denied; those four follow `ctx.config`. Its approval
       reasons need approval.
    5. A non-financial `high` risk tool needs approval (`high_risk`).
    6. A session holding all three Rule of Two flags needs approval for a tool that changes
       state (`rule_of_two`), unless it is a financial call its mandate fully verified.
    7. Otherwise allow.

    Reasons are stable, machine-readable strings, `<code>` or `<code>:<detail>`:

    - `tool_mismatch`: the call names another tool than the spec it was routed to.
    - `invalid_args:<loc>:<type>`: one per validation error, with pydantic's error `type`.
      `<loc>` is `args` plus the dotted path inside the arguments (`args.amount_minor`), or
      `closed` plus the path inside the model that rejected the charge built from them
      (`closed`, `closed.minor`). Error messages are left out: they may echo PII.
    - `budget_exceeded:<kind>`: one per exceeded run limit (`budgets.ExceededKind`).
    - `agent_mismatch`: the charge names another agent than the session's.
    - `mandate:<violation>` and `mandate:<approval reason>`: what `mandates.check` reported,
      with its own names (`mandate:amount_out_of_range`, `mandate:open_amount`).
    - `no_mandate`, `high_risk`, `rule_of_two`: reasons a human must approve.

    Caller duties for a financial call under a mandate:

    - Compute `ctx.consumed` (`mandates.consumed_in_period`) and `ctx.ref_state`
      (`mandates.ref_state` for the charge's `charge_ref`) from this mandate's ledger only,
      never from tool arguments.
    - Hold a lock on the mandate from computing them until the reserve of an allowed or
      approved charge is recorded after execution: two calls judged against one snapshot can
      both pass and together overspend.
    - Treat a `ValueError` raised while computing them (an inconsistent ledger) as a deny,
      never as an allow.
    - Execute exactly the returned `closed`, not a charge rebuilt from the arguments.

    Raises ValueError, before any rule, when a financial tool runs under a mandate and
    `ctx.consumed` or `ctx.ref_state` is missing: treating it as empty would overspend.
    Exceptions other than `ValueError` from the tool's models or builder propagate.
    """
    inputs = _mandate_inputs(tool, ctx)
    if call.name != tool.name:
        return Deny(reasons=("tool_mismatch",))
    args = _parse_args(call, tool)
    if isinstance(args, Deny):
        return args
    exceeded = budgets.check(ctx.budget_state, ctx.budget_limits)
    if exceeded:
        return Deny(reasons=tuple(f"budget_exceeded:{kind}" for kind in exceeded))

    charge = _Charge(closed=None, approvals=(), verified=False)
    if tool.financial:
        judged = _judge_charge(args, tool, ctx, inputs)
        if isinstance(judged, Deny):
            return judged
        charge = judged
    approvals = list(charge.approvals)
    if not tool.financial and tool.risk == "high":
        approvals.append("high_risk")
    if _holds_rule_of_two(ctx.session) and tool.effects.changes_state and not charge.verified:
        approvals.append("rule_of_two")

    if approvals:
        return RequireApproval(reasons=tuple(approvals), args=args, closed=charge.closed)
    return Allow(args=args, closed=charge.closed)


def _mandate_inputs(tool: ToolSpec, ctx: GateContext) -> _MandateInputs | None:
    """The mandate and its ledger values a financial call needs; None without a mandate."""
    if not tool.financial or ctx.mandate is None:
        return None
    if ctx.consumed is None or ctx.ref_state is None:
        ledger = {"consumed": ctx.consumed, "ref_state": ctx.ref_state}
        missing = " and ".join(name for name, value in ledger.items() if value is None)
        raise ValueError(
            f"financial tool {tool.name!r} under mandate {ctx.mandate.id!r} needs {missing} "
            "from the mandate's ledger"
        )
    return _MandateInputs(mandate=ctx.mandate, consumed=ctx.consumed, ref_state=ctx.ref_state)


def _parse_args(call: ToolCall, tool: ToolSpec) -> BaseModel | Deny:
    try:
        return tool.args_model.model_validate(call.args, strict=True, extra="forbid")
    except ValidationError as error:
        return Deny(reasons=_invalid_args("args", error))


def _judge_charge(
    args: BaseModel, tool: ToolSpec, ctx: GateContext, inputs: _MandateInputs | None
) -> _Charge | Deny:
    if inputs is None:
        return _Charge(closed=None, approvals=("no_mandate",), verified=False)
    build = tool.to_closed_mandate
    if build is None:  # ToolSpec validation forbids it; model_construct can bypass that.
        raise ValueError(f"financial tool {tool.name!r} has no to_closed_mandate")
    try:
        closed = build(args)
    except ValueError as error:
        return Deny(reasons=_invalid_args("closed", error))
    if closed.agent_id != ctx.session.agent_id:
        return Deny(reasons=("agent_mismatch",))

    verdict = mandates.check(
        closed, inputs.mandate, inputs.consumed, ctx.now, ref_state=inputs.ref_state
    )
    deny_overridable = ctx.config.on_mandate_violation == "deny"
    denied = [v for v in verdict.violations if deny_overridable or v not in _OVERRIDABLE]
    if denied:
        return Deny(reasons=tuple(f"mandate:{violation}" for violation in denied))
    approvals = (*verdict.violations, *verdict.approval_reasons)
    return _Charge(
        closed=closed,
        approvals=tuple(f"mandate:{reason}" for reason in approvals),
        verified=verdict.ok,
    )


def _invalid_args(origin: str, error: ValueError) -> tuple[str, ...]:
    if not isinstance(error, ValidationError):
        return (f"invalid_args:{origin}:value_error",)
    return tuple(
        f"invalid_args:{'.'.join([origin, *map(str, detail['loc'])])}:{detail['type']}"
        for detail in error.errors(include_url=False, include_context=False, include_input=False)
    )


def _holds_rule_of_two(session: SessionProfile) -> bool:
    return session.untrusted_input and session.sensitive_access and session.state_change

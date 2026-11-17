"""LangGraph tool node that runs the tool calls of a turn only as the policy gate allows.

Install with the `langgraph` extra. Put it where a graph would put `ToolNode`:
`builder.add_node("tools", GatedToolNode(tools, specs, context, on_decision=audit))`.

Each run reads the `tool_calls` of `state["messages"][-1]`, which must be an `AIMessage` (the
node never looks further back, so re-entering it cannot run a turn twice), and needs
`configurable["thread_id"]` (ValueError otherwise, before anything is decided). Each call's
idempotency key is the SHA-256 hex of the compact JSON array `[thread_id, call_id]`: the same
on every execution of the call, and different for any other call. Then it:

1. Builds the context with `context(state)` and decides every call with
   `decide(..., thread_id=thread_id)`, in order, before running any; a financial tool's builder
   gets `BuildContext(thread_id, call_id)`. Each call is judged against the ledger plus a
   reserve (`mandates.reserve_amount(closed)`, dated `ctx.now`) for every earlier charge of the
   turn that was allowed or needs approval, so one turn cannot overspend the mandate's budget
   (a later rejection only frees budget). A call on the `charge_ref` of an earlier call of the
   turn that was not denied is denied (`charge_ref_in_turn`, not configurable): its check would
   rest on a reserve that rejecting the earlier call undoes, so multi-step changes to one
   charge (create then raise, raise twice) must span turns. The rule is checked on the ref the
   builder gives the call and again on the ref of the charge `decide` checked, so a builder
   that is not deterministic cannot slip a second change past it. All of this is pure and
   deterministic, so the re-run on resume rebuilds the same decisions and hashes. A call to a
   tool missing from `tools` or `specs` is denied (`unknown_tool`).
2. If calls need approval, pauses the graph with one `interrupt` whose value is plain JSON, as
   LangGraph requires: a list with one `ApprovalRequest.model_dump(mode="json")` per call,
   `{"call_id", "tool", "args", "reasons", "closed_mandate_hash", "closed", "request_hash"}`,
   where `closed` is the charge a financial call will issue (null for others). The approval
   screen reads one back with `ApprovalRequest.model_validate_json(json.dumps(item))`.
   Nothing has run yet.
3. On resume LangGraph re-runs the node from its start, so steps 1 and 2 repeat with a fresh
   context and `interrupt` returns the resume value, a `dict[call_id, ApprovalDecision]`. An
   approval counts only if its `request_hash` equals the one recomputed now, which binds the
   thread id, call id, tool, validated arguments, closed mandate and reasons: an approval
   covers neither another thread that reuses the call id, nor other arguments or another
   charge, nor a reason that appeared after it was given (for example
   `mandate:budget_exceeded` once the ledger moved). The hash is an integrity binding, not
   authentication: anyone who can resume the thread can read it in the interrupt, so the
   resume channel itself must be authenticated. A resume value that is not
   such a mapping rejects every pending call (`approval_invalid`); a missing, mismatched or
   refused approval rejects its own call (`approval_missing`, `approval_mismatch`,
   `approval_rejected`).
4. Runs the allowed and approved calls one at a time, in order, with their validated
   arguments (`args.model_dump(mode="json")`, exactly what the approval showed). The tool's
   config holds the checked `ClosedMandate` (None if the call has none) in
   `configurable[CLOSED_MANDATE_KEY]` and the call's idempotency key in
   `configurable[IDEMPOTENCY_KEY]`; a financial tool reads them with
   `closed_mandate_from_config` and `idempotency_key_from_config`, issues exactly that charge
   and sends the key to the provider. Every other call gets
   `ToolMessage(status="error", content="Not executed: <reasons>")`, except a call denied for
   `mandate:charge_ref_reused` or `mandate:previous_amount_mismatch`, which also describe a
   charge that already ran (see "Re-runs"): it gets `ToolMessage(status="error",
   content="Check the charge's status before retrying, it may already exist: <reasons>")`.
5. Calls `on_decision(call, decision, approval)` once per call with its final decision, always
   before the call would run: first a `Deny` for every call that does not run, before any
   tool runs; then, right before each call runs, `Allow`, or `RequireApproval` with the
   approval that lets it run. So an `on_decision` that raises stops its call, and every call
   after it, from running: an audit log that cannot record a charge blocks it. A rejection's
   reasons are its `approval_*` code followed by the gate's, and it carries the human's answer
   if there was one. The report is the decision, not the result: a call that then fails was
   reported as allowed. Calls after a failed one are neither reported nor run (see
   "Re-runs"). The application wires `on_decision` to `audit-log`; its exceptions propagate.

Re-runs. A node is LangGraph's unit of retry. When a tool or `on_decision` raises after some
calls of the turn ran, the node fails and its replies are lost; running the thread again
(`graph.invoke(None, config)`) executes the whole turn again, and LangGraph replays the
approvals it already stored instead of interrupting. Every call is decided and reported again,
so the audit log can hold two records of one call, and a call that already ran:

- under a mandate, with its reserve recorded, is denied (`mandate:charge_ref_reused`, or
  `mandate:previous_amount_mismatch` for an increase); the model is told to check the charge's
  status, not that it never ran, and `on_decision` receives that `Deny`;
- without a reserve (no mandate, or an open amount) runs again on the stored approval, with the
  same idempotency key: the tool must not issue it twice (send the key to the provider).

Resume with `graph.invoke(Command(resume={call_id: decision.model_dump(mode="json")}), config)`,
`decision` an `ApprovalDecision` (its `at` then travels as an ISO 8601 string): the checkpoint
stores the resume value, so keep it JSON too (an `ApprovalDecision` object is also accepted).
LangGraph reads a dict whose keys all look like interrupt ids (32 lowercase hex characters),
the empty dict included, as answers keyed by interrupt id: it answers nothing and the graph
interrupts again. Key the answers by the interrupt to avoid that,
`Command(resume={interrupt.id: {call_id: ...}})`, as LangGraph requires anyway while several
interrupts are pending.

Duties of the application:

- Resume only from the trusted approval screen through an authenticated channel, never from
  chat input. The interrupt has no `response_schema`, so chat clients render no approval
  form. Run the graph with a checkpointer and a thread.
- `context(state)` runs on every execution, the re-run on resume included. Hold the mandate
  lock from the first decision, across the interrupt, until the charge's reserve is recorded;
  without it the re-run re-verifies against whatever changed, and a call can end denied or
  gain reasons that void its approval.
- `ToolSpec.to_closed_mandate(args, ctx)` must be deterministic and free of side effects:
  the node calls it more than once per call (to find the call's `charge_ref`, inside `decide`,
  and again on the re-run), and the approval binds the charge's `content_hash`, so a builder
  that mints a random `charge_ref` per call voids every approval of a financial call (it fails
  closed). Mint a new charge's ref server-side from `ctx.thread_id` and `ctx.call_id`
  together (for example a hash of both; a call id alone can repeat in another thread), never
  from tool arguments.
- A financial tool issues the charge from `closed_mandate_from_config(config)`, not one
  rebuilt from its arguments, and the reserve is recorded under its `charge_ref`.
- Make tools that move money idempotent under `idempotency_key_from_config(config)`: send it
  as the provider's idempotency key, so a re-run (see "Re-runs") never issues a charge twice.
  The key, not `charge_ref`, identifies the call: an increase keeps its charge's ref. Give this
  node no `RetryPolicy`.
- Tools must not call `interrupt()`: resuming it re-runs this node, and every call that ran
  before it would run again.
- Each tool takes exactly the arguments of its spec's `args_model` (checked at construction):
  the tool re-validates them with its own schema, which must not add arguments the gate never
  judged and the approval never showed.

No error is swallowed. A tool's exception propagates unless the tool handles it itself
(`handle_tool_error`, which replies with an error `ToolMessage`). So does the
`BudgetExceededError` that `budgets.adapters.langchain.BudgetCallbackHandler` raises when a tool
starts over budget. The stock `ToolNode` re-raises it with its default `handle_tool_errors`,
but `handle_tool_errors=True` (or any handler covering `Exception`) turns it into an error reply
to the model; the sticky handler then stops the run at its next callback.

The node is synchronous (under `ainvoke` LangGraph runs it in a worker thread). A tool must
return content or a `ToolMessage`, not a `Command`.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent_toolkit import mandates
from agent_toolkit.mandates import ClosedMandate, ConsumptionEntry
from agent_toolkit.policy_gate.decide import decide
from agent_toolkit.policy_gate.models import (
    Allow,
    BuildContext,
    Decision,
    Deny,
    GateContext,
    RequireApproval,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "CLOSED_MANDATE_KEY",
    "IDEMPOTENCY_KEY",
    "ApprovalDecision",
    "ApprovalRequest",
    "GatedToolNode",
    "closed_mandate_from_config",
    "idempotency_key_from_config",
]

CLOSED_MANDATE_KEY = "agent_toolkit_closed_mandate"
"""Key in `configurable` of the config a tool runs with: the gate's checked `ClosedMandate`."""

IDEMPOTENCY_KEY = "agent_toolkit_idempotency_key"
"""Key in `configurable` of the config a tool runs with: the running call's idempotency key."""

Identifier = Annotated[str, Field(min_length=1)]


def closed_mandate_from_config(config: RunnableConfig) -> ClosedMandate | None:
    """The charge the gate checked (and a human approved, if needed) for the running call.

    A financial tool declares a `config: RunnableConfig` parameter and issues exactly this
    charge, with its `charge_ref` and amount; it never rebuilds one from its arguments. None
    for a call without a closed mandate (not financial, or financial without a mandate).
    Raises TypeError if the key holds anything but a `ClosedMandate`.
    """
    closed = config.get("configurable", {}).get(CLOSED_MANDATE_KEY)
    if closed is not None and not isinstance(closed, ClosedMandate):
        raise TypeError(f"{CLOSED_MANDATE_KEY} holds {type(closed).__name__}, not a ClosedMandate")
    return closed


def idempotency_key_from_config(config: RunnableConfig) -> str:
    """The idempotency key of the running call: stable across re-runs, unique to the call.

    The SHA-256 hex of the compact JSON array `[thread_id, call_id]`. A tool that moves money
    sends it as the provider's idempotency key, so running the turn again never issues the same
    request twice. Raises ValueError outside `GatedToolNode` (no key) and TypeError if the key
    holds anything but a string.
    """
    key = config.get("configurable", {}).get(IDEMPOTENCY_KEY)
    if key is None:
        raise ValueError(f"no {IDEMPOTENCY_KEY} in the config: run the tool through GatedToolNode")
    if not isinstance(key, str):
        raise TypeError(f"{IDEMPOTENCY_KEY} holds {type(key).__name__}, not a str")
    return key


class ApprovalRequest(BaseModel):
    """One call waiting for a human, as the trusted approval screen shows it.

    `args` are the validated arguments the call will run with; `closed` is the charge a
    financial call will issue, exactly as the tool gets it (None for a call without one), and
    `closed_mandate_hash` its `content_hash`. `request_hash` is the SHA-256 hex of the canonical
    JSON (sorted keys, no spaces) of every other field but `closed`, which its hash binds,
    `reasons` included so an approval never covers a reason it did not show, plus the graph's
    thread id so it never covers another thread; an `ApprovalDecision` echoes it.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: Identifier
    tool: Identifier
    args: dict[str, object]
    reasons: tuple[Identifier, ...]
    closed_mandate_hash: str | None
    closed: ClosedMandate | None
    request_hash: Identifier


class ApprovalDecision(BaseModel):
    """A human's answer to one `ApprovalRequest`, from the trusted approval screen.

    It counts only for the request whose `request_hash` it echoes. `approver_id` and
    `approval_id` identify the human and the approval record for the audit log. `at` also takes
    an ISO 8601 string, as JSON carries it; every other field is strict.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    request_hash: Identifier
    approved: bool
    approver_id: Identifier
    approval_id: Identifier
    at: Annotated[AwareDatetime, Field(strict=False)]


_ANSWERS = TypeAdapter(dict[str, ApprovalDecision])
_SAME_CHARGE_IN_TURN = Deny(reasons=("charge_ref_in_turn",))
# Denials that a re-run also gives a charge that already ran: the refusal must not claim that
# the charge was never executed.
_MAY_HAVE_RUN = frozenset({"mandate:charge_ref_reused", "mandate:previous_amount_mismatch"})


@dataclass(frozen=True)
class _Outcome:
    call: ToolCall
    decision: Decision
    approval: ApprovalDecision | None


class GatedToolNode[StateT: Mapping[str, object]]:
    """A graph node that decides, asks a human when needed, then runs the turn's tool calls.

    `tools` run the calls and `specs` judge them, both keyed by tool name. `context` builds
    the `GateContext` from trusted state on every execution. See the module docstring for the
    algorithm and the application's duties. Raises ValueError for repeated tool names, a spec
    stored under another name than its own, or a tool whose arguments (as the model sends
    them) are not exactly the fields of its spec's `args_model`.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        specs: Mapping[str, ToolSpec],
        context: Callable[[StateT], GateContext],
        on_decision: Callable[[ToolCall, Decision, ApprovalDecision | None], None] | None = None,
    ) -> None:
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("tool names must be unique")
        for name, spec in specs.items():
            if spec.name != name:
                raise ValueError(f"the spec under {name!r} is named {spec.name!r}")
            tool = by_name.get(name)
            if tool is not None and _arg_names(tool) != frozenset(spec.args_model.model_fields):
                raise ValueError(f"tool {name!r} takes other arguments than its spec's args_model")
        self._tools = by_name
        self._specs = dict(specs)
        self._context = context
        self._on_decision = on_decision

    def __call__(self, state: StateT, config: RunnableConfig) -> dict[str, list[ToolMessage]]:
        calls = _tool_calls(state)
        if not calls:
            return {"messages": []}
        thread_id = _thread_id(config)
        decisions = self._decide_in_order(calls, self._context(state), thread_id)
        requests = _requests(calls, decisions, thread_id)
        answers = _ask(list(requests.values()))
        outcomes = [
            _resolve(call, decision, requests.get(call.id), answers)
            for call, decision in zip(calls, decisions, strict=True)
        ]

        for outcome in outcomes:  # Report what will not run before anything runs.
            if isinstance(outcome.decision, Deny):
                self._report(outcome)
        replies: list[ToolMessage] = []
        for outcome in outcomes:
            if isinstance(outcome.decision, Deny):
                replies.append(_refusal(outcome.call, outcome.decision))
                continue
            self._report(outcome)  # Before it runs: a failing audit log stops the call.
            key = _idempotency_key(thread_id, outcome.call.id)
            replies.append(self._run(outcome.call, outcome.decision, config, key))
        return {"messages": replies}

    def _decide_in_order(
        self, calls: Sequence[ToolCall], ctx: GateContext, thread_id: str
    ) -> list[Decision]:
        """Decide each call against the ledger plus the reserves of the turn's earlier charges.

        A call on a charge that an earlier call of the turn may still change is denied
        (`charge_ref_in_turn`): its check would rest on a reserve that rejecting the earlier
        call undoes. Budget alone needs no such rule: a rejection only frees budget. The ref is
        checked before deciding (as the builder plans it) and after (as `decide` checked it).
        """
        decisions: list[Decision] = []
        taken: set[str] = set()  # Refs of the turn's charges that may still run.
        for call in calls:
            if self._planned_ref(call, ctx, thread_id) in taken:
                decision: Decision = _SAME_CHARGE_IN_TURN
            else:
                decision = self._decide(call, ctx, thread_id)
            closed = None if isinstance(decision, Deny) else decision.closed
            if closed is not None and closed.charge_ref in taken:  # A builder that drifts.
                decision = _SAME_CHARGE_IN_TURN
            elif closed is not None:
                taken.add(closed.charge_ref)
            decisions.append(decision)
            ctx = _with_reserve(ctx, decision)
        return decisions

    def _planned_ref(self, call: ToolCall, ctx: GateContext, thread_id: str) -> str | None:
        """The `charge_ref` the call's builder gives it, as `decide` would build it, if any."""
        spec = self._specs.get(call.name)
        if spec is None or spec.to_closed_mandate is None or ctx.mandate is None:
            return None
        try:
            args = spec.args_model.model_validate(call.args, strict=True, extra="forbid")
            build_ctx = BuildContext(thread_id=thread_id, call_id=call.id)
            return spec.to_closed_mandate(args, build_ctx).charge_ref
        except ValueError:  # `decide` denies the call (`invalid_args:…`).
            return None

    def _decide(self, call: ToolCall, ctx: GateContext, thread_id: str) -> Decision:
        spec = self._specs.get(call.name)
        if spec is None or call.name not in self._tools:
            return Deny(reasons=("unknown_tool",))
        return decide(call, spec, ctx, thread_id=thread_id)

    def _run(
        self, call: ToolCall, decision: Allow | RequireApproval, config: RunnableConfig, key: str
    ) -> ToolMessage:
        validated = tool_call(name=call.name, args=_json_args(decision), id=call.id)
        # Always set, None included, so a value the caller put under a key never reaches a tool.
        checked = patch_config(
            config, configurable={CLOSED_MANDATE_KEY: decision.closed, IDEMPOTENCY_KEY: key}
        )
        output = self._tools[call.name].invoke(validated, checked)
        if not isinstance(output, ToolMessage):
            raise TypeError(
                f"tool {call.name!r} returned {type(output).__name__}, not content or a ToolMessage"
            )
        return output

    def _report(self, outcome: _Outcome) -> None:
        if self._on_decision is not None:
            self._on_decision(outcome.call, outcome.decision, outcome.approval)


def _arg_names(tool: BaseTool) -> frozenset[str]:
    """The arguments the model sends the tool: its schema minus injected arguments."""
    schema = tool.tool_call_schema
    if isinstance(schema, dict):
        return frozenset(schema.get("properties", {}))
    if not issubclass(schema, BaseModel):
        raise TypeError(f"tool {tool.name!r} has a pydantic.v1 schema; use pydantic v2")
    return frozenset(schema.model_fields)


def _tool_calls(state: Mapping[str, object]) -> list[ToolCall]:
    messages = state.get("messages")
    last = messages[-1] if isinstance(messages, Sequence) and messages else None
    if not isinstance(last, AIMessage):
        raise ValueError("GatedToolNode needs state['messages'] to end with an AIMessage")
    calls: list[ToolCall] = []
    for proposed in last.tool_calls:
        call_id = proposed["id"]
        if not call_id:
            raise ValueError("every tool call needs an id")
        calls.append(ToolCall(id=call_id, name=proposed["name"], args=proposed["args"]))
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("tool call ids must be unique within a turn")
    return calls


def _with_reserve(ctx: GateContext, decision: Decision) -> GateContext:
    """`ctx` with the budget a charge that may still run takes, as if already recorded."""
    if isinstance(decision, Deny) or decision.closed is None or ctx.ledger is None:
        return ctx
    amount = mandates.reserve_amount(decision.closed)
    if amount is None:  # An open amount reserves nothing.
        return ctx
    entry = ConsumptionEntry(
        charge_ref=decision.closed.charge_ref, kind="reserve", amount=amount, at=ctx.now
    )
    return ctx.model_copy(update={"ledger": (*ctx.ledger, entry)})


def _idempotency_key(thread_id: str, call_id: str) -> str:
    canonical = json.dumps([thread_id, call_id], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _json_args(decision: Allow | RequireApproval) -> dict[str, object]:
    return decision.args.model_dump(mode="json")


def _requests(
    calls: Sequence[ToolCall], decisions: Sequence[Decision], thread_id: str
) -> dict[str, ApprovalRequest]:
    return {
        call.id: _request(call, decision, thread_id)
        for call, decision in zip(calls, decisions, strict=True)
        if isinstance(decision, RequireApproval)
    }


def _thread_id(config: RunnableConfig) -> str:
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id is None or not str(thread_id):
        raise ValueError(
            "GatedToolNode needs config['configurable']['thread_id']: run the graph with a "
            "checkpointer and a thread"
        )
    return str(thread_id)


def _request(call: ToolCall, decision: RequireApproval, thread_id: str) -> ApprovalRequest:
    args = _json_args(decision)
    closed_hash = None if decision.closed is None else decision.closed.content_hash
    bound = {
        "thread_id": thread_id,
        "call_id": call.id,
        "tool": call.name,
        "args": args,
        "reasons": decision.reasons,
        "closed_mandate_hash": closed_hash,
    }
    canonical = json.dumps(bound, sort_keys=True, separators=(",", ":"))
    return ApprovalRequest(
        call_id=call.id,
        tool=call.name,
        args=args,
        reasons=decision.reasons,
        closed_mandate_hash=closed_hash,
        closed=decision.closed,
        request_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _ask(requests: list[ApprovalRequest]) -> dict[str, ApprovalDecision] | None:
    """The human's answers by call id: empty if nothing needs one, None if malformed."""
    if not requests:
        return {}
    resumed = interrupt([request.model_dump(mode="json") for request in requests])
    try:
        return _ANSWERS.validate_python(resumed)
    except ValidationError:
        return None


def _resolve(
    call: ToolCall,
    decision: Decision,
    request: ApprovalRequest | None,
    answers: Mapping[str, ApprovalDecision] | None,
) -> _Outcome:
    if request is None or not isinstance(decision, RequireApproval):
        return _Outcome(call=call, decision=decision, approval=None)
    if answers is None:
        return _Outcome(call=call, decision=_rejected("approval_invalid", decision), approval=None)
    approval = answers.get(call.id)
    if approval is None:
        rejected = _rejected("approval_missing", decision)
    elif approval.request_hash != request.request_hash:
        rejected = _rejected("approval_mismatch", decision)
    elif not approval.approved:
        rejected = _rejected("approval_rejected", decision)
    else:
        return _Outcome(call=call, decision=decision, approval=approval)
    return _Outcome(call=call, decision=rejected, approval=approval)


def _rejected(code: str, decision: RequireApproval) -> Deny:
    return Deny(reasons=(code, *decision.reasons))


def _refusal(call: ToolCall, deny: Deny) -> ToolMessage:
    reasons = ", ".join(deny.reasons)
    if _MAY_HAVE_RUN.intersection(deny.reasons):
        content = f"Check the charge's status before retrying, it may already exist: {reasons}"
    else:
        content = f"Not executed: {reasons}"
    return ToolMessage(
        content=content,
        tool_call_id=call.id,
        name=call.name,
        status="error",
    )

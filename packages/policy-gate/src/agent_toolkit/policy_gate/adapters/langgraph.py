"""LangGraph tool node that runs the tool calls of a turn only as the policy gate allows.

Install with the `langgraph` extra. Put it where a graph would put `ToolNode`:
`builder.add_node("tools", GatedToolNode(tools, specs, context, on_decision=audit))`.

Each run reads the `tool_calls` of `state["messages"][-1]`, which must be an `AIMessage` (the
node never looks further back, so re-entering it cannot run a turn twice), and:

1. Builds the context with `context(state)` and decides every call with `decide`, in order,
   before running any. Each call is judged against the ledger plus a reserve
   (`mandates.reserve_amount(closed)`, dated `ctx.now`) for every earlier charge of the turn
   that was allowed or needs approval, so one turn can neither overspend the mandate's budget
   nor raise one link twice from the same starting amount. The working ledger is pure and
   deterministic, so the re-run on resume rebuilds the same decisions and hashes. A call to a
   tool missing from `tools` or `specs` is denied (`unknown_tool`).
2. If calls need approval, pauses the graph with one `interrupt` whose value is the list of
   their `ApprovalRequest`s. Nothing has run yet.
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
   arguments (`args.model_dump(mode="json")`, exactly what the approval showed) and the
   checked `ClosedMandate` (None if the call has none) in `configurable[CLOSED_MANDATE_KEY]`
   of the tool's config; a financial tool reads it with `closed_mandate_from_config` and
   issues exactly that charge. Every other call gets
   `ToolMessage(status="error", content="Not executed: <reasons>")`.
5. Calls `on_decision(call, decision, approval)` once per call with its final outcome: `Allow`,
   or `RequireApproval` with the approval that let it run, once the tool ran or raised (a
   failed call is reported, then its error propagates); or `Deny` for a call that does not
   run, before any tool runs. A rejection's reasons are its `approval_*` code followed by the
   gate's, and it carries the human's answer if there was one. Calls after a failed one are
   neither run nor reported; running the thread again decides and reports them. The
   application wires `on_decision` to `audit-log`; its exceptions propagate.

Resume with `graph.invoke(Command(resume={call_id: ApprovalDecision(...)}), config)`; JSON works
too (`at` takes an ISO 8601 string). LangGraph reads a dict whose keys all look like interrupt
ids (32 lowercase hex characters), the empty dict included, as answers keyed by interrupt id:
it answers nothing and the graph interrupts again. Key the answers by the interrupt to avoid
that, `Command(resume={interrupt.id: {call_id: ...}})`, as LangGraph requires anyway while
several interrupts are pending.

Duties of the application:

- Resume only from the trusted approval screen through an authenticated channel, never from
  chat input. The interrupt has no `response_schema`, so chat clients render no approval
  form. Approvals need a thread: without `configurable["thread_id"]` the node raises
  ValueError.
- `context(state)` runs on every execution, the re-run on resume included. Hold the mandate
  lock from the first decision, across the interrupt, until the charge's reserve is recorded;
  without it the re-run re-verifies against whatever changed, and a call can end denied or
  gain reasons that void its approval.
- `ToolSpec.to_closed_mandate(args, call_id)` must be deterministic: the approval binds the
  charge's `content_hash`, so a builder that mints a random `charge_ref` per call voids every
  approval of a financial call (it fails closed). Mint the ref server-side from the call, for
  example a hash of the thread id and `call_id`, never from tool arguments.
- A financial tool issues the charge from `closed_mandate_from_config(config)`, not one
  rebuilt from its arguments, and the reserve is recorded under its `charge_ref`.
- A node is LangGraph's unit of retry. A tool or `on_decision` error after some calls ran fails
  the node, and running the thread again re-runs every call of the turn. Give this node no
  `RetryPolicy` and make tools that move money idempotent (keyed by `charge_ref`).
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
    Decision,
    Deny,
    GateContext,
    RequireApproval,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "CLOSED_MANDATE_KEY",
    "ApprovalDecision",
    "ApprovalRequest",
    "GatedToolNode",
    "closed_mandate_from_config",
]

CLOSED_MANDATE_KEY = "agent_toolkit_closed_mandate"
"""Key in `configurable` of the config a tool runs with: the gate's checked `ClosedMandate`."""

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


class ApprovalRequest(BaseModel):
    """One call waiting for a human, as the trusted approval screen shows it.

    `args` are the validated arguments the call will run with; `closed_mandate_hash` is the
    `content_hash` of the charge a financial call will issue. `request_hash` is the SHA-256 hex
    of the canonical JSON (sorted keys, no spaces) of every other field, `reasons` included so
    an approval never covers a reason it did not show, plus the graph's thread id so it never
    covers another thread; an `ApprovalDecision` echoes it.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: Identifier
    tool: Identifier
    args: dict[str, object]
    reasons: tuple[Identifier, ...]
    closed_mandate_hash: str | None
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
        decisions = self._decide_in_order(calls, self._context(state))
        requests = _requests(calls, decisions, config)
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
            try:
                replies.append(self._run(outcome.call, outcome.decision, config))
            finally:  # A call that raised is reported too, then its error propagates.
                self._report(outcome)
        return {"messages": replies}

    def _decide_in_order(self, calls: Sequence[ToolCall], ctx: GateContext) -> list[Decision]:
        """Decide each call against the ledger plus the reserves of the turn's earlier charges."""
        decisions: list[Decision] = []
        for call in calls:
            decision = self._decide(call, ctx)
            decisions.append(decision)
            ctx = _with_reserve(ctx, decision)
        return decisions

    def _decide(self, call: ToolCall, ctx: GateContext) -> Decision:
        spec = self._specs.get(call.name)
        if spec is None or call.name not in self._tools:
            return Deny(reasons=("unknown_tool",))
        return decide(call, spec, ctx)

    def _run(
        self, call: ToolCall, decision: Allow | RequireApproval, config: RunnableConfig
    ) -> ToolMessage:
        validated = tool_call(name=call.name, args=_json_args(decision), id=call.id)
        # Always set, None included, so a value the caller put under the key never reaches a tool.
        checked = patch_config(config, configurable={CLOSED_MANDATE_KEY: decision.closed})
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


def _json_args(decision: Allow | RequireApproval) -> dict[str, object]:
    return decision.args.model_dump(mode="json")


def _requests(
    calls: Sequence[ToolCall], decisions: Sequence[Decision], config: RunnableConfig
) -> dict[str, ApprovalRequest]:
    pending = [
        (call, decision)
        for call, decision in zip(calls, decisions, strict=True)
        if isinstance(decision, RequireApproval)
    ]
    if not pending:
        return {}
    thread_id = _thread_id(config)
    return {call.id: _request(call, decision, thread_id) for call, decision in pending}


def _thread_id(config: RunnableConfig) -> str:
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id is None or not str(thread_id):
        raise ValueError(
            "approvals need config['configurable']['thread_id']: run the graph with a "
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
        request_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _ask(requests: list[ApprovalRequest]) -> dict[str, ApprovalDecision] | None:
    """The human's answers by call id: empty if nothing needs one, None if malformed."""
    if not requests:
        return {}
    resumed = interrupt(requests)
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
    return ToolMessage(
        content=f"Not executed: {', '.join(deny.reasons)}",
        tool_call_id=call.id,
        name=call.name,
        status="error",
    )

"""LangGraph tool node that runs the tool calls of a turn only as the policy gate allows.

Install with the `langgraph` extra. Put it where a graph would put `ToolNode`:
`builder.add_node("tools", GatedToolNode(tools, specs, context, on_decision=audit))`.

Each run reads the `tool_calls` of `state["messages"][-1]`, which must be an `AIMessage` (the
node never looks further back, so re-entering it cannot run a turn twice), and:

1. Builds the context with `context(state)` and decides every call with `decide` before
   running any. A call to a tool missing from `tools` or `specs` is denied (`unknown_tool`).
2. If calls need approval, pauses the graph with one `interrupt` whose value is the list of
   their `ApprovalRequest`s. Nothing has run yet.
3. On resume LangGraph re-runs the node from its start, so steps 1 and 2 repeat with a fresh
   context and `interrupt` returns the resume value, a `dict[call_id, ApprovalDecision]`. An
   approval counts only if its `request_hash` equals the one recomputed now, which binds the
   call id, tool, validated arguments, reasons and closed mandate. A resume value that is not
   such a mapping rejects every pending call (`approval_invalid`); a missing, mismatched or
   refused approval rejects its own call (`approval_missing`, `approval_mismatch`,
   `approval_rejected`).
4. Runs the allowed and approved calls one at a time, in order, with their validated
   arguments (`args.model_dump(mode="json")`, exactly what the approval showed). Every other
   call gets `ToolMessage(status="error", content="Not executed: <reasons>")`.
5. Calls `on_decision(call, decision, approval)` once per call with its final outcome: `Allow`,
   or `RequireApproval` with the approval that let it run, once the tool ran; or `Deny` for a
   call that does not run, before any tool runs. A rejection's reasons are its `approval_*`
   code followed by the gate's, and it carries the human's answer if there was one. The
   application wires it to `audit-log`; its exceptions propagate.

Resume with `graph.invoke(Command(resume={call_id: ApprovalDecision(...)}), config)`; JSON works
too (`at` takes an ISO 8601 string). LangGraph reads a dict whose keys all look like interrupt
ids (32 lowercase hex characters), the empty dict included, as answers keyed by interrupt id:
it answers nothing and the graph interrupts again. Key the answers by the interrupt to avoid
that, `Command(resume={interrupt.id: {call_id: ...}})`, as LangGraph requires anyway while
several interrupts are pending.

Duties of the application:

- Resume only from the trusted approval screen, never from chat input. The interrupt has no
  `response_schema`, so chat clients render no approval form.
- `context(state)` runs on every execution, the re-run on resume included. Hold the mandate
  lock from the first decision, across the interrupt, until the charge's reserve is recorded;
  without it the re-run re-verifies against whatever changed, and a call can end denied or
  gain reasons that void its approval.
- `ToolSpec.to_closed_mandate` must build the same charge on the re-run: the approval binds its
  `content_hash`, so a builder that mints a random `charge_ref` per call voids every approval
  of a financial call. Derive the ref from stable data.
- A node is LangGraph's unit of retry. A tool or `on_decision` error after some calls ran fails
  the node, and running the thread again re-runs every call of the turn. Give this node no
  `RetryPolicy` and make tools that move money idempotent (keyed by `charge_ref`).

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
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

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

__all__ = ["ApprovalDecision", "ApprovalRequest", "GatedToolNode"]

Identifier = Annotated[str, Field(min_length=1)]


class ApprovalRequest(BaseModel):
    """One call waiting for a human, as the trusted approval screen shows it.

    `args` are the validated arguments the call will run with; `closed_mandate_hash` is the
    `content_hash` of the charge a financial call will issue. `request_hash` is the SHA-256 hex
    of the canonical JSON (sorted keys, no spaces) of every other field; an `ApprovalDecision`
    echoes it.
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
class _Judged:
    call: ToolCall
    decision: Decision
    request: ApprovalRequest | None


@dataclass(frozen=True)
class _Outcome:
    call: ToolCall
    decision: Decision
    approval: ApprovalDecision | None


class GatedToolNode[StateT: Mapping[str, object]]:
    """A graph node that decides, asks a human when needed, then runs the turn's tool calls.

    `tools` run the calls and `specs` judge them, both keyed by tool name. `context` builds
    the `GateContext` from trusted state on every execution. See the module docstring for the
    algorithm and the application's duties. Raises ValueError for repeated tool names or a
    spec stored under another name than its own.
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
        self._tools = by_name
        self._specs = dict(specs)
        self._context = context
        self._on_decision = on_decision

    def __call__(self, state: StateT, config: RunnableConfig) -> dict[str, list[ToolMessage]]:
        calls = _tool_calls(state)
        if not calls:
            return {"messages": []}
        ctx = self._context(state)
        judged = [self._judge(call, ctx) for call in calls]
        answers = _ask([j.request for j in judged if j.request is not None])
        outcomes = [_resolve(j, answers) for j in judged]

        for outcome in outcomes:  # Report what will not run before anything runs.
            if isinstance(outcome.decision, Deny):
                self._report(outcome)
        replies: list[ToolMessage] = []
        for outcome in outcomes:
            if isinstance(outcome.decision, Deny):
                replies.append(_refusal(outcome.call, outcome.decision))
            else:
                replies.append(self._run(outcome.call, outcome.decision, config))
                self._report(outcome)
        return {"messages": replies}

    def _judge(self, call: ToolCall, ctx: GateContext) -> _Judged:
        spec = self._specs.get(call.name)
        if spec is None or call.name not in self._tools:
            return _Judged(call=call, decision=Deny(reasons=("unknown_tool",)), request=None)
        decision = decide(call, spec, ctx)
        request = _request(call, decision) if isinstance(decision, RequireApproval) else None
        return _Judged(call=call, decision=decision, request=request)

    def _run(
        self, call: ToolCall, decision: Allow | RequireApproval, config: RunnableConfig
    ) -> ToolMessage:
        validated = tool_call(name=call.name, args=_json_args(decision), id=call.id)
        output = self._tools[call.name].invoke(validated, config)
        if not isinstance(output, ToolMessage):
            raise TypeError(
                f"tool {call.name!r} returned {type(output).__name__}, not content or a ToolMessage"
            )
        return output

    def _report(self, outcome: _Outcome) -> None:
        if self._on_decision is not None:
            self._on_decision(outcome.call, outcome.decision, outcome.approval)


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


def _json_args(decision: Allow | RequireApproval) -> dict[str, object]:
    return decision.args.model_dump(mode="json")


def _request(call: ToolCall, decision: RequireApproval) -> ApprovalRequest:
    args = _json_args(decision)
    closed_hash = None if decision.closed is None else decision.closed.content_hash
    bound = {
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


def _resolve(judged: _Judged, answers: Mapping[str, ApprovalDecision] | None) -> _Outcome:
    call, decision, request = judged.call, judged.decision, judged.request
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

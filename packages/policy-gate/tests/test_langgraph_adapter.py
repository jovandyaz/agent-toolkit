"""`GatedToolNode`: a LangGraph tool node that runs only the tool calls the policy gate allows."""

import asyncio
import hashlib
import json
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.messages.tool import ToolCall as LcToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Interrupt
from pydantic import BaseModel, ConfigDict, TypeAdapter

from agent_toolkit import mandates
from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.budgets.adapters.langchain import BudgetCallbackHandler, BudgetExceededError
from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    ConsumptionEntry,
    OpenMandate,
)
from agent_toolkit.policy_gate import (
    Allow,
    BuildContext,
    Decision,
    Deny,
    Effects,
    GateConfig,
    GateContext,
    RequireApproval,
    Risk,
    SessionProfile,
    ToolCall,
    ToolSpec,
)
from agent_toolkit.policy_gate.adapters.langgraph import (
    CLOSED_MANDATE_KEY,
    FINGERPRINT_KEY,
    IDEMPOTENCY_KEY,
    ApprovalDecision,
    ApprovalRequest,
    GatedToolNode,
    RecordedReply,
    call_fingerprint_from_config,
    closed_mandate_from_config,
    idempotency_key_from_config,
)

NOW = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)
READS = Effects(reads_sensitive=True, changes_state=False, external_comm=False)
WRITES = Effects(reads_sensitive=False, changes_state=True, external_comm=False)
SPENDS = Effects(reads_sensitive=False, changes_state=True, external_comm=True)


class CustomerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str


class ChargeArgs(BaseModel):
    """What the gate judges: it normalizes the payer before the mandate sees it."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", str_strip_whitespace=True)

    payer: str
    amount_minor: int | None = None  # None: the payer sets the amount.
    previous_amount_minor: int | None = None  # An increase of the pending link `link_ref`.
    link_ref: str | None = None


class ToolChargeArgs(BaseModel):
    """The tool's own schema, which does not normalize."""

    payer: str
    amount_minor: int | None = None
    previous_amount_minor: int | None = None
    link_ref: str | None = None


Builder = Callable[[BaseModel, BuildContext], ClosedMandate]


def mint_ref(ctx: BuildContext) -> str:
    """A new charge's ref, minted server-side from the thread, message and call ids."""
    canonical = json.dumps([ctx.thread_id, ctx.message_id, ctx.call_id], separators=(",", ":"))
    return f"ch_{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"


@dataclass(frozen=True)
class StoredCharge:
    """A charge the application issued, as its own store keeps it."""

    payer: str
    receiving_account: str = "acct-own"
    rail: str = "codi_push"


STORED: Mapping[str, StoredCharge] = MappingProxyType(
    {"ch_link": StoredCharge(payer="cust-juan")}  # The pending link the increases raise.
)


def new_charge(
    args: BaseModel, charge_ref: str, stored: Mapping[str, StoredCharge] = STORED
) -> ClosedMandate:
    """A new charge under `charge_ref`, or an increase of the stored link the arguments name.

    An increase loads its payer, receiving account and rail from the stored charge, never from
    the arguments: `check` binds its previous amount to the ledger, but nothing else binds them.
    """
    assert isinstance(args, ChargeArgs)
    previous = args.previous_amount_minor
    if previous is None:
        ref, payer, account, rail = charge_ref, args.payer, "acct-own", "codi_push"
    else:
        ref = args.link_ref or ""
        link = stored.get(ref)
        if link is None:
            raise ValueError("an increase raises a charge the application stored")
        payer, account, rail = link.payer, link.receiving_account, link.rail
    return ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge" if previous is None else "amount_increase",
        charge_ref=ref,
        payer=payer,
        receiving_account=account,
        rail=rail,
        amount=None if args.amount_minor is None else mxn(args.amount_minor),
        previous_amount=None if previous is None else mxn(previous),
        reference="R-1",
    )


def build_charge(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    """The application's builder: it mints the ref from the call, so a re-run rebuilds it."""
    return new_charge(args, mint_ref(ctx))


def builder_over(stored: Mapping[str, StoredCharge]) -> Builder:
    """The application's builder, over another store of issued charges."""
    return lambda args, ctx: new_charge(args, mint_ref(ctx), stored)


def build_with_random_ref(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    """A careless builder: a fresh ref on every call, so the re-run builds another charge."""
    return new_charge(args, f"ch_{uuid.uuid4().hex}")


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


def plain_spec(name: str, risk: Risk, effects: Effects) -> ToolSpec:
    return ToolSpec(
        name=name,
        risk=risk,
        effects=effects,
        reversible=not effects.changes_state,
        financial=False,
        args_model=CustomerArgs,
        to_closed_mandate=None,
    )


def charge_spec(builder: Builder) -> ToolSpec:
    return ToolSpec(
        name="request_payment",
        risk="high",
        effects=SPENDS,
        reversible=False,
        financial=True,
        args_model=ChargeArgs,
        to_closed_mandate=builder,
    )


LOOKUP = plain_spec("lookup_customer", "low", READS)  # Allowed.
DELETE = plain_spec("delete_customer", "high", WRITES)  # Needs approval: high_risk.


@dataclass
class FakeTool:
    """A tool that records the arguments, closed mandate and idempotency key of every run.

    With `reserves_into`, it also records the reserve of the charge it issued, as the
    application's money tool must; with `replies_into`, its reply under its idempotency key,
    for a replay hook.
    """

    name: str
    args_schema: type[BaseModel] = CustomerArgs
    runs: list[dict[str, object]] = field(default_factory=list)
    closed_seen: list[ClosedMandate | None] = field(default_factory=list)
    keys_seen: list[str] = field(default_factory=list)
    fingerprints_seen: list[str] = field(default_factory=list)
    fails_with: Exception | None = None
    handles_errors: bool = False
    reserves_into: list[ConsumptionEntry] | None = None
    replies_into: dict[str, RecordedReply] | None = None

    def as_tool(self) -> BaseTool:
        def run(config: RunnableConfig, **kwargs: object) -> str:
            self.runs.append(kwargs)
            closed = closed_mandate_from_config(config)
            self.closed_seen.append(closed)
            self.keys_seen.append(idempotency_key_from_config(config))
            self.fingerprints_seen.append(call_fingerprint_from_config(config))
            if self.fails_with is not None:
                raise self.fails_with
            reserved = None if closed is None else mandates.reserve_amount(closed)
            if self.reserves_into is not None and closed is not None and reserved is not None:
                entry = ConsumptionEntry(
                    charge_ref=closed.charge_ref, kind="reserve", amount=reserved, at=NOW
                )
                self.reserves_into.append(entry)
            reply = f"{self.name} ran"
            if self.replies_into is not None:
                fingerprint = call_fingerprint_from_config(config)
                recorded = RecordedReply(fingerprint=fingerprint, content=reply)
                self.replies_into[idempotency_key_from_config(config)] = recorded
            return reply

        return StructuredTool.from_function(
            func=run,
            name=self.name,
            description=f"Fake {self.name}.",
            args_schema=self.args_schema,
            handle_tool_error=self.handles_errors,
        )


def session(**overrides: object) -> SessionProfile:
    fields: dict[str, object] = {
        "agent_id": "agent_1",
        "principal_id": "owner_1",
        "untrusted_input": False,
        "sensitive_access": False,
        "state_change": False,
    }
    return SessionProfile.model_validate(fields | overrides)


def mandate() -> OpenMandate:
    return OpenMandate(
        id="om_1",
        principal_id="owner_1",
        agent_id="agent_1",
        purpose="Collect payments from customers of the shop",
        amount_range=AmountRange(min=mxn(100), max=mxn(500000)),
        budget=Budget(limit=mxn(1000000), period="day"),
        receiving_accounts=frozenset({"acct-own"}),
        allowed_payers=frozenset({"cust-juan"}),
        allowed_rails=frozenset({"codi_push"}),
        daily_window=None,
        tz="America/Mexico_City",
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        revoked_at=None,
    )


def gate_context(**overrides: object) -> GateContext:
    fields: dict[str, object] = {
        "session": session(),
        "mandate": mandate(),
        "ledger": (),
        "budget_state": BudgetState.empty(),
        "budget_limits": BudgetLimits(),
        "now": NOW,
        "config": GateConfig(),
    }
    return GateContext.model_validate(fields | overrides)


def gate_ctx(state: MessagesState) -> GateContext:
    return gate_context()


Report = tuple[ToolCall, Decision, ApprovalDecision | None]


@dataclass
class Rig:
    """A one-node graph (START -> tools -> END) with a checkpointer, and what it observed."""

    lookup: FakeTool = field(default_factory=lambda: FakeTool("lookup_customer"))
    delete: FakeTool = field(default_factory=lambda: FakeTool("delete_customer"))
    charge: FakeTool = field(default_factory=lambda: FakeTool("request_payment", ToolChargeArgs))
    builder: Builder = build_charge
    context: GateContext = field(default_factory=gate_context)
    reports: list[Report] = field(default_factory=list)
    ran_at_report: list[tuple[str, int]] = field(default_factory=list)
    audit_error: Exception | None = None  # Raised by `record`, as a failing audit log would.
    recorded: list[ConsumptionEntry] = field(default_factory=list)  # Reserves tools recorded.
    stored: dict[str, RecordedReply] = field(default_factory=dict)  # Tools' replies, by key.
    replay: Callable[[str], RecordedReply | None] | None = None
    contexts_built: int = 0
    thread: RunnableConfig = field(
        default_factory=lambda: RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
    )

    def node(
        self, *, drop_spec: str | None = None, drop_tool: str | None = None
    ) -> GatedToolNode[MessagesState]:
        fakes = [self.lookup, self.delete, self.charge]
        specs = {
            spec.name: spec
            for spec in (LOOKUP, DELETE, charge_spec(self.builder))
            if spec.name != drop_spec
        }
        return GatedToolNode(
            tools=[fake.as_tool() for fake in fakes if fake.name != drop_tool],
            specs=specs,
            context=self.build_context,
            on_decision=self.record,
            replay=self.replay,
        )

    @property
    def thread_id(self) -> str:
        return str(self.thread["configurable"]["thread_id"])

    def built(self, args: ChargeArgs, call_id: str) -> ClosedMandate:
        """The charge the application's builder makes for `call_id` in this rig's thread."""
        ctx = BuildContext(thread_id=self.thread_id, message_id=MESSAGE_ID, call_id=call_id)
        return build_charge(args, ctx)

    def build_context(self, state: MessagesState) -> GateContext:
        self.contexts_built += 1
        if not self.recorded:
            return self.context
        ledger = (*(self.context.ledger or ()), *self.recorded)
        return self.context.model_copy(update={"ledger": ledger})

    def recording_reserves(self) -> "Rig":
        """This rig, with a charge tool that records its reserves into the context's ledger."""
        self.charge.reserves_into = self.recorded
        return self

    def replaying(self) -> "Rig":
        """This rig, with tools that store their replies and a replay hook that reads them."""
        for fake in (self.lookup, self.delete, self.charge):
            fake.replies_into = self.stored
        self.replay = self.stored.get
        return self.recording_reserves()

    def record(self, call: ToolCall, decision: Decision, approval: ApprovalDecision | None) -> None:
        if self.audit_error is not None:
            raise self.audit_error
        self.reports.append((call, decision, approval))
        self.ran_at_report.append((call.id, self.total_runs()))

    def total_runs(self) -> int:
        return len(self.lookup.runs) + len(self.delete.runs) + len(self.charge.runs)

    def graph(
        self, node: Any = None, *, durable: bool = True
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        builder = StateGraph(MessagesState)
        builder.add_node("tools", node if node is not None else self.node())
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        return builder.compile(checkpointer=InMemorySaver() if durable else None)


def call(name: str, call_id: str, **args: object) -> LcToolCall:
    return tool_call(name=name, args=args or {"customer_id": f"cust-{call_id}"}, id=call_id)


MESSAGE_ID = "msg_1"  # The id of a turn's AIMessage, unless a test gives another.


def turn(*calls: LcToolCall, message_id: str = MESSAGE_ID) -> dict[str, list[AnyMessage]]:
    return {"messages": [AIMessage(content="", tool_calls=list(calls), id=message_id)]}


REQUESTS = TypeAdapter(list[ApprovalRequest])


def pending(result: dict[str, Any]) -> list[ApprovalRequest]:
    """The requests of the turn's interrupt, read back from its JSON value."""
    (raised,) = result["__interrupt__"]
    assert isinstance(raised, Interrupt)
    assert isinstance(raised.value, list)
    assert all(isinstance(request, dict) for request in raised.value)
    return REQUESTS.validate_json(json.dumps(raised.value))


def answer(request: ApprovalRequest, *, approved: bool = True) -> ApprovalDecision:
    return ApprovalDecision(
        request_hash=request.request_hash,
        approved=approved,
        approver_id="owner_1",
        approval_id=f"apr_{request.call_id}",
        at=NOW,
    )


def replies(result: dict[str, Any]) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}


# The brief's tests.


def test_allowed_call_executes() -> None:
    rig = Rig()
    result = rig.graph().invoke(turn(call("lookup_customer", "c1")), rig.thread)

    assert "__interrupt__" not in result
    assert rig.lookup.runs == [{"customer_id": "cust-c1"}]
    reply = replies(result)["c1"]
    assert (reply.status, reply.content) == ("success", "lookup_customer ran")


def test_denied_call_returns_error_message_and_does_not_execute() -> None:
    rig = Rig()
    result = rig.graph().invoke(turn(call("lookup_customer", "c1", customer_id=7)), rig.thread)

    assert "__interrupt__" not in result
    assert rig.lookup.runs == []
    reply = replies(result)["c1"]
    assert reply.status == "error"
    assert reply.content == "Not executed: invalid_args:args.customer_id:string_type"


def test_approval_interrupts_then_executes_on_approved_resume() -> None:
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(turn(call("delete_customer", "c1")), rig.thread)

    (request,) = pending(first)
    assert request.call_id == "c1"
    assert request.tool == "delete_customer"
    assert request.args == {"customer_id": "cust-c1"}
    assert request.reasons == ("high_risk",)
    assert request.closed_mandate_hash is None
    assert request.closed is None
    assert len(request.request_hash) == 64
    assert rig.delete.runs == []

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]
    assert replies(resumed)["c1"].status == "success"


def test_interrupt_value_is_plain_json() -> None:
    # LangGraph requires a JSON-serializable interrupt value; the checkpoint keeps it as is.
    rig = Rig()
    graph = rig.graph()
    calls = turn(call("delete_customer", "c1"), call("request_payment", "c2", payer="cust-juan"))

    (raised,) = graph.invoke(calls, rig.thread)["__interrupt__"]

    value = raised.value
    assert json.loads(json.dumps(value)) == value
    assert [sorted(request) for request in value] == 2 * [
        ["args", "call_id", "closed", "closed_mandate_hash", "reasons", "request_hash", "tool"]
    ]
    assert value[0]["reasons"] == ["high_risk"]
    assert value[1]["closed"]["amount"] is None
    assert graph.get_state(rig.thread).interrupts[0].value == value


def test_approval_request_shows_the_charge_that_will_run() -> None:
    rig = Rig()
    graph = rig.graph()
    open_amount = call("request_payment", "c1", payer="cust-juan")
    (request,) = pending(graph.invoke(turn(open_amount), rig.thread))

    assert request.closed == rig.built(ChargeArgs(payer="cust-juan"), "c1")
    assert request.closed is not None
    assert request.closed.content_hash == request.closed_mandate_hash

    graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.charge.closed_seen == [request.closed]


def test_resume_executes_each_tool_once() -> None:
    # Review Focus 1: the node re-runs from its start on resume; nothing may run twice.
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(
        turn(call("lookup_customer", "c1"), call("delete_customer", "c2")), rig.thread
    )

    (request,) = pending(first)
    assert request.call_id == "c2"
    assert (rig.lookup.runs, rig.delete.runs) == ([], [])  # Nothing runs before the interrupt.

    resumed = graph.invoke(Command(resume={"c2": answer(request)}), rig.thread)

    assert rig.lookup.runs == [{"customer_id": "cust-c1"}]
    assert rig.delete.runs == [{"customer_id": "cust-c2"}]
    assert [m.tool_call_id for m in resumed["messages"] if isinstance(m, ToolMessage)] == [
        "c1",
        "c2",
    ]
    assert all(reply.status == "success" for reply in replies(resumed).values())


def test_resume_with_mismatched_hash_rejects() -> None:
    # Review Focus 2: an approval given for other arguments must not run this call.
    stale, rig = Rig(), Rig()
    (stale_request,) = pending(
        stale.graph().invoke(turn(call("delete_customer", "c1", customer_id="a")), stale.thread)
    )
    graph = rig.graph()
    (request,) = pending(
        graph.invoke(turn(call("delete_customer", "c1", customer_id="b")), rig.thread)
    )
    assert stale_request.request_hash != request.request_hash

    resumed = graph.invoke(Command(resume={"c1": answer(stale_request)}), rig.thread)

    assert rig.delete.runs == []
    reply = replies(resumed)["c1"]
    assert reply.status == "error"
    assert reply.content == "Not executed: approval_mismatch, high_risk"


def test_on_decision_called_once_per_call() -> None:
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(
        turn(
            call("lookup_customer", "c1"),
            call("delete_customer", "c2"),
            call("lookup_customer", "c3", customer_id=7),
            call("delete_customer", "c4"),
        ),
        rig.thread,
    )
    approve, reject = pending(first)
    assert rig.reports == []  # An interrupted run reports nothing; the resumed one reports all.

    graph.invoke(
        Command(resume={"c2": answer(approve), "c4": answer(reject, approved=False)}),
        rig.thread,
    )

    reported = {call.id: (decision, approval) for call, decision, approval in rig.reports}
    assert len(rig.reports) == len(reported) == 4
    assert isinstance(reported["c1"][0], Allow)
    assert reported["c1"][1] is None
    assert isinstance(reported["c2"][0], RequireApproval)
    assert reported["c2"][1] == answer(approve)
    assert reported["c3"] == (Deny(reasons=("invalid_args:args.customer_id:string_type",)), None)
    assert reported["c4"] == (
        Deny(reasons=("approval_rejected", "high_risk")),
        answer(reject, approved=False),
    )
    # Calls that do not run are reported before any tool runs; the others right before they run.
    assert rig.ran_at_report == [("c3", 0), ("c4", 0), ("c1", 0), ("c2", 1)]


# Resume values.


def _missing_field(decision: ApprovalDecision) -> object:
    return {"c2": decision.model_dump(exclude={"approver_id"})}


def _string_bool(decision: ApprovalDecision) -> object:
    return {"c2": decision.model_dump() | {"approved": "true"}}


def _extra_field(decision: ApprovalDecision) -> object:
    return {"c2": decision.model_dump() | {"scope": "all"}}


def _naive_time(decision: ApprovalDecision) -> object:
    return {"c2": decision.model_dump() | {"at": datetime(2026, 12, 1, 18, 0)}}


def _one_bad_entry(decision: ApprovalDecision) -> object:
    return {"c2": decision, "c3": "approved"}


@pytest.mark.parametrize(
    "malformed",
    [
        lambda decision: "approved",
        lambda decision: True,
        lambda decision: [decision],
        lambda decision: [("c2", decision)],
        lambda decision: {"c2": "approved"},
        _missing_field,
        _string_bool,
        _extra_field,
        _naive_time,
        _one_bad_entry,
    ],
)
def test_malformed_resume_rejects_every_pending_call(
    malformed: Callable[[ApprovalDecision], object],
) -> None:
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(
        turn(
            call("lookup_customer", "c1"),
            call("delete_customer", "c2"),
            call("delete_customer", "c3"),
        ),
        rig.thread,
    )
    approve, _ = pending(first)

    resumed = graph.invoke(Command(resume=malformed(answer(approve))), rig.thread)

    assert rig.lookup.runs == [{"customer_id": "cust-c1"}]  # Not pending: runs as allowed.
    assert rig.delete.runs == []
    for call_id in ("c2", "c3"):
        reply = replies(resumed)[call_id]
        assert reply.status == "error"
        assert reply.content == "Not executed: approval_invalid, high_risk"


def test_json_resume_is_accepted() -> None:
    # An approval screen that posts JSON sends the timestamp as an ISO 8601 string.
    rig = Rig()
    graph = rig.graph()
    (request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), rig.thread))

    graph.invoke(Command(resume={"c1": answer(request).model_dump(mode="json")}), rig.thread)

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]


def test_missing_approval_rejects_that_call_only() -> None:
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(
        turn(call("delete_customer", "c1"), call("delete_customer", "c2")), rig.thread
    )
    request, _ = pending(first)

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]
    assert replies(resumed)["c2"].content == "Not executed: approval_missing, high_risk"


def test_resume_keyed_by_interrupt_id_is_accepted() -> None:
    rig = Rig()
    graph = rig.graph()
    first = graph.invoke(turn(call("delete_customer", "c1")), rig.thread)
    (raised,) = first["__interrupt__"]
    (request,) = pending(first)

    graph.invoke(Command(resume={raised.id: {"c1": answer(request)}}), rig.thread)

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]


def test_empty_resume_mapping_interrupts_again() -> None:
    # LangGraph reads an empty mapping as a map of interrupt ids that answers none of them.
    rig = Rig()
    graph = rig.graph()
    graph.invoke(turn(call("lookup_customer", "c1"), call("delete_customer", "c2")), rig.thread)

    again = graph.invoke(Command(resume={}), rig.thread)

    assert [request.call_id for request in pending(again)] == ["c2"]
    assert (rig.lookup.runs, rig.delete.runs, rig.reports) == ([], [], [])


# What the re-run on resume re-checks.


def test_resume_re_decides_with_a_fresh_context() -> None:
    rig = Rig()
    graph = rig.graph()
    (request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), rig.thread))
    rig.context = gate_context(
        budget_state=BudgetState.empty().model_copy(update={"tool_calls": 1}),
        budget_limits=BudgetLimits(max_tool_calls=0),
    )

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.contexts_built == 2
    assert rig.delete.runs == []
    assert replies(resumed)["c1"].content == "Not executed: budget_exceeded:tool_calls"


def test_approval_does_not_cover_new_reasons() -> None:
    rig = Rig()
    graph = rig.graph()
    (request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), rig.thread))
    rig.context = gate_context(
        session=session(untrusted_input=True, sensitive_access=True, state_change=True)
    )

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.delete.runs == []
    assert replies(resumed)["c1"].content == (
        "Not executed: approval_mismatch, high_risk, rule_of_two"
    )


def test_deterministic_builder_keeps_the_approval_valid_on_the_re_run() -> None:
    rig = Rig()
    graph = rig.graph()
    open_amount = call("request_payment", "c1", payer="cust-juan")
    (request,) = pending(graph.invoke(turn(open_amount), rig.thread))
    assert request.reasons == ("mandate:open_amount",)
    checked = rig.built(ChargeArgs(payer="cust-juan"), "c1")
    assert request.closed_mandate_hash == checked.content_hash

    graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.charge.closed_seen == [checked]


def test_builder_minting_a_random_ref_fails_closed() -> None:
    rig = Rig(builder=build_with_random_ref)
    graph = rig.graph()
    open_amount = call("request_payment", "c1", payer="cust-juan")
    (request,) = pending(graph.invoke(turn(open_amount), rig.thread))

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.charge.runs == []
    assert replies(resumed)["c1"].content == (
        "Not executed: approval_mismatch, mandate:open_amount"
    )


def test_builder_gets_the_thread_message_and_call_ids() -> None:
    seen: list[BuildContext] = []

    def recording(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        seen.append(ctx)
        return build_charge(args, ctx)

    rig = Rig(builder=recording)
    rig.graph().invoke(
        turn(call("request_payment", "c1", payer="cust-juan", amount_minor=35000)), rig.thread
    )

    assert seen
    assert set(seen) == {BuildContext(thread_id=rig.thread_id, message_id=MESSAGE_ID, call_id="c1")}
    assert rig.charge.closed_seen == [
        rig.built(ChargeArgs(payer="cust-juan", amount_minor=35000), "c1")
    ]


def test_a_charge_rebuilt_onto_a_ref_the_turn_took_is_denied() -> None:
    # A builder that is not deterministic: the charge `decide` checks for c2 lands on c1's ref,
    # although the ref built to plan the turn did not. It must not run as a second charge there.
    builds: Counter[str] = Counter()

    def drifting(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        builds[ctx.call_id] += 1
        if ctx.call_id == "c2" and builds["c2"] % 2:
            return new_charge(args, "ch_fresh")
        return new_charge(args, "ch_shared")

    rig = Rig(builder=drifting)
    graph = rig.graph()
    calls = turn(
        call("request_payment", "c1", payer="cust-juan"),  # Open amount: it reserves nothing.
        call("request_payment", "c2", payer="cust-juan", amount_minor=35000),
    )
    (request,) = pending(graph.invoke(calls, rig.thread))
    assert request.call_id == "c1"

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert [closed.charge_ref for closed in rig.charge.closed_seen if closed] == ["ch_shared"]
    assert replies(resumed)["c2"].content == "Not executed: charge_ref_in_turn"
    assert ("c2", Deny(reasons=("charge_ref_in_turn",))) in [(c.id, d) for c, d, _ in rig.reports]


def test_tools_receive_the_checked_closed_mandate_via_config() -> None:
    # Whatever the caller put under the key, each tool gets the charge checked for its call.
    rig = Rig()
    graph = rig.graph()
    in_t1 = BuildContext(thread_id="t1", message_id=MESSAGE_ID, call_id="forged")
    forged = build_charge(ChargeArgs(payer="cust-juan", amount_minor=500000), in_t1)
    config = RunnableConfig(configurable={"thread_id": "t1", CLOSED_MANDATE_KEY: forged})
    calls = turn(
        call("request_payment", "c1", payer="cust-juan"),  # Open amount: needs approval.
        call("request_payment", "c2", payer="cust-juan", amount_minor=35000),  # Allowed.
        call("lookup_customer", "c3"),
    )
    (request,) = pending(graph.invoke(calls, config))

    graph.invoke(Command(resume={"c1": answer(request)}), config)

    checked = {c.id: d.closed for c, d, _ in rig.reports if not isinstance(d, Deny)}
    c1, c2 = (in_t1.model_copy(update={"call_id": call_id}) for call_id in ("c1", "c2"))
    assert checked["c1"] == build_charge(ChargeArgs(payer="cust-juan"), c1)
    assert checked["c1"].content_hash == request.closed_mandate_hash
    assert checked["c2"] == build_charge(ChargeArgs(payer="cust-juan", amount_minor=35000), c2)
    assert rig.charge.closed_seen == [checked["c1"], checked["c2"]]
    assert rig.lookup.closed_seen == [None]


def test_closed_mandate_from_config_accepts_only_a_closed_mandate() -> None:
    assert closed_mandate_from_config(RunnableConfig()) is None
    with pytest.raises(TypeError, match="ClosedMandate"):
        closed_mandate_from_config(
            RunnableConfig(configurable={CLOSED_MANDATE_KEY: {"charge_ref": "ch_x"}})
        )


def test_approved_financial_call_runs_with_the_validated_args() -> None:
    # The tool gets the arguments the gate judged and the human saw, not the model's text.
    rig = Rig()
    graph = rig.graph()
    (request,) = pending(
        graph.invoke(turn(call("request_payment", "c1", payer=" cust-juan ")), rig.thread)
    )
    assert request.args == {
        "payer": "cust-juan",
        "amount_minor": None,
        "previous_amount_minor": None,
        "link_ref": None,
    }

    graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.charge.runs == [request.args]
    _, decision, approval = rig.reports[0]
    assert isinstance(decision, RequireApproval)
    assert decision.closed is not None
    assert decision.closed.content_hash == request.closed_mandate_hash
    assert approval == answer(request)


def test_charges_in_one_turn_share_the_mandate_budget() -> None:
    # Each charge is judged with the reserves of the turn's earlier charges: 3 x 400k > 1M.
    rig = Rig()
    graph = rig.graph()
    charges = [
        call("request_payment", f"c{n}", payer="cust-juan", amount_minor=400000) for n in (1, 2, 3)
    ]
    (request,) = pending(graph.invoke(turn(*charges), rig.thread))
    assert (request.call_id, request.reasons) == ("c3", ("mandate:budget_exceeded",))
    assert rig.charge.runs == []

    resumed = graph.invoke(Command(resume={"c3": answer(request, approved=False)}), rig.thread)

    assert [run["amount_minor"] for run in rig.charge.runs] == [400000, 400000]
    assert replies(resumed)["c3"].content == (
        "Not executed: approval_rejected, mandate:budget_exceeded"
    )


def test_charges_in_one_turn_share_the_budget_when_violations_deny() -> None:
    rig = Rig(context=gate_context(config=GateConfig(on_mandate_violation="deny")))
    charges = [
        call("request_payment", f"c{n}", payer="cust-juan", amount_minor=400000) for n in (1, 2, 3)
    ]

    result = rig.graph().invoke(turn(*charges), rig.thread)

    assert "__interrupt__" not in result
    assert len(rig.charge.runs) == 2
    assert replies(result)["c3"].content == "Not executed: mandate:budget_exceeded"


def test_a_charge_waiting_for_approval_also_reserves_budget() -> None:
    # It may still run once approved, so the charges after it are judged as if it will.
    rig = Rig()
    charges = turn(
        call("request_payment", "c1", payer="cust-ana", amount_minor=400000),
        call("request_payment", "c2", payer="cust-juan", amount_minor=400000),
        call("request_payment", "c3", payer="cust-juan", amount_minor=400000),
    )

    requests = pending(rig.graph().invoke(charges, rig.thread))

    assert [(request.call_id, request.reasons) for request in requests] == [
        ("c1", ("mandate:payer_not_allowed",)),
        ("c3", ("mandate:budget_exceeded",)),
    ]


def increase(
    call_id: str, previous_minor: int, amount_minor: int, payer: str = "cust-juan"
) -> LcToolCall:
    return call(
        "request_payment",
        call_id,
        payer=payer,
        amount_minor=amount_minor,
        previous_amount_minor=previous_minor,
        link_ref="ch_link",
    )


def link_reserved(minor: int) -> GateContext:
    """A context whose ledger holds the pending link `ch_link` at `minor`."""
    link = ConsumptionEntry(charge_ref="ch_link", kind="reserve", amount=mxn(minor), at=NOW)
    return gate_context(ledger=(link,))


def test_two_increases_of_one_link_in_one_turn() -> None:
    # One charge changes at most once per turn: the second step waits for the next turn.
    rig = Rig(context=link_reserved(30000))

    result = rig.graph().invoke(
        turn(increase("c1", 30000, 45000), increase("c2", 30000, 50000)), rig.thread
    )

    assert "__interrupt__" not in result
    assert [run["amount_minor"] for run in rig.charge.runs] == [45000]
    assert replies(result)["c2"].content == "Not executed: charge_ref_in_turn"


@pytest.mark.parametrize("approved", [True, False])
def test_a_step_after_a_pending_change_of_its_charge_never_runs(approved: bool) -> None:
    # B only matches the link if A runs; rejecting A must not leave B raising it from 40k.
    # The link was issued to cust-ana, outside the mandate's payers, so A needs approval.
    rig = Rig(
        context=link_reserved(40000),
        builder=builder_over({"ch_link": StoredCharge(payer="cust-ana")}),
    )
    graph = rig.graph()
    first = graph.invoke(
        turn(increase("a", 40000, 50000), increase("b", 50000, 60000)),
        rig.thread,
    )
    (request,) = pending(first)
    assert (request.call_id, request.reasons) == ("a", ("mandate:payer_not_allowed",))

    resumed = graph.invoke(Command(resume={"a": answer(request, approved=approved)}), rig.thread)

    ran = [run["amount_minor"] for run in rig.charge.runs]
    assert ran == ([50000] if approved else [])
    assert replies(resumed)["b"].content == "Not executed: charge_ref_in_turn"


def test_an_increase_of_a_charge_created_in_the_same_turn_is_denied() -> None:
    # The builder loads an increase's charge from the store, which knows c1 only once it ran,
    # so c2 cannot even be built (`charge_ref_in_turn` backs this up for builders that can).
    rig = Rig()
    created = call("request_payment", "c1", payer="cust-juan", amount_minor=35000)
    raised = call(
        "request_payment",
        "c2",
        payer="cust-juan",
        amount_minor=45000,
        previous_amount_minor=35000,
        link_ref=rig.built(ChargeArgs(payer="cust-juan"), "c1").charge_ref,  # c1's ref.
    )

    result = rig.graph().invoke(turn(created, raised), rig.thread)

    assert [run["amount_minor"] for run in rig.charge.runs] == [35000]
    assert replies(result)["c2"].content == "Not executed: invalid_args:closed:value_error"


@pytest.mark.parametrize("named_payer", ["cust-juan", "cust-stranger"])
def test_an_increase_takes_its_payer_from_the_stored_charge(named_payer: str) -> None:
    # The link went to a payer outside the mandate, approved by a human. Naming an allowed
    # payer in the increase's arguments must not skip the approval its real payer needs.
    rig = Rig(
        context=link_reserved(10000),
        builder=builder_over({"ch_link": StoredCharge(payer="cust-stranger")}),
    )
    graph = rig.graph()

    (request,) = pending(
        graph.invoke(turn(increase("c1", 10000, 50000, payer=named_payer)), rig.thread)
    )

    assert request.reasons == ("mandate:payer_not_allowed",)
    assert request.closed is not None
    assert request.closed.payer == "cust-stranger"
    graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)
    assert [closed.payer for closed in rig.charge.closed_seen if closed] == ["cust-stranger"]


def test_charges_on_different_refs_in_one_turn_are_judged_normally() -> None:
    rig = Rig(context=link_reserved(30000))
    created = call("request_payment", "c2", payer="cust-juan", amount_minor=35000)

    result = rig.graph().invoke(turn(increase("c1", 30000, 45000), created), rig.thread)

    assert "__interrupt__" not in result
    assert [run["amount_minor"] for run in rig.charge.runs] == [45000, 35000]
    assert all(reply.status == "success" for reply in replies(result).values())


def test_approval_does_not_carry_over_to_another_thread() -> None:
    # Same call id and arguments in another conversation: the approval must not apply there.
    first, other = Rig(), Rig()
    (request,) = pending(first.graph().invoke(turn(call("delete_customer", "c1")), first.thread))
    graph = other.graph()
    (other_request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), other.thread))
    assert other_request.request_hash != request.request_hash

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), other.thread)

    assert other.delete.runs == []
    assert replies(resumed)["c1"].content == "Not executed: approval_mismatch, high_risk"


def test_approval_needs_a_thread_id() -> None:
    rig = Rig()

    with pytest.raises(ValueError, match="thread_id"):
        rig.graph(durable=False).invoke(turn(call("delete_customer", "c1")))
    assert rig.total_runs() == 0


@pytest.mark.parametrize(
    "proposed",
    [
        call("lookup_customer", "c1"),
        call("request_payment", "c1", payer="cust-juan", amount_minor=35000),
    ],
)
def test_every_turn_needs_a_thread_id(proposed: LcToolCall) -> None:
    # Builders mint refs from the thread id: without one, nothing is decided or run.
    rig = Rig()

    with pytest.raises(ValueError, match="thread_id"):
        rig.graph(durable=False).invoke(turn(proposed))
    assert (rig.total_runs(), rig.contexts_built, rig.reports) == (0, 0, [])


def test_a_lock_around_the_node_covers_one_execution() -> None:
    # The README recipe: the mandate's critical section is one execution of the node, from
    # `context(state)` until the tools have run. An interrupt raises out of the node, which
    # releases the lock; the re-run on resume takes it again.
    rig = Rig()
    lock = threading.Lock()
    held: list[bool] = []

    def locked_context(state: MessagesState) -> GateContext:
        held.append(lock.locked())
        return gate_context()

    node = GatedToolNode(
        tools=[rig.delete.as_tool()], specs={"delete_customer": DELETE}, context=locked_context
    )

    def gated_tools(state: MessagesState, config: RunnableConfig) -> dict[str, list[ToolMessage]]:
        with lock:
            return node(state, config)

    graph = rig.graph(gated_tools)
    (request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), rig.thread))
    assert not lock.locked()

    graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert not lock.locked()
    assert held == [True, True]
    assert rig.delete.runs == [{"customer_id": "cust-c1"}]


# Idempotency keys and re-runs after a partial failure.


def expected_key(thread_id: str, call_id: str, message_id: str = MESSAGE_ID) -> str:
    """The documented key: SHA-256 hex of the compact JSON [thread_id, message_id, call_id]."""
    canonical = json.dumps([thread_id, message_id, call_id], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_each_call_gets_an_idempotency_key_from_its_thread_and_id() -> None:
    # Whatever the caller put under the key, each tool gets the key of its own call.
    rig = Rig()
    config = RunnableConfig(configurable={"thread_id": rig.thread_id, IDEMPOTENCY_KEY: "forged"})

    rig.graph().invoke(turn(call("lookup_customer", "c1"), call("lookup_customer", "c2")), config)

    assert rig.lookup.keys_seen == [expected_key(rig.thread_id, c) for c in ("c1", "c2")]
    assert expected_key(rig.thread_id, "c1") != expected_key("another-thread", "c1")
    assert expected_key(rig.thread_id, "c1") != expected_key(rig.thread_id, "c1", "msg_2")


def test_idempotency_key_from_config_needs_the_node() -> None:
    with pytest.raises(ValueError, match="GatedToolNode"):
        idempotency_key_from_config(RunnableConfig())
    with pytest.raises(TypeError, match="str"):
        idempotency_key_from_config(RunnableConfig(configurable={IDEMPOTENCY_KEY: 7}))
    with pytest.raises(ValueError, match="GatedToolNode"):
        call_fingerprint_from_config(RunnableConfig())
    with pytest.raises(TypeError, match="str"):
        call_fingerprint_from_config(RunnableConfig(configurable={FINGERPRINT_KEY: 7}))


def expected_fingerprint(tool: str, args: dict[str, object], closed: ClosedMandate | None) -> str:
    """The documented fingerprint: tool, validated arguments and the charge's content hash."""
    content = {
        "tool": tool,
        "args": args,
        "closed_mandate_hash": None if closed is None else closed.content_hash,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_each_call_gets_a_fingerprint_of_its_tool_arguments_and_charge() -> None:
    rig = Rig()
    config = RunnableConfig(configurable={"thread_id": rig.thread_id, FINGERPRINT_KEY: "forged"})
    charge = call("request_payment", "c2", payer=" cust-juan ", amount_minor=35000)

    rig.graph().invoke(turn(call("lookup_customer", "c1"), charge), config)

    charge_args = ChargeArgs(payer="cust-juan", amount_minor=35000)
    assert rig.lookup.fingerprints_seen == [
        expected_fingerprint("lookup_customer", {"customer_id": "cust-c1"}, None)
    ]
    assert rig.charge.fingerprints_seen == [
        expected_fingerprint(
            "request_payment", charge_args.model_dump(mode="json"), rig.built(charge_args, "c2")
        )
    ]


def fail_once(rig: Rig) -> None:
    rig.lookup.fails_with = RuntimeError("provider down")


@pytest.mark.parametrize(
    ("context", "charge", "reason"),
    [
        pytest.param(
            gate_context(),
            call("request_payment", "c1", payer="cust-juan", amount_minor=35000),
            "mandate:charge_ref_reused",
            id="new_charge",
        ),
        pytest.param(
            link_reserved(30000),
            increase("c1", 30000, 45000),
            "mandate:previous_amount_mismatch",
            id="amount_increase",
        ),
    ],
)
def test_re_run_after_a_recorded_reserve_tells_the_model_to_check_the_charge(
    context: GateContext, charge: LcToolCall, reason: str
) -> None:
    # c1 ran and recorded its reserve, then c2 failed, failing the node. Running the thread
    # again judges c1 against a ledger that already holds its reserve: it is denied, and the
    # model must not be told that a charge which exists was never executed.
    rig = Rig(context=context).recording_reserves()
    fail_once(rig)
    graph = rig.graph()
    with pytest.raises(RuntimeError, match="provider down"):
        graph.invoke(turn(charge, call("lookup_customer", "c2")), rig.thread)
    assert (len(rig.charge.runs), len(rig.recorded)) == (1, 1)
    rig.lookup.fails_with = None
    rig.reports.clear()

    again = graph.invoke(None, rig.thread)

    assert len(rig.charge.runs) == 1
    reply = replies(again)["c1"]
    assert reply.status == "error"
    assert (
        reply.content
        == f"Check the charge's status before retrying, it may already exist: {reason}"
    )
    assert [(c.id, d.kind) for c, d, _ in rig.reports] == [("c1", "deny"), ("c2", "allow")]
    assert rig.reports[0][1] == Deny(reasons=(reason,))
    assert replies(again)["c2"].status == "success"


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        pytest.param(gate_context(mandate=None, ledger=None), "no_mandate", id="no_mandate"),
        pytest.param(gate_context(), "mandate:open_amount", id="open_amount"),
    ],
)
def test_re_run_after_an_approved_call_without_a_reserve_runs_it_again_under_its_key(
    context: GateContext, reason: str
) -> None:
    # Nothing on the ledger shows that c1 ran, and LangGraph replays the stored approval, so
    # the re-run runs c1 again: with the same idempotency key, which the tool must honour.
    rig = Rig(context=context).recording_reserves()
    fail_once(rig)
    graph = rig.graph()
    charge = call("request_payment", "c1", payer="cust-juan")
    (request,) = pending(graph.invoke(turn(charge, call("lookup_customer", "c2")), rig.thread))
    assert request.reasons == (reason,)
    with pytest.raises(RuntimeError, match="provider down"):
        graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)
    rig.lookup.fails_with = None

    again = graph.invoke(None, rig.thread)

    assert len(rig.charge.runs) == 2
    assert rig.charge.keys_seen == 2 * [expected_key(rig.thread_id, "c1")]
    assert replies(again)["c1"].status == "success"


@pytest.mark.parametrize(
    ("context", "charge"),
    [
        pytest.param(
            gate_context(),
            call("request_payment", "c1", payer="cust-juan", amount_minor=35000),
            id="new_charge",
        ),
        pytest.param(link_reserved(30000), increase("c1", 30000, 45000), id="amount_increase"),
        pytest.param(
            gate_context(mandate=None, ledger=None),
            call("request_payment", "c1", payer="cust-juan", amount_minor=35000),
            id="no_mandate",
        ),
        pytest.param(
            gate_context(), call("request_payment", "c1", payer="cust-juan"), id="open_amount"
        ),
    ],
)
def test_replay_answers_a_call_that_already_ran(context: GateContext, charge: LcToolCall) -> None:
    # With the hook, the re-run neither denies c1 nor runs it again: it replies what c1 stored.
    rig = Rig(context=context).replaying()
    fail_once(rig)
    graph = rig.graph()
    with pytest.raises(RuntimeError, match="provider down"):
        first = graph.invoke(turn(charge, call("lookup_customer", "c2")), rig.thread)
        # Only a call that needs approval gets here; the approved resume then fails.
        graph.invoke(Command(resume={r.call_id: answer(r) for r in pending(first)}), rig.thread)
    assert len(rig.charge.runs) == 1
    rig.lookup.fails_with = None
    rig.reports.clear()

    again = graph.invoke(None, rig.thread)

    assert len(rig.charge.runs) == 1
    reply = replies(again)["c1"]
    assert (reply.status, reply.content, reply.name) == (
        "success",
        "request_payment ran",
        "request_payment",
    )
    assert [(c.id, d.kind) for c, d, _ in rig.reports] == [("c2", "allow")]  # c1 not again.
    assert replies(again)["c2"].status == "success"


def test_replay_never_runs_a_call_and_leaves_the_others_to_the_gate() -> None:
    # The node never runs a call the hook answers; every call it does not answer is gated.
    rig = Rig()
    asked: list[str] = []

    ran = RecordedReply(
        fingerprint=expected_fingerprint("delete_customer", {"customer_id": "cust-c1"}, None),
        content="recorded earlier",
    )

    def replay(key: str) -> RecordedReply | None:
        asked.append(key)
        return ran if key == expected_key(rig.thread_id, "c1") else None

    rig.replay = replay
    graph = rig.graph()
    first = graph.invoke(
        turn(call("delete_customer", "c1"), call("delete_customer", "c2")), rig.thread
    )
    (request,) = pending(first)
    assert request.call_id == "c2"  # c1 is neither decided nor put to a human.

    resumed = graph.invoke(Command(resume={"c2": answer(request, approved=False)}), rig.thread)

    assert rig.delete.runs == []
    assert replies(resumed)["c1"].content == "recorded earlier"
    assert replies(resumed)["c2"].content == "Not executed: approval_rejected, high_risk"
    assert [c.id for c, _, _ in rig.reports] == ["c2"]
    assert asked[:2] == [expected_key(rig.thread_id, call_id) for call_id in ("c1", "c2")]


def test_a_replayed_change_still_takes_its_charge_for_the_turn() -> None:
    # c2 changes the charge c1 already changed: it is denied on the re-run as on the first run,
    # although the ledger now holds the reserve c2 claims as its previous amount.
    rig = Rig(context=link_reserved(30000)).replaying()
    fail_once(rig)
    graph = rig.graph()
    calls = turn(
        increase("c1", 30000, 45000), increase("c2", 45000, 50000), call("lookup_customer", "c3")
    )
    with pytest.raises(RuntimeError, match="provider down"):
        graph.invoke(calls, rig.thread)
    rig.lookup.fails_with = None

    again = graph.invoke(None, rig.thread)

    assert [run["amount_minor"] for run in rig.charge.runs] == [45000]
    assert replies(again)["c2"].content == "Not executed: charge_ref_in_turn"


def test_replay_must_return_a_recorded_reply() -> None:
    rig = Rig()
    rig.replay = lambda key: "lookup_customer ran"  # type: ignore[assignment,return-value]

    with pytest.raises(TypeError, match="RecordedReply"):
        rig.graph().invoke(turn(call("lookup_customer", "c1")), rig.thread)
    assert rig.total_runs() == 0


@pytest.mark.parametrize(
    ("first", "then"),
    [
        pytest.param(
            call("lookup_customer", "call_0"), call("lookup_customer", "call_0"), id="same_lookup"
        ),
        pytest.param(
            call("lookup_customer", "call_0", customer_id="cust-1"),
            call("lookup_customer", "call_0", customer_id="cust-2"),
            id="other_lookup",
        ),
        pytest.param(
            call("request_payment", "call_0", payer="cust-juan", amount_minor=35000),
            call("request_payment", "call_0", payer="cust-juan", amount_minor=35000),
            id="same_charge",
        ),
        pytest.param(
            call("request_payment", "call_0", payer="cust-juan", amount_minor=35000),
            call("request_payment", "call_0", payer="cust-juan", amount_minor=50000),
            id="other_charge",
        ),
    ],
)
def test_a_later_turn_that_reuses_a_call_id_is_decided_and_runs(
    first: LcToolCall, then: LcToolCall
) -> None:
    # Some providers number calls per message (`call_0`). A later turn's call is a new call,
    # even with the same arguments (a second invoice for the same amount): its key and its
    # charge's ref come from its own message, so it is decided, reported and run, not replayed.
    rig = Rig().replaying()
    graph = rig.graph()
    graph.invoke(turn(first), rig.thread)
    ran_before = rig.total_runs()
    rig.reports.clear()

    later = graph.invoke(turn(then, message_id="msg_2"), rig.thread)

    assert (ran_before, rig.total_runs()) == (1, 2)
    assert replies(later)["call_0"].status == "success"
    assert [(c.id, d.kind) for c, d, _ in rig.reports] == [("call_0", "allow")]


def test_approval_does_not_carry_over_to_a_later_turn() -> None:
    # The same call id and arguments in a later message of the thread: a new request.
    rig = Rig()
    graph = rig.graph()
    (request,) = pending(graph.invoke(turn(call("delete_customer", "c1")), rig.thread))
    graph.invoke(Command(resume={"c1": answer(request, approved=False)}), rig.thread)
    (later,) = pending(
        graph.invoke(turn(call("delete_customer", "c1"), message_id="msg_2"), rig.thread)
    )
    assert later.request_hash != request.request_hash

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.delete.runs == []
    assert replies(resumed)["c1"].content == "Not executed: approval_mismatch, high_risk"


def test_a_turn_without_a_message_id_raises() -> None:
    # A custom state without `add_messages` leaves the AIMessage without an id, and the call
    # keys need it: fail before building the context, deciding or running anything.
    rig = Rig()
    state: MessagesState = {
        "messages": [AIMessage(content="", tool_calls=[call("lookup_customer", "c1")])]
    }

    with pytest.raises(ValueError, match="message id"):
        rig.node()(state, rig.thread)
    assert (rig.total_runs(), rig.contexts_built, rig.reports) == (0, 0, [])


CALL_ID_REUSED_REPLY = (
    "Check this call's result before retrying, it may already have run: call_id_reused"
)


def test_a_record_with_another_fingerprint_is_denied() -> None:
    # Under the call's own key, but for another request: the hook answered for another call.
    rig = Rig()
    rig.replay = lambda key: RecordedReply(fingerprint="0" * 64, content="someone else's reply")

    result = rig.graph().invoke(turn(call("lookup_customer", "c1")), rig.thread)

    assert rig.lookup.runs == []
    reply = replies(result)["c1"]
    assert (reply.status, reply.content) == ("error", CALL_ID_REUSED_REPLY)
    assert [(c.id, d) for c, d, _ in rig.reports] == [("c1", Deny(reasons=("call_id_reused",)))]


def test_a_re_run_whose_builder_drifted_is_denied_not_replayed() -> None:
    # c1 ran; on the re-run its builder gives another charge, so its stored record no longer
    # matches: neither replayed as that charge nor run again.
    drifted: list[bool] = []

    def drifting(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        closed = build_charge(args, ctx)
        return closed.model_copy(update={"reference": "R-2"}) if drifted else closed

    rig = Rig(builder=drifting).replaying()
    fail_once(rig)
    graph = rig.graph()
    charge = call("request_payment", "c1", payer="cust-juan", amount_minor=35000)
    with pytest.raises(RuntimeError, match="provider down"):
        graph.invoke(turn(charge, call("lookup_customer", "c2")), rig.thread)
    rig.lookup.fails_with = None
    drifted.append(True)

    again = graph.invoke(None, rig.thread)

    # The call ran, so the reply must not claim it did not.
    assert len(rig.charge.runs) == 1
    reply = replies(again)["c1"]
    assert (reply.status, reply.content) == ("error", CALL_ID_REUSED_REPLY)
    assert "Not executed" not in str(reply.content)


# Unknown tools, bad input and configuration.


@pytest.mark.parametrize(
    ("name", "drop_spec", "drop_tool"),
    [
        ("transfer_funds", None, None),
        ("delete_customer", "delete_customer", None),
        ("delete_customer", None, "delete_customer"),
    ],
)
def test_unknown_tool_is_denied_and_not_executed(
    name: str, drop_spec: str | None, drop_tool: str | None
) -> None:
    rig = Rig()
    graph = rig.graph(rig.node(drop_spec=drop_spec, drop_tool=drop_tool))
    result = graph.invoke(turn(call(name, "c1")), rig.thread)

    assert "__interrupt__" not in result
    assert rig.total_runs() == 0
    assert replies(result)["c1"].content == "Not executed: unknown_tool"
    assert [(c.id, d) for c, d, _ in rig.reports] == [("c1", Deny(reasons=("unknown_tool",)))]


def test_reads_only_the_last_message() -> None:
    # A re-entered node must not find the turn's calls again behind their replies.
    rig = Rig()
    answered: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[call("lookup_customer", "c1")]),
        ToolMessage(content="done", tool_call_id="c1"),
    ]

    with pytest.raises(ValueError, match="AIMessage"):
        rig.graph().invoke({"messages": answered}, rig.thread)
    assert rig.total_runs() == 0


def test_duplicate_call_ids_are_rejected() -> None:
    rig = Rig()
    twice = turn(call("delete_customer", "c1"), call("delete_customer", "c1"))

    with pytest.raises(ValueError, match="unique"):
        rig.graph().invoke(twice, rig.thread)
    assert rig.total_runs() == 0


def test_message_without_tool_calls_returns_no_messages() -> None:
    rig = Rig()
    result = rig.graph().invoke({"messages": [AIMessage(content="hi")]}, rig.thread)

    assert replies(result) == {}
    assert rig.contexts_built == 0


def test_rejects_ambiguous_configuration() -> None:
    tool = FakeTool("lookup_customer").as_tool()
    with pytest.raises(ValueError, match="unique"):
        GatedToolNode(tools=[tool, tool], specs={"lookup_customer": LOOKUP}, context=gate_ctx)
    with pytest.raises(ValueError, match="named"):
        GatedToolNode(tools=[tool], specs={"lookup": LOOKUP}, context=gate_ctx)


class NotedArgs(CustomerArgs):
    """A tool schema with an argument the spec never validates nor shows a human."""

    note: str = "added by the tool"


def test_tool_arguments_must_match_its_spec() -> None:
    lax = FakeTool("lookup_customer", args_schema=NotedArgs).as_tool()
    with pytest.raises(ValueError, match="arguments"):
        GatedToolNode(tools=[lax], specs={"lookup_customer": LOOKUP}, context=gate_ctx)


# Errors: none is swallowed.


def test_tool_error_propagates_after_its_call_is_reported() -> None:
    # An approved charge whose provider call failed still leaves an audit record.
    rig = Rig(lookup=FakeTool("lookup_customer", fails_with=RuntimeError("provider down")))

    with pytest.raises(RuntimeError, match="provider down"):
        rig.graph().invoke(
            turn(call("lookup_customer", "c1"), call("lookup_customer", "c2")), rig.thread
        )
    assert [(c.id, d.kind, a) for c, d, a in rig.reports] == [("c1", "allow", None)]
    assert rig.ran_at_report == [("c1", 0)]  # Reported before it ran; c2 never ran.


def test_tool_that_handles_its_errors_replies_with_an_error() -> None:
    failing = FakeTool(
        "lookup_customer", fails_with=ToolException("no such customer"), handles_errors=True
    )
    rig = Rig(lookup=failing)

    result = rig.graph().invoke(turn(call("lookup_customer", "c1")), rig.thread)

    reply = replies(result)["c1"]
    assert (reply.status, reply.content) == ("error", "no such customer")
    assert isinstance(rig.reports[0][1], Allow)


def test_on_decision_error_propagates() -> None:
    def broken_audit(call: ToolCall, decision: Decision, approval: ApprovalDecision | None) -> None:
        raise RuntimeError("audit log unavailable")

    rig = Rig()
    node = GatedToolNode(
        tools=[rig.lookup.as_tool()],
        specs={"lookup_customer": LOOKUP},
        context=gate_ctx,
        on_decision=broken_audit,
    )

    with pytest.raises(RuntimeError, match="audit log unavailable"):
        rig.graph(node).invoke(turn(call("lookup_customer", "c1")), rig.thread)


def test_audit_failure_blocks_an_allowed_call() -> None:
    # The decision is reported before the call runs: an audit log that cannot record it stops
    # the charge, instead of failing after the money moved.
    rig = Rig(audit_error=RuntimeError("audit log unavailable"))
    allowed = call("request_payment", "c1", payer="cust-juan", amount_minor=35000)

    with pytest.raises(RuntimeError, match="audit log unavailable"):
        rig.graph().invoke(turn(allowed), rig.thread)
    assert rig.charge.runs == []


def test_audit_failure_blocks_an_approved_call() -> None:
    rig = Rig(audit_error=RuntimeError("audit log unavailable"))
    graph = rig.graph()
    open_amount = call("request_payment", "c1", payer="cust-juan")
    (request,) = pending(graph.invoke(turn(open_amount), rig.thread))  # Reports nothing yet.

    with pytest.raises(RuntimeError, match="audit log unavailable"):
        graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)
    assert rig.charge.runs == []


def test_budget_exceeded_in_a_tool_callback_stops_the_run() -> None:
    rig = Rig()
    handler = BudgetCallbackHandler(BudgetLimits(max_tool_calls=0), None)

    with pytest.raises(BudgetExceededError):
        rig.graph().invoke(
            turn(call("lookup_customer", "c1")), rig.thread | {"callbacks": [handler]}
        )
    assert rig.lookup.runs == []


@pytest.mark.parametrize("handles_every_error", [False, True])
def test_stock_tool_node_swallows_budget_errors_when_it_handles_every_error(
    handles_every_error: bool,
) -> None:
    # Documents langgraph-prebuilt: its default handler re-raises everything but invalid
    # arguments; `handle_tool_errors=True` turns BudgetExceededError into an error reply.
    lookup = FakeTool("lookup_customer")
    tools = [lookup.as_tool()]
    node = ToolNode(tools, handle_tool_errors=True) if handles_every_error else ToolNode(tools)
    handler = BudgetCallbackHandler(BudgetLimits(max_tool_calls=0), None)
    graph = Rig().graph(node)
    config: RunnableConfig = {"callbacks": [handler], "configurable": {"thread_id": "t"}}

    if handles_every_error:
        result = graph.invoke(turn(call("lookup_customer", "c1")), config)
        assert replies(result)["c1"].status == "error"
    else:
        with pytest.raises(BudgetExceededError):
            graph.invoke(turn(call("lookup_customer", "c1")), config)
    assert lookup.runs == []


def test_approval_round_trip_under_ainvoke() -> None:
    rig = Rig()
    graph = rig.graph()

    async def run() -> dict[str, Any]:
        first = await graph.ainvoke(turn(call("delete_customer", "c1")), rig.thread)
        (request,) = pending(first)
        return await graph.ainvoke(Command(resume={"c1": answer(request)}), rig.thread)

    resumed = asyncio.run(run())

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]
    assert replies(resumed)["c1"].status == "success"

"""`GatedToolNode`: a LangGraph tool node that runs only the tool calls the policy gate allows."""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from pydantic import BaseModel, ConfigDict

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
    ApprovalDecision,
    ApprovalRequest,
    GatedToolNode,
    closed_mandate_from_config,
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


Builder = Callable[[BaseModel, str], ClosedMandate]


def new_charge(args: BaseModel, charge_ref: str) -> ClosedMandate:
    """A new charge under `charge_ref`, or an increase of the link the arguments name."""
    assert isinstance(args, ChargeArgs)
    previous = args.previous_amount_minor
    return ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge" if previous is None else "amount_increase",
        charge_ref=args.link_ref or charge_ref,
        payer=args.payer,
        receiving_account="acct-own",
        rail="codi_push",
        amount=None if args.amount_minor is None else mxn(args.amount_minor),
        previous_amount=None if previous is None else mxn(previous),
        reference="R-1",
    )


def build_charge(args: BaseModel, call_id: str) -> ClosedMandate:
    """The application's builder: it mints the ref from the call, so a re-run rebuilds it."""
    return new_charge(args, f"ch_{call_id}")


def build_with_random_ref(args: BaseModel, call_id: str) -> ClosedMandate:
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
    """A tool that records the arguments and the gate's closed mandate of every run."""

    name: str
    args_schema: type[BaseModel] = CustomerArgs
    runs: list[dict[str, object]] = field(default_factory=list)
    closed_seen: list[ClosedMandate | None] = field(default_factory=list)
    fails_with: Exception | None = None
    handles_errors: bool = False

    def as_tool(self) -> BaseTool:
        def run(config: RunnableConfig, **kwargs: object) -> str:
            self.runs.append(kwargs)
            self.closed_seen.append(closed_mandate_from_config(config))
            if self.fails_with is not None:
                raise self.fails_with
            return f"{self.name} ran"

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
        )

    def build_context(self, state: MessagesState) -> GateContext:
        self.contexts_built += 1
        return self.context

    def record(self, call: ToolCall, decision: Decision, approval: ApprovalDecision | None) -> None:
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


def turn(*calls: LcToolCall) -> dict[str, list[AnyMessage]]:
    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


def pending(result: dict[str, Any]) -> list[ApprovalRequest]:
    (raised,) = result["__interrupt__"]
    assert isinstance(raised, Interrupt)
    requests = raised.value
    assert isinstance(requests, list)
    assert all(isinstance(request, ApprovalRequest) for request in requests)
    return requests


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
    assert len(request.request_hash) == 64
    assert rig.delete.runs == []

    resumed = graph.invoke(Command(resume={"c1": answer(request)}), rig.thread)

    assert rig.delete.runs == [{"customer_id": "cust-c1"}]
    assert replies(resumed)["c1"].status == "success"


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
    # Calls that do not run are reported before any tool runs; the others after they ran.
    assert rig.ran_at_report == [("c3", 0), ("c4", 0), ("c1", 1), ("c2", 2)]


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
    checked = build_charge(ChargeArgs(payer="cust-juan"), "c1")
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


def test_tools_receive_the_checked_closed_mandate_via_config() -> None:
    # Whatever the caller put under the key, each tool gets the charge checked for its call.
    rig = Rig()
    graph = rig.graph()
    forged = build_charge(ChargeArgs(payer="cust-juan", amount_minor=500000), "forged")
    config = RunnableConfig(configurable={"thread_id": "t1", CLOSED_MANDATE_KEY: forged})
    calls = turn(
        call("request_payment", "c1", payer="cust-juan"),  # Open amount: needs approval.
        call("request_payment", "c2", payer="cust-juan", amount_minor=35000),  # Allowed.
        call("lookup_customer", "c3"),
    )
    (request,) = pending(graph.invoke(calls, config))

    graph.invoke(Command(resume={"c1": answer(request)}), config)

    checked = {c.id: d.closed for c, d, _ in rig.reports if not isinstance(d, Deny)}
    assert checked["c1"] == build_charge(ChargeArgs(payer="cust-juan"), "c1")
    assert checked["c1"].content_hash == request.closed_mandate_hash
    assert checked["c2"] == build_charge(ChargeArgs(payer="cust-juan", amount_minor=35000), "c2")
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


def test_two_increases_of_one_link_in_one_turn() -> None:
    # The first increase reserves its delta, so the second no longer starts from 30000.
    link = ConsumptionEntry(charge_ref="ch_link", kind="reserve", amount=mxn(30000), at=NOW)
    rig = Rig(context=gate_context(ledger=(link,)))

    def increase(call_id: str, amount_minor: int) -> LcToolCall:
        return call(
            "request_payment",
            call_id,
            payer="cust-juan",
            amount_minor=amount_minor,
            previous_amount_minor=30000,
            link_ref="ch_link",
        )

    result = rig.graph().invoke(turn(increase("c1", 45000), increase("c2", 50000)), rig.thread)

    assert "__interrupt__" not in result
    assert [run["amount_minor"] for run in rig.charge.runs] == [45000]
    assert replies(result)["c2"].content == "Not executed: mandate:previous_amount_mismatch"


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

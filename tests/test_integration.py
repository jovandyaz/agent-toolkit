"""The packages together: a model proposes, the gate decides, a tool charges and reserves, the
audit log records each decision. In-memory stores, a fake model and a fake payment provider.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, ConfigDict

from agent_toolkit import mandates
from agent_toolkit.audit_log import AuditEvent, AuditStore, append, hash_payload, verify_chain
from agent_toolkit.audit_log.testing import InMemoryAuditStore
from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    ConsumptionEntry,
    OpenMandate,
)
from agent_toolkit.policy_gate import (
    BuildContext,
    Decision,
    Effects,
    GateConfig,
    GateContext,
    SessionProfile,
    ToolCall,
    ToolSpec,
)
from agent_toolkit.policy_gate.adapters.langgraph import (
    ApprovalDecision,
    GatedToolNode,
    RecordedReply,
    call_fingerprint_from_config,
    closed_mandate_from_config,
    idempotency_key_from_config,
)

NOW = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)
THREAD = RunnableConfig(configurable={"thread_id": "thread-1"})
AUDIT_KEY = b"integration-test-payload-key-32b"  # Test only; a real key lives in a secrets store.


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


class ChargeArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    payer: str
    amount_minor: int


class CustomerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str


def build_charge(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    assert isinstance(args, ChargeArgs)
    return ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge",
        charge_ref=f"ch_{ctx.thread_id}_{ctx.call_id}",
        payer=args.payer,
        receiving_account="acct-own",
        rail="codi_push",
        amount=mxn(args.amount_minor),
        previous_amount=None,
        reference="R-1",
    )


SPECS = {
    "request_payment": ToolSpec(
        name="request_payment",
        risk="high",
        effects=Effects(reads_sensitive=False, changes_state=True, external_comm=True),
        reversible=False,
        financial=True,
        args_model=ChargeArgs,
        to_closed_mandate=build_charge,
    ),
    "lookup_customer": ToolSpec(
        name="lookup_customer",
        risk="low",
        effects=Effects(reads_sensitive=True, changes_state=False, external_comm=False),
        reversible=True,
        financial=False,
        args_model=CustomerArgs,
        to_closed_mandate=None,
    ),
}
MANDATE = OpenMandate(
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


class UnavailableAuditStore(InMemoryAuditStore):
    def append(self, event: AuditEvent) -> None:
        raise ConnectionError("audit database unavailable")


@dataclass
class App:
    """The application around the gate: its stores, a payment provider and its tools."""

    audit: AuditStore = field(default_factory=InMemoryAuditStore)
    ledger: list[ConsumptionEntry] = field(default_factory=list)
    replies: dict[str, RecordedReply] = field(default_factory=dict)
    issued: dict[str, ClosedMandate] = field(default_factory=dict)  # The provider, by key.
    provider_requests: int = 0
    crm_down: bool = False

    def request_payment(self, payer: str, amount_minor: int, config: RunnableConfig) -> str:
        closed = closed_mandate_from_config(config)
        assert closed is not None
        key = idempotency_key_from_config(config)
        self.provider_requests += 1
        self.issued.setdefault(key, closed)  # The provider dedupes on the idempotency key.
        reserved = mandates.reserve_amount(closed)
        assert reserved is not None
        # One transaction in a real app: the reserve and the reply the replay hook returns.
        self.ledger.append(
            ConsumptionEntry(charge_ref=closed.charge_ref, kind="reserve", amount=reserved, at=NOW)
        )
        return self.store(config, f"payment request {closed.charge_ref} sent")

    def lookup_customer(self, customer_id: str, config: RunnableConfig) -> str:
        if self.crm_down:
            raise RuntimeError("CRM unavailable")
        return self.store(config, f"{customer_id} found")

    def store(self, config: RunnableConfig, reply: str) -> str:
        fingerprint = call_fingerprint_from_config(config)
        self.replies[idempotency_key_from_config(config)] = RecordedReply(
            fingerprint=fingerprint, content=reply
        )
        return reply

    def context(self, state: MessagesState) -> GateContext:
        return GateContext(
            session=SessionProfile(
                agent_id="agent_1",
                principal_id="owner_1",
                untrusted_input=True,
                sensitive_access=True,
                state_change=True,
            ),
            mandate=MANDATE,
            ledger=tuple(self.ledger),
            budget_state=BudgetState.empty(),
            budget_limits=BudgetLimits(max_tool_calls=10),
            now=NOW,
            config=GateConfig(),
        )

    def audit_decision(
        self, call: ToolCall, decision: Decision, approval: ApprovalDecision | None
    ) -> None:
        append(
            self.audit,
            actor_id="agent_1",
            actor_kind="agent",
            action=f"gate:{call.name}",
            subject=call.id,
            payload_hash=hash_payload(decision.model_dump_json().encode(), AUDIT_KEY),
            decision=decision.kind,
            at=NOW,
        )

    def tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                func=self.request_payment,
                name="request_payment",
                description="Send a payment request.",
                args_schema=ChargeArgs,
            ),
            StructuredTool.from_function(
                func=self.lookup_customer,
                name="lookup_customer",
                description="Find a customer.",
                args_schema=CustomerArgs,
            ),
        ]

    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            name="request_payment",
                            args={"payer": "cust-juan", "amount_minor": 35000},
                            id="c1",
                        ),
                        tool_call(
                            name="lookup_customer", args={"customer_id": "cust-juan"}, id="c2"
                        ),
                    ],
                ),
                AIMessage(content="Payment request sent to Juan."),
            ]
        )
        node = GatedToolNode(
            self.tools(),
            SPECS,
            self.context,
            on_decision=self.audit_decision,
            replay=self.replies.get,
        )

        def call_model(state: MessagesState) -> dict[str, list[AIMessage]]:
            reply = model.invoke(state["messages"])
            assert isinstance(reply, AIMessage)
            return {"messages": [reply]}

        builder = StateGraph(MessagesState)
        builder.add_node("model", call_model)
        builder.add_node("tools", node)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", tools_condition)
        builder.add_edge("tools", "model")
        return builder.compile(checkpointer=InMemorySaver())


def ask() -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content="Charge Juan 350 pesos")]}


def test_a_re_run_after_a_partial_failure_replays_the_charge_instead_of_repeating_it() -> None:
    app = App(crm_down=True)
    graph = app.graph()
    with pytest.raises(RuntimeError, match="CRM unavailable"):
        graph.invoke(ask(), THREAD)
    assert (app.provider_requests, len(app.ledger)) == (1, 1)
    app.crm_down = False

    final = graph.invoke(None, THREAD)

    assert (app.provider_requests, len(app.issued), len(app.ledger)) == (1, 1, 1)
    replies = {m.tool_call_id: m for m in final["messages"] if isinstance(m, ToolMessage)}
    assert replies["c1"].status == "success"
    assert replies["c1"].content == "payment request ch_thread-1_c1 sent"
    assert replies["c2"].content == "cust-juan found"
    assert final["messages"][-1].content == "Payment request sent to Juan."
    events = list(app.audit.iter_all())
    assert verify_chain(events).ok
    # c1 was recorded before it ran and is not recorded again; c2 was attempted twice.
    assert [(e.subject, e.decision) for e in events] == [
        ("c1", "allow"),
        ("c2", "allow"),
        ("c2", "allow"),
    ]


def test_an_audit_failure_blocks_the_charge() -> None:
    app = App(audit=UnavailableAuditStore())

    with pytest.raises(ConnectionError, match="audit database unavailable"):
        app.graph().invoke(ask(), THREAD)

    assert (app.provider_requests, app.issued, app.ledger) == (0, {}, [])

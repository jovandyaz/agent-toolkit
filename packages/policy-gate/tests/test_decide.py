"""`decide`: the deterministic gate between a tool call the model proposes and its execution."""

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ConfigDict

from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    ConsumptionEntry,
    DailyWindow,
    OpenMandate,
)
from agent_toolkit.policy_gate import (
    Allow,
    BuildContext,
    Deny,
    Effects,
    GateConfig,
    GateContext,
    RequireApproval,
    SessionProfile,
    ToolCall,
    ToolSpec,
    decide,
)

MEXICO_CITY = ZoneInfo("America/Mexico_City")
NOW = datetime(2026, 12, 1, 12, 0, tzinfo=MEXICO_CITY)
CALL_ID = "call_1"
THREAD_ID = "thread_1"
MESSAGE_ID = "msg_1"
BUILD = BuildContext(thread_id=THREAD_ID, message_id=MESSAGE_ID, call_id=CALL_ID)


def mint_ref(ctx: BuildContext) -> str:
    """The application mints a new charge's ref server-side from the thread, message and call.

    All three: a call id alone can repeat in another message or another thread.
    """
    canonical = json.dumps([ctx.thread_id, ctx.message_id, ctx.call_id], separators=(",", ":"))
    return f"ch_{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"


MINTED_REF = mint_ref(BUILD)


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


def reserve(ref: str, amount: Amount) -> ConsumptionEntry:
    return ConsumptionEntry(
        charge_ref=ref, kind="reserve", amount=amount, at=NOW - timedelta(hours=2)
    )


def release(ref: str, amount: Amount) -> ConsumptionEntry:
    return ConsumptionEntry(
        charge_ref=ref, kind="release", amount=amount, at=NOW - timedelta(hours=1)
    )


class ChargeArgs(BaseModel):
    """What the model may say about a charge: nothing about mandates, agents or the ledger."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    payer: str
    receiving_account: str
    rail: str
    amount_minor: int | None
    currency: str
    reference: str
    previous_amount_minor: int | None = None
    link_ref: str | None = None  # The pending link an increase raises.


class LeakyChargeArgs(ChargeArgs):
    """A careless application that lets the model name the acting agent."""

    agent_id: str


class CustomerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str


class ScoredArgs(BaseModel):
    """Takes a mapping whose keys the model chooses."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str
    scores: dict[str, int] | None = None


class Note(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str
    weight: int = 0


class IgnoringArgs(BaseModel):
    """Lax on purpose: ignores unknown keys and coerces types unless the gate says otherwise."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    customer_id: str
    priority: int = 0
    note: Note | None = None


class AllowingArgs(BaseModel):
    """Lax on purpose: keeps unknown keys in `model_extra`."""

    model_config = ConfigDict(frozen=True, extra="allow")

    customer_id: str


def build_charge(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    """The application's builder: it binds the mandate and agent, and mints a new charge's ref.

    An increase names the link it raises; the gate reads that ref's reserve from the ledger.
    """
    assert isinstance(args, ChargeArgs)
    previous = args.previous_amount_minor
    if (previous is None) != (args.link_ref is None):
        raise ValueError("an increase names its link and its previous amount; a new charge neither")
    return ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge" if previous is None else "amount_increase",
        charge_ref=args.link_ref or mint_ref(ctx),
        payer=args.payer,
        receiving_account=args.receiving_account,
        rail=args.rail,
        amount=(
            None
            if args.amount_minor is None
            else Amount(minor=args.amount_minor, currency=args.currency)
        ),
        previous_amount=(
            None if previous is None else Amount(minor=previous, currency=args.currency)
        ),
        reference=args.reference,
    )


def build_leaky_charge(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    """Copies the model's `agent_id` into the charge: exactly what the gate must not trust."""
    assert isinstance(args, LeakyChargeArgs)
    return ClosedMandate.model_validate(
        build_charge(args, ctx).model_dump() | {"agent_id": args.agent_id}
    )


SPENDS = Effects(reads_sensitive=False, changes_state=True, external_comm=True)
READS = Effects(reads_sensitive=True, changes_state=False, external_comm=False)
WRITES = Effects(reads_sensitive=False, changes_state=True, external_comm=False)
NOTIFIES = Effects(reads_sensitive=False, changes_state=False, external_comm=True)


def financial_tool(
    args_model: type[BaseModel] = ChargeArgs,
    builder: Callable[[BaseModel, BuildContext], ClosedMandate] = build_charge,
) -> ToolSpec:
    return ToolSpec(
        name="request_payment",
        risk="high",
        effects=SPENDS,
        reversible=False,
        financial=True,
        args_model=args_model,
        to_closed_mandate=builder,
    )


def plain_tool(
    name: str,
    risk: str,
    effects: Effects,
    args_model: type[BaseModel] = CustomerArgs,
) -> ToolSpec:
    return ToolSpec.model_validate(
        {
            "name": name,
            "risk": risk,
            "effects": effects,
            "reversible": not effects.changes_state,
            "financial": False,
            "args_model": args_model,
            "to_closed_mandate": None,
        }
    )


CHARGE = financial_tool()
LOOKUP = plain_tool("lookup_customer", "low", READS)
TAG = plain_tool("tag_customer", "medium", WRITES)
DELETE = plain_tool("delete_customer", "high", WRITES)
EMAIL = plain_tool("email_customer", "low", NOTIFIES)

CHARGE_ARGS: dict[str, object] = {
    "payer": "cust-juan",
    "receiving_account": "acct-own",
    "rail": "codi_push",
    "amount_minor": 35000,
    "currency": "MXN",
    "reference": "R-1001",
}
CUSTOMER_ARGS: dict[str, object] = {"customer_id": "cust-juan"}


def charge_call(**overrides: object) -> ToolCall:
    return ToolCall(id=CALL_ID, name="request_payment", args=CHARGE_ARGS | overrides)


def customer_call(name: str, **overrides: object) -> ToolCall:
    return ToolCall(id="call_2", name=name, args=CUSTOMER_ARGS | overrides)


def charge_args(**overrides: object) -> ChargeArgs:
    return ChargeArgs.model_validate(CHARGE_ARGS | overrides)


def customer_args() -> CustomerArgs:
    return CustomerArgs.model_validate(CUSTOMER_ARGS)


def mandate(**overrides: object) -> OpenMandate:
    fields: dict[str, object] = {
        "id": "om_1",
        "principal_id": "owner_1",
        "agent_id": "agent_1",
        "purpose": "Collect payments from customers of the shop",
        "amount_range": AmountRange(min=mxn(100), max=mxn(500000)),
        "budget": Budget(limit=mxn(1000000), period="day"),
        "receiving_accounts": frozenset({"acct-own"}),
        "allowed_payers": frozenset({"cust-juan"}),
        "allowed_rails": frozenset({"codi_push", "codi_link"}),
        "daily_window": None,
        "tz": "America/Mexico_City",
        "expires_at": NOW + timedelta(days=30),
        "revoked_at": None,
    }
    return OpenMandate.model_validate(fields | overrides)


def session(**overrides: object) -> SessionProfile:
    fields: dict[str, object] = {
        "agent_id": "agent_1",
        "principal_id": "owner_1",
        "untrusted_input": False,
        "sensitive_access": False,
        "state_change": False,
    }
    return SessionProfile.model_validate(fields | overrides)


RULE_OF_TWO = {"untrusted_input": True, "sensitive_access": True, "state_change": True}


def context(**overrides: object) -> GateContext:
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


def without_mandate(**overrides: object) -> GateContext:
    return context(mandate=None, ledger=None, **overrides)


def config(on_mandate_violation: str) -> GateConfig:
    return GateConfig.model_validate({"on_mandate_violation": on_mandate_violation})


# Rule 7 and the plain allow path.


def test_low_risk_read_is_allowed_without_mandate() -> None:
    decision = decide(
        customer_call("lookup_customer"),
        LOOKUP,
        without_mandate(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Allow(args=customer_args(), closed=None)


def test_allow_returns_parsed_args_and_closed_mandate() -> None:
    # A high-risk financial tool inside its mandate needs no approval: the mandate governs it.
    decision = decide(charge_call(), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID)

    assert isinstance(decision, Allow)
    assert isinstance(decision.args, ChargeArgs)
    assert decision.args == charge_args()
    assert decision.closed == ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge",
        charge_ref=MINTED_REF,
        payer="cust-juan",
        receiving_account="acct-own",
        rail="codi_push",
        amount=mxn(35000),
        previous_amount=None,
        reference="R-1001",
    )


def test_increase_inside_its_mandate_is_allowed() -> None:
    decision = decide(
        charge_call(link_ref="ch_1", previous_amount_minor=30000),
        CHARGE,
        context(ledger=(reserve("ch_1", mxn(30000)),)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert isinstance(decision, Allow)
    assert decision.closed is not None
    assert decision.closed.kind == "amount_increase"


def test_increase_is_judged_against_the_reserve_of_its_own_ref() -> None:
    # The ledger holds two links; the call raises ch_1, whose reserve is 30000.
    ledger = context(ledger=(reserve("ch_2", mxn(40000)), reserve("ch_1", mxn(30000))))

    own = decide(
        charge_call(amount_minor=45000, link_ref="ch_1", previous_amount_minor=30000),
        CHARGE,
        ledger,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    other = decide(
        charge_call(amount_minor=45000, link_ref="ch_1", previous_amount_minor=40000),
        CHARGE,
        ledger,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    fresh = decide(charge_call(), CHARGE, ledger, thread_id=THREAD_ID, message_id=MESSAGE_ID)

    assert isinstance(own, Allow)
    assert own.closed is not None
    assert own.closed.charge_ref == "ch_1"
    assert other == Deny(reasons=("mandate:previous_amount_mismatch",))
    assert isinstance(fresh, Allow)


# Rule 1: the call must be for this tool.


def test_tool_name_mismatch_is_denied() -> None:
    decision = decide(
        customer_call("delete_customer"),
        LOOKUP,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("tool_mismatch",))


# Rule 2: model output is untrusted, so arguments are validated strictly.


def test_args_are_validated_strictly() -> None:
    decision = decide(
        charge_call(amount_minor="35000"),
        CHARGE,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args.amount_minor:int_type",))


def test_every_invalid_arg_is_a_reason() -> None:
    args = {key: value for key, value in CHARGE_ARGS.items() if key != "payer"}

    decision = decide(
        ToolCall(id="call_1", name="request_payment", args=args | {"rail": 7}),
        CHARGE,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(
        reasons=("invalid_args:args.payer:missing", "invalid_args:args.rail:string_type")
    )


def test_unknown_arg_key_is_denied() -> None:
    decision = decide(
        charge_call(agent_id="agent_2"),
        CHARGE,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args:extra_forbidden",))


HOSTILE_KEYS = [
    pytest.param("evil\nkey", "evil_key", id="newline"),
    pytest.param("a:b", "a_b", id="colon"),
    pytest.param("k" * 1000, "k" * 64, id="long"),
]


@pytest.mark.parametrize(("key", "segment"), HOSTILE_KEYS)
def test_unknown_key_stays_out_of_the_reason(key: str, segment: str) -> None:
    # Reasons reach the model and the audit log; an unknown key is model-chosen text.
    decision = decide(
        customer_call("tag_customer", **{key: 1}),
        TAG,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args:extra_forbidden",))


@pytest.mark.parametrize(("key", "segment"), HOSTILE_KEYS)
def test_model_chosen_keys_in_a_location_are_sanitized(key: str, segment: str) -> None:
    tool = plain_tool("score_customer", "low", WRITES, ScoredArgs)

    decision = decide(
        customer_call("score_customer", scores={key: "high"}),
        tool,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=(f"invalid_args:args.scores.{segment}:int_type",))


def test_several_unknown_keys_give_one_reason() -> None:
    decision = decide(
        customer_call("tag_customer", a=1, b=2),
        TAG,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args:extra_forbidden",))


@pytest.mark.parametrize("args_model", [IgnoringArgs, AllowingArgs])
def test_unknown_arg_key_is_denied_even_if_the_model_accepts_it(
    args_model: type[BaseModel],
) -> None:
    tool = plain_tool("tag_customer", "low", WRITES, args_model)

    decision = decide(
        customer_call("tag_customer", agent_id="agent_2"),
        tool,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args:extra_forbidden",))


def test_lax_models_are_validated_strictly_at_every_depth() -> None:
    tool = plain_tool("tag_customer", "low", WRITES, IgnoringArgs)

    coerced = decide(
        customer_call("tag_customer", priority="1"),
        tool,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    nested_coerced = decide(
        customer_call("tag_customer", note={"text": "vip", "weight": "5"}),
        tool,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    nested_unknown = decide(
        customer_call("tag_customer", note={"text": "vip", "payer": "x"}),
        tool,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert coerced == Deny(reasons=("invalid_args:args.priority:int_type",))
    assert nested_coerced == Deny(reasons=("invalid_args:args.note.weight:int_type",))
    assert nested_unknown == Deny(reasons=("invalid_args:args.note:extra_forbidden",))


def test_args_are_checked_before_the_budget() -> None:
    spent = BudgetState.empty().model_copy(update={"tool_calls": 5})

    decision = decide(
        charge_call(amount_minor="1"),
        CHARGE,
        context(budget_state=spent, budget_limits=BudgetLimits(max_tool_calls=1)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:args.amount_minor:int_type",))


# Rule 3: an exhausted run budget stops every call.


def test_exceeded_budget_is_denied_with_each_kind() -> None:
    state = BudgetState(
        iterations=0,
        tool_calls=3,
        input_tokens=8,
        output_tokens=8,
        cost_usd=Decimal(0),
        consecutive_failures=0,
    )
    limits = BudgetLimits(max_tool_calls=2, max_tokens=10)

    decision = decide(
        customer_call("lookup_customer"),
        LOOKUP,
        context(budget_state=state, budget_limits=limits),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("budget_exceeded:tool_calls", "budget_exceeded:tokens"))


def test_budget_is_checked_before_the_mandate() -> None:
    spent = BudgetState.empty().model_copy(update={"consecutive_failures": 4})

    decision = decide(
        charge_call(receiving_account="acct-other"),
        CHARGE,
        context(budget_state=spent, budget_limits=BudgetLimits(max_consecutive_failures=3)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("budget_exceeded:consecutive_failures",))


# Rule 4: financial calls are judged against the mandate.


def test_financial_call_without_mandate_requires_approval() -> None:
    decision = decide(
        charge_call(), CHARGE, without_mandate(), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )

    assert decision == RequireApproval(reasons=("no_mandate",), args=charge_args(), closed=None)


def test_foreign_account_is_denied_even_if_config_says_approval() -> None:
    # Rule 7.3(c): charges go only to the credential holder's accounts; no approval overrides it.
    decision = decide(
        charge_call(receiving_account="acct-other"),
        CHARGE,
        context(config=config("require_approval")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("mandate:account_not_allowed",))


ALWAYS_DENIED = [
    pytest.param({"receiving_account": "acct-other"}, {}, "account_not_allowed", id="account"),
    pytest.param({}, {"mandate": mandate(id="om_2")}, "mandate_mismatch", id="mandate"),
    pytest.param({}, {"mandate": mandate(agent_id="agent_2")}, "agent_mismatch", id="agent"),
    pytest.param({"currency": "USD"}, {}, "currency_mismatch", id="currency"),
    pytest.param({}, {"mandate": mandate(expires_at=NOW)}, "expired", id="expired"),
    pytest.param({}, {"mandate": mandate(revoked_at=NOW)}, "revoked", id="revoked"),
    pytest.param(
        {"link_ref": "ch_1", "previous_amount_minor": 30000},
        {},
        "previous_amount_mismatch",
        id="previous_amount",
    ),
    pytest.param(
        {},
        {"ledger": (reserve(MINTED_REF, mxn(100)),)},
        "charge_ref_reused",
        id="ref_reserved",
    ),
    pytest.param(
        {},
        {"ledger": (reserve(MINTED_REF, mxn(100)), release(MINTED_REF, mxn(100)))},
        "charge_ref_reused",
        id="ref_released",
    ),
]


@pytest.mark.parametrize("on_violation", ["deny", "require_approval"])
@pytest.mark.parametrize(("args", "ctx", "violation"), ALWAYS_DENIED)
def test_binding_violations_are_always_denied(
    args: dict[str, object], ctx: dict[str, object], violation: str, on_violation: str
) -> None:
    decision = decide(
        charge_call(**args),
        CHARGE,
        context(config=config(on_violation), **ctx),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=(f"mandate:{violation}",))


def test_violation_follows_config() -> None:
    out_of_range = charge_call(amount_minor=600000)

    approve = decide(
        out_of_range,
        CHARGE,
        context(config=config("require_approval")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    deny = decide(
        out_of_range,
        CHARGE,
        context(config=config("deny")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert approve == RequireApproval(
        reasons=("mandate:amount_out_of_range",),
        args=charge_args(amount_minor=600000),
        closed=build_charge(charge_args(amount_minor=600000), BUILD),
    )
    assert deny == Deny(reasons=("mandate:amount_out_of_range",))


CONFIGURABLE = [
    pytest.param({"rail": "spei"}, {}, "rail_not_allowed", id="rail"),
    pytest.param({"amount_minor": 600000}, {}, "amount_out_of_range", id="range"),
    pytest.param({}, {"ledger": (reserve("ch_0", mxn(990000)),)}, "budget_exceeded", id="budget"),
    pytest.param(
        {},
        {"mandate": mandate(daily_window=DailyWindow(start=time(7), end=time(11)))},
        "outside_window",
        id="window",
    ),
]


@pytest.mark.parametrize(("args", "ctx", "violation"), CONFIGURABLE)
def test_configurable_violations_follow_config(
    args: dict[str, object], ctx: dict[str, object], violation: str
) -> None:
    call = charge_call(**args)

    approve = decide(
        call,
        CHARGE,
        context(config=config("require_approval"), **ctx),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    deny = decide(
        call,
        CHARGE,
        context(config=config("deny"), **ctx),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert isinstance(approve, RequireApproval)
    assert approve.reasons == (f"mandate:{violation}",)
    assert deny == Deny(reasons=(f"mandate:{violation}",))


def test_open_amount_qr_requires_approval() -> None:
    decision = decide(
        charge_call(amount_minor=None),
        CHARGE,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == RequireApproval(
        reasons=("mandate:open_amount",),
        args=charge_args(amount_minor=None),
        closed=build_charge(charge_args(amount_minor=None), BUILD),
    )


def test_unknown_payer_requires_approval() -> None:
    decision = decide(
        charge_call(payer="cust-ana"), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )

    assert isinstance(decision, RequireApproval)
    assert decision.reasons == ("mandate:payer_not_allowed",)


def test_deny_takes_precedence_over_approval() -> None:
    decision = decide(
        charge_call(receiving_account="acct-other", payer="cust-ana", amount_minor=None),
        CHARGE,
        context(session=session(**RULE_OF_TWO)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("mandate:account_not_allowed",))


def test_deny_lists_every_denying_violation() -> None:
    call = charge_call(receiving_account="acct-other", rail="spei")

    strict = decide(
        call, CHARGE, context(config=config("deny")), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )
    lenient = decide(
        call,
        CHARGE,
        context(config=config("require_approval")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert strict == Deny(reasons=("mandate:account_not_allowed", "mandate:rail_not_allowed"))
    assert lenient == Deny(reasons=("mandate:account_not_allowed",))


def test_financial_call_under_mandate_needs_its_ledger() -> None:
    # Treating a missing ledger as empty would overspend: it is a programming error.
    with pytest.raises(ValueError, match="needs the mandate's ledger"):
        decide(
            charge_call(), CHARGE, context(ledger=None), thread_id=THREAD_ID, message_id=MESSAGE_ID
        )


def test_ledger_is_needed_only_for_financial_calls_under_a_mandate() -> None:
    no_ledger = context(ledger=None)

    assert decide(
        customer_call("lookup_customer"),
        LOOKUP,
        no_ledger,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    ) == Allow(args=customer_args(), closed=None)
    assert isinstance(
        decide(
            charge_call(), CHARGE, without_mandate(), thread_id=THREAD_ID, message_id=MESSAGE_ID
        ),
        RequireApproval,
    )


INCONSISTENT_LEDGERS = [
    pytest.param((release("ch_0", mxn(100)),), id="release_without_reserve"),
    pytest.param(
        (reserve("ch_0", mxn(100)), release("ch_0", mxn(100)), release("ch_0", mxn(100))),
        id="released_twice",
    ),
    pytest.param((reserve("ch_0", mxn(100)), release("ch_0", mxn(60))), id="partial_release"),
    pytest.param((reserve("ch_0", Amount(minor=100, currency="USD")),), id="other_currency"),
]


@pytest.mark.parametrize("on_violation", ["deny", "require_approval"])
@pytest.mark.parametrize("ledger", INCONSISTENT_LEDGERS)
def test_inconsistent_ledger_is_denied(
    ledger: tuple[ConsumptionEntry, ...], on_violation: str
) -> None:
    decision = decide(
        charge_call(),
        CHARGE,
        context(ledger=ledger, config=config(on_violation)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("ledger_inconsistent",))


@pytest.mark.parametrize("on_violation", ["deny", "require_approval"])
def test_mandate_of_another_principal_is_denied(on_violation: str) -> None:
    ctx = context(session=session(principal_id="owner_2"), config=config(on_violation))

    in_mandate = decide(charge_call(), CHARGE, ctx, thread_id=THREAD_ID, message_id=MESSAGE_ID)
    # Judged before mandates.check, so the foreign account is not what decides.
    foreign_account = decide(
        charge_call(receiving_account="acct-other"),
        CHARGE,
        ctx,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert in_mandate == Deny(reasons=("principal_mismatch",))
    assert foreign_account == Deny(reasons=("principal_mismatch",))


def test_tool_args_cannot_change_the_acting_agent() -> None:
    # The mandate belongs to agent_2. The session runs agent_1, and the model claims agent_2.
    leaky = financial_tool(LeakyChargeArgs, build_leaky_charge)
    call = charge_call(agent_id="agent_2")

    impersonating = decide(
        call,
        leaky,
        context(mandate=mandate(agent_id="agent_2")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    own_agent = decide(
        call,
        leaky,
        context(session=session(agent_id="agent_2"), mandate=mandate(agent_id="agent_2")),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert impersonating == Deny(reasons=("agent_mismatch",))
    assert isinstance(own_agent, Allow)


def test_agent_mismatch_is_reported_before_principal_mismatch() -> None:
    leaky = financial_tool(LeakyChargeArgs, build_leaky_charge)
    other_principal = context(session=session(principal_id="owner_2"))

    decision = decide(
        charge_call(agent_id="agent_2"),
        leaky,
        other_principal,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("agent_mismatch",))


def test_closed_mandate_that_does_not_validate_is_denied() -> None:
    # Amount allows 0 and rejects negatives; a charge needs a positive amount.
    zero = decide(
        charge_call(amount_minor=0), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )
    negative = decide(
        charge_call(amount_minor=-5), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )

    assert zero == Deny(reasons=("invalid_args:closed:value_error",))
    assert negative == Deny(reasons=("invalid_args:closed.minor:greater_than_equal",))


def test_builder_gets_the_parsed_args_and_the_thread_and_call_ids() -> None:
    # Both ids let the application mint a ref that a re-run of the gate rebuilds the same.
    seen: list[tuple[BaseModel, BuildContext]] = []

    def recording(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        seen.append((args, ctx))
        return build_charge(args, ctx)

    decision = decide(
        charge_call(),
        financial_tool(builder=recording),
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert seen == [(charge_args(), BUILD)]
    assert isinstance(decision, Allow)
    assert decision.closed is not None
    assert decision.closed.charge_ref == MINTED_REF


def test_one_call_id_in_two_threads_builds_two_charges() -> None:
    here = decide(charge_call(), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID)
    there = decide(charge_call(), CHARGE, context(), thread_id="thread_2", message_id=MESSAGE_ID)

    assert isinstance(here, Allow) and isinstance(there, Allow)
    assert here.closed is not None and there.closed is not None
    assert here.closed.charge_ref != there.closed.charge_ref


def test_one_call_id_in_two_messages_builds_two_charges() -> None:
    first = decide(charge_call(), CHARGE, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID)
    later = decide(charge_call(), CHARGE, context(), thread_id=THREAD_ID, message_id="msg_2")

    assert isinstance(first, Allow) and isinstance(later, Allow)
    assert first.closed is not None and later.closed is not None
    assert first.closed.charge_ref != later.closed.charge_ref


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param({}, id="neither"),
        pytest.param({"message_id": MESSAGE_ID}, id="no_thread"),
        pytest.param({"thread_id": THREAD_ID}, id="no_message"),
    ],
)
def test_financial_call_under_mandate_needs_the_thread_and_message_ids(ids: dict[str, str]) -> None:
    # The builder mints the charge's ref from the thread, message and call ids: without them, a
    # ref from the call id alone could collide with another turn's or conversation's.
    with pytest.raises(ValueError, match="thread_id and message_id"):
        decide(charge_call(), CHARGE, context(), **ids)


def test_thread_id_is_needed_only_for_financial_calls_under_a_mandate() -> None:
    assert decide(customer_call("lookup_customer"), LOOKUP, context()) == Allow(
        args=customer_args(), closed=None
    )
    assert isinstance(decide(charge_call(), CHARGE, without_mandate()), RequireApproval)


def test_builder_value_error_is_denied_without_its_message() -> None:
    def refuse(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        raise ValueError("payer cust-juan has no consent on file")

    decision = decide(
        charge_call(),
        financial_tool(builder=refuse),
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Deny(reasons=("invalid_args:closed:value_error",))


def test_builder_programming_errors_propagate() -> None:
    def broken(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
        raise KeyError("bug in the application's builder")

    with pytest.raises(KeyError):
        decide(
            charge_call(),
            financial_tool(builder=broken),
            context(),
            thread_id=THREAD_ID,
            message_id=MESSAGE_ID,
        )


def test_financial_tool_without_builder_fails_loud() -> None:
    # Only reachable by skipping ToolSpec validation; it must never fall through to allow.
    unbuilt = ToolSpec.model_construct(
        name="request_payment",
        risk="high",
        effects=SPENDS,
        reversible=False,
        financial=True,
        args_model=ChargeArgs,
        to_closed_mandate=None,
    )

    with pytest.raises(ValueError, match="no to_closed_mandate"):
        decide(charge_call(), unbuilt, context(), thread_id=THREAD_ID, message_id=MESSAGE_ID)


# Rule 5: high risk without money needs a human.


def test_high_risk_tool_without_money_requires_approval() -> None:
    decision = decide(
        customer_call("delete_customer"),
        DELETE,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == RequireApproval(reasons=("high_risk",), args=customer_args(), closed=None)


# Rule 6: Meta's Rule of Two.


def test_rule_of_two_allows_verified_mandate_but_not_unmandated_state_change() -> None:
    risky = session(**RULE_OF_TWO)

    verified = decide(
        charge_call(), CHARGE, context(session=risky), thread_id=THREAD_ID, message_id=MESSAGE_ID
    )
    state_change = decide(
        customer_call("tag_customer"),
        TAG,
        context(session=risky),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    unmandated = decide(
        charge_call(),
        CHARGE,
        without_mandate(session=risky),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert isinstance(verified, Allow)
    assert state_change == RequireApproval(
        reasons=("rule_of_two",), args=customer_args(), closed=None
    )
    assert unmandated == RequireApproval(
        reasons=("no_mandate", "rule_of_two"), args=charge_args(), closed=None
    )


@pytest.mark.parametrize("lowered", sorted(RULE_OF_TWO))
def test_rule_of_two_needs_all_three_flags(lowered: str) -> None:
    flags = RULE_OF_TWO | {lowered: False}

    decision = decide(
        customer_call("tag_customer"),
        TAG,
        context(session=session(**flags)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Allow(args=customer_args(), closed=None)


def test_rule_of_two_gates_external_communication() -> None:
    # Meta's third property is changing state OR communicating externally: the exfiltration path.
    risky = decide(
        customer_call("email_customer"),
        EMAIL,
        context(session=session(**RULE_OF_TWO)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    calm = decide(
        customer_call("email_customer"),
        EMAIL,
        context(),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert risky == RequireApproval(reasons=("rule_of_two",), args=customer_args(), closed=None)
    assert calm == Allow(args=customer_args(), closed=None)


def test_rule_of_two_leaves_reads_alone() -> None:
    decision = decide(
        customer_call("lookup_customer"),
        LOOKUP,
        context(session=session(**RULE_OF_TWO)),
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )

    assert decision == Allow(args=customer_args(), closed=None)


def test_approval_reasons_accumulate() -> None:
    risky = context(session=session(**RULE_OF_TWO))

    charge = decide(
        charge_call(payer="cust-ana", amount_minor=None),
        CHARGE,
        risky,
        thread_id=THREAD_ID,
        message_id=MESSAGE_ID,
    )
    delete = decide(
        customer_call("delete_customer"), DELETE, risky, thread_id=THREAD_ID, message_id=MESSAGE_ID
    )

    assert isinstance(charge, RequireApproval)
    assert charge.reasons == ("mandate:payer_not_allowed", "mandate:open_amount", "rule_of_two")
    assert isinstance(delete, RequireApproval)
    assert delete.reasons == ("high_risk", "rule_of_two")

"""`decide`: the deterministic gate between a tool call the model proposes and its execution."""

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
    DailyWindow,
    OpenMandate,
    RefState,
)
from agent_toolkit.policy_gate import (
    Allow,
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
UNKNOWN = RefState(status="unknown", reserved=None)


def mxn(minor: int) -> Amount:
    return Amount(minor=minor, currency="MXN")


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


class LeakyChargeArgs(ChargeArgs):
    """A careless application that lets the model name the acting agent."""

    agent_id: str


class CustomerArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    customer_id: str


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


def build_charge(args: BaseModel) -> ClosedMandate:
    """The application's builder: it binds the mandate, agent and charge ref, not the model."""
    assert isinstance(args, ChargeArgs)
    previous = args.previous_amount_minor
    return ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge" if previous is None else "amount_increase",
        charge_ref="ch_1",
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


def build_leaky_charge(args: BaseModel) -> ClosedMandate:
    """Copies the model's `agent_id` into the charge: exactly what the gate must not trust."""
    assert isinstance(args, LeakyChargeArgs)
    return ClosedMandate.model_validate(
        build_charge(args).model_dump() | {"agent_id": args.agent_id}
    )


SPENDS = Effects(reads_sensitive=False, changes_state=True, external_comm=True)
READS = Effects(reads_sensitive=True, changes_state=False, external_comm=False)
WRITES = Effects(reads_sensitive=False, changes_state=True, external_comm=False)


def financial_tool(
    args_model: type[BaseModel] = ChargeArgs,
    builder: Callable[[BaseModel], ClosedMandate] = build_charge,
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
    return ToolCall(id="call_1", name="request_payment", args=CHARGE_ARGS | overrides)


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
        "consumed": mxn(0),
        "ref_state": UNKNOWN,
        "budget_state": BudgetState.empty(),
        "budget_limits": BudgetLimits(),
        "now": NOW,
        "config": GateConfig(),
    }
    return GateContext.model_validate(fields | overrides)


def without_mandate(**overrides: object) -> GateContext:
    return context(mandate=None, consumed=None, ref_state=None, **overrides)


def config(on_mandate_violation: str) -> GateConfig:
    return GateConfig.model_validate({"on_mandate_violation": on_mandate_violation})


# Rule 7 and the plain allow path.


def test_low_risk_read_is_allowed_without_mandate() -> None:
    decision = decide(customer_call("lookup_customer"), LOOKUP, without_mandate())

    assert decision == Allow(args=customer_args(), closed=None)


def test_allow_returns_parsed_args_and_closed_mandate() -> None:
    # A high-risk financial tool inside its mandate needs no approval: the mandate governs it.
    decision = decide(charge_call(), CHARGE, context())

    assert isinstance(decision, Allow)
    assert isinstance(decision.args, ChargeArgs)
    assert decision.args == charge_args()
    assert decision.closed == ClosedMandate(
        open_mandate_id="om_1",
        agent_id="agent_1",
        kind="new_charge",
        charge_ref="ch_1",
        payer="cust-juan",
        receiving_account="acct-own",
        rail="codi_push",
        amount=mxn(35000),
        previous_amount=None,
        reference="R-1001",
    )


def test_increase_inside_its_mandate_is_allowed() -> None:
    ledger = RefState(status="reserved", reserved=mxn(30000))

    decision = decide(
        charge_call(previous_amount_minor=30000),
        CHARGE,
        context(consumed=mxn(30000), ref_state=ledger),
    )

    assert isinstance(decision, Allow)
    assert decision.closed is not None
    assert decision.closed.kind == "amount_increase"


# Rule 1: the call must be for this tool.


def test_tool_name_mismatch_is_denied() -> None:
    decision = decide(customer_call("delete_customer"), LOOKUP, context())

    assert decision == Deny(reasons=("tool_mismatch",))


# Rule 2: model output is untrusted, so arguments are validated strictly.


def test_args_are_validated_strictly() -> None:
    decision = decide(charge_call(amount_minor="35000"), CHARGE, context())

    assert decision == Deny(reasons=("invalid_args:args.amount_minor:int_type",))


def test_every_invalid_arg_is_a_reason() -> None:
    args = {key: value for key, value in CHARGE_ARGS.items() if key != "payer"}

    decision = decide(
        ToolCall(id="call_1", name="request_payment", args=args | {"rail": 7}), CHARGE, context()
    )

    assert decision == Deny(
        reasons=("invalid_args:args.payer:missing", "invalid_args:args.rail:string_type")
    )


def test_unknown_arg_key_is_denied() -> None:
    decision = decide(charge_call(agent_id="agent_2"), CHARGE, context())

    assert decision == Deny(reasons=("invalid_args:args.agent_id:extra_forbidden",))


@pytest.mark.parametrize("args_model", [IgnoringArgs, AllowingArgs])
def test_unknown_arg_key_is_denied_even_if_the_model_accepts_it(
    args_model: type[BaseModel],
) -> None:
    tool = plain_tool("tag_customer", "low", WRITES, args_model)

    decision = decide(customer_call("tag_customer", agent_id="agent_2"), tool, context())

    assert decision == Deny(reasons=("invalid_args:args.agent_id:extra_forbidden",))


def test_lax_models_are_validated_strictly_at_every_depth() -> None:
    tool = plain_tool("tag_customer", "low", WRITES, IgnoringArgs)

    coerced = decide(customer_call("tag_customer", priority="1"), tool, context())
    nested_coerced = decide(
        customer_call("tag_customer", note={"text": "vip", "weight": "5"}), tool, context()
    )
    nested_unknown = decide(
        customer_call("tag_customer", note={"text": "vip", "payer": "x"}), tool, context()
    )

    assert coerced == Deny(reasons=("invalid_args:args.priority:int_type",))
    assert nested_coerced == Deny(reasons=("invalid_args:args.note.weight:int_type",))
    assert nested_unknown == Deny(reasons=("invalid_args:args.note.payer:extra_forbidden",))


def test_args_are_checked_before_the_budget() -> None:
    spent = BudgetState.empty().model_copy(update={"tool_calls": 5})

    decision = decide(
        charge_call(amount_minor="1"),
        CHARGE,
        context(budget_state=spent, budget_limits=BudgetLimits(max_tool_calls=1)),
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
    )

    assert decision == Deny(reasons=("budget_exceeded:tool_calls", "budget_exceeded:tokens"))


def test_budget_is_checked_before_the_mandate() -> None:
    spent = BudgetState.empty().model_copy(update={"consecutive_failures": 4})

    decision = decide(
        charge_call(receiving_account="acct-other"),
        CHARGE,
        context(budget_state=spent, budget_limits=BudgetLimits(max_consecutive_failures=3)),
    )

    assert decision == Deny(reasons=("budget_exceeded:consecutive_failures",))


# Rule 4: financial calls are judged against the mandate.


def test_financial_call_without_mandate_requires_approval() -> None:
    decision = decide(charge_call(), CHARGE, without_mandate())

    assert decision == RequireApproval(reasons=("no_mandate",), args=charge_args(), closed=None)


def test_foreign_account_is_denied_even_if_config_says_approval() -> None:
    # Rule 7.3(c): charges go only to the credential holder's accounts; no approval overrides it.
    decision = decide(
        charge_call(receiving_account="acct-other"),
        CHARGE,
        context(config=config("require_approval")),
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
        {"previous_amount_minor": 30000}, {}, "previous_amount_mismatch", id="previous_amount"
    ),
    pytest.param(
        {},
        {"ref_state": RefState(status="reserved", reserved=mxn(100))},
        "charge_ref_reused",
        id="ref_reserved",
    ),
    pytest.param(
        {},
        {"ref_state": RefState(status="released", reserved=None)},
        "charge_ref_reused",
        id="ref_released",
    ),
]


@pytest.mark.parametrize("on_violation", ["deny", "require_approval"])
@pytest.mark.parametrize(("args", "ctx", "violation"), ALWAYS_DENIED)
def test_binding_violations_are_always_denied(
    args: dict[str, object], ctx: dict[str, object], violation: str, on_violation: str
) -> None:
    decision = decide(charge_call(**args), CHARGE, context(config=config(on_violation), **ctx))

    assert decision == Deny(reasons=(f"mandate:{violation}",))


def test_violation_follows_config() -> None:
    out_of_range = charge_call(amount_minor=600000)

    approve = decide(out_of_range, CHARGE, context(config=config("require_approval")))
    deny = decide(out_of_range, CHARGE, context(config=config("deny")))

    assert approve == RequireApproval(
        reasons=("mandate:amount_out_of_range",),
        args=charge_args(amount_minor=600000),
        closed=build_charge(charge_args(amount_minor=600000)),
    )
    assert deny == Deny(reasons=("mandate:amount_out_of_range",))


CONFIGURABLE = [
    pytest.param({"rail": "spei"}, {}, "rail_not_allowed", id="rail"),
    pytest.param({"amount_minor": 600000}, {}, "amount_out_of_range", id="range"),
    pytest.param({}, {"consumed": mxn(990000)}, "budget_exceeded", id="budget"),
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

    approve = decide(call, CHARGE, context(config=config("require_approval"), **ctx))
    deny = decide(call, CHARGE, context(config=config("deny"), **ctx))

    assert isinstance(approve, RequireApproval)
    assert approve.reasons == (f"mandate:{violation}",)
    assert deny == Deny(reasons=(f"mandate:{violation}",))


def test_open_amount_qr_requires_approval() -> None:
    decision = decide(charge_call(amount_minor=None), CHARGE, context())

    assert decision == RequireApproval(
        reasons=("mandate:open_amount",),
        args=charge_args(amount_minor=None),
        closed=build_charge(charge_args(amount_minor=None)),
    )


def test_unknown_payer_requires_approval() -> None:
    decision = decide(charge_call(payer="cust-ana"), CHARGE, context())

    assert isinstance(decision, RequireApproval)
    assert decision.reasons == ("mandate:payer_not_allowed",)


def test_deny_takes_precedence_over_approval() -> None:
    decision = decide(
        charge_call(receiving_account="acct-other", payer="cust-ana", amount_minor=None),
        CHARGE,
        context(session=session(**RULE_OF_TWO)),
    )

    assert decision == Deny(reasons=("mandate:account_not_allowed",))


def test_deny_lists_every_denying_violation() -> None:
    call = charge_call(receiving_account="acct-other", rail="spei")

    strict = decide(call, CHARGE, context(config=config("deny")))
    lenient = decide(call, CHARGE, context(config=config("require_approval")))

    assert strict == Deny(reasons=("mandate:account_not_allowed", "mandate:rail_not_allowed"))
    assert lenient == Deny(reasons=("mandate:account_not_allowed",))


@pytest.mark.parametrize("missing", ["consumed", "ref_state"])
def test_financial_call_under_mandate_needs_both_ledger_inputs(missing: str) -> None:
    # Treating a missing ledger value as zero would overspend: it is a programming error.
    with pytest.raises(ValueError, match=missing):
        decide(charge_call(), CHARGE, context(**{missing: None}))


def test_ledger_inputs_are_needed_only_for_financial_calls_under_a_mandate() -> None:
    no_ledger = context(consumed=None, ref_state=None)

    assert decide(customer_call("lookup_customer"), LOOKUP, no_ledger) == Allow(
        args=customer_args(), closed=None
    )
    assert isinstance(decide(charge_call(), CHARGE, without_mandate()), RequireApproval)


def test_tool_args_cannot_change_the_acting_agent() -> None:
    # The mandate belongs to agent_2. The session runs agent_1, and the model claims agent_2.
    leaky = financial_tool(LeakyChargeArgs, build_leaky_charge)
    call = charge_call(agent_id="agent_2")

    impersonating = decide(call, leaky, context(mandate=mandate(agent_id="agent_2")))
    own_agent = decide(
        call,
        leaky,
        context(session=session(agent_id="agent_2"), mandate=mandate(agent_id="agent_2")),
    )

    assert impersonating == Deny(reasons=("agent_mismatch",))
    assert isinstance(own_agent, Allow)


def test_closed_mandate_that_does_not_validate_is_denied() -> None:
    # Amount allows 0 and rejects negatives; a charge needs a positive amount.
    zero = decide(charge_call(amount_minor=0), CHARGE, context())
    negative = decide(charge_call(amount_minor=-5), CHARGE, context())

    assert zero == Deny(reasons=("invalid_args:closed:value_error",))
    assert negative == Deny(reasons=("invalid_args:closed.minor:greater_than_equal",))


def test_builder_value_error_is_denied_without_its_message() -> None:
    def refuse(args: BaseModel) -> ClosedMandate:
        raise ValueError("payer cust-juan has no consent on file")

    decision = decide(charge_call(), financial_tool(builder=refuse), context())

    assert decision == Deny(reasons=("invalid_args:closed:value_error",))


def test_builder_programming_errors_propagate() -> None:
    def broken(args: BaseModel) -> ClosedMandate:
        raise KeyError("bug in the application's builder")

    with pytest.raises(KeyError):
        decide(charge_call(), financial_tool(builder=broken), context())


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
        decide(charge_call(), unbuilt, context())


# Rule 5: high risk without money needs a human.


def test_high_risk_tool_without_money_requires_approval() -> None:
    decision = decide(customer_call("delete_customer"), DELETE, context())

    assert decision == RequireApproval(reasons=("high_risk",), args=customer_args(), closed=None)


# Rule 6: Meta's Rule of Two.


def test_rule_of_two_allows_verified_mandate_but_not_unmandated_state_change() -> None:
    risky = session(**RULE_OF_TWO)

    verified = decide(charge_call(), CHARGE, context(session=risky))
    state_change = decide(customer_call("tag_customer"), TAG, context(session=risky))
    unmandated = decide(charge_call(), CHARGE, without_mandate(session=risky))

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

    decision = decide(customer_call("tag_customer"), TAG, context(session=session(**flags)))

    assert decision == Allow(args=customer_args(), closed=None)


def test_rule_of_two_leaves_calls_without_state_change_alone() -> None:
    decision = decide(
        customer_call("lookup_customer"), LOOKUP, context(session=session(**RULE_OF_TWO))
    )

    assert decision == Allow(args=customer_args(), closed=None)


def test_approval_reasons_accumulate() -> None:
    risky = context(session=session(**RULE_OF_TWO))

    charge = decide(charge_call(payer="cust-ana", amount_minor=None), CHARGE, risky)
    delete = decide(customer_call("delete_customer"), DELETE, risky)

    assert isinstance(charge, RequireApproval)
    assert charge.reasons == ("mandate:payer_not_allowed", "mandate:open_amount", "rule_of_two")
    assert isinstance(delete, RequireApproval)
    assert delete.reasons == ("high_risk", "rule_of_two")

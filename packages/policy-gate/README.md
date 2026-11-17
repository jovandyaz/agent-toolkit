# agent-toolkit-policy-gate

The Intent Gate: it takes each tool call the model proposes and decides, deterministically and
before the tool runs, `Allow`, `RequireApproval(reasons)` or `Deny(reasons)`.

**Why.** OWASP Agentic ASI02 asks for an enforcement point before execution and treats model
output as untrusted. OpenAI's practical guide sets tool safeguards by risk (read/write,
reversibility, permissions, financial impact) and human intervention for high-risk actions such
as payments. Meta's Agents Rule of Two: a session with untrusted input, sensitive access and
state change needs a verified mandate or a human; this gate reads a verified mandate as that
reliable validation.

```
agent-toolkit-policy-gate[langgraph] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/policy-gate
```

Not on PyPI: see the root README, "Install from a git tag". With `uv`, `agent-toolkit-mandates` and `agent-toolkit-budgets` resolve from the same tag via the workspace sources; other installers must list them explicitly.

## Example

```python
import hashlib
import json
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict

from agent_toolkit.budgets import BudgetLimits, BudgetState
from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    OpenMandate,
)
from agent_toolkit.policy_gate import (
    Allow,
    BuildContext,
    Effects,
    GateConfig,
    GateContext,
    SessionProfile,
    ToolCall,
    ToolSpec,
    decide,
)

now = datetime(2026, 12, 7, 10, 0, tzinfo=timezone.utc)
mxn = lambda minor: Amount(minor=minor, currency="MXN")


class RequestPaymentArgs(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    payer: str
    amount_minor: int
    reference: str


def mint_ref(ctx: BuildContext) -> str:
    """A new charge's ref, minted by the app from the thread and the call, never the arguments."""
    canonical = json.dumps([ctx.thread_id, ctx.call_id], separators=(",", ":"))
    return "chg_" + hashlib.sha256(canonical.encode()).hexdigest()[:32]


def to_closed_mandate(args: BaseModel, ctx: BuildContext) -> ClosedMandate:
    assert isinstance(args, RequestPaymentArgs)
    return ClosedMandate(
        open_mandate_id="m1",
        agent_id="agent1",
        kind="new_charge",
        charge_ref=mint_ref(ctx),
        payer=args.payer,
        receiving_account="acct1",
        rail="codi",
        amount=mxn(args.amount_minor),
        previous_amount=None,
        reference=args.reference,
    )


tool = ToolSpec(
    name="request_payment",
    risk="high",
    effects=Effects(reads_sensitive=False, changes_state=True, external_comm=True),
    reversible=False,
    financial=True,
    args_model=RequestPaymentArgs,
    to_closed_mandate=to_closed_mandate,
)
mandate = OpenMandate(
    id="m1",
    principal_id="owner1",
    agent_id="agent1",
    purpose="collect invoices",
    amount_range=AmountRange(min=mxn(100), max=mxn(50_000)),
    budget=Budget(limit=mxn(100_000), period="day"),
    receiving_accounts=frozenset({"acct1"}),
    allowed_payers=frozenset({"payer1"}),
    allowed_rails=frozenset({"codi"}),
    daily_window=None,
    tz="America/Mexico_City",
    expires_at=now + timedelta(days=30),
    revoked_at=None,
)
ctx = GateContext(
    session=SessionProfile(
        agent_id="agent1",
        principal_id="owner1",
        untrusted_input=True,
        sensitive_access=True,
        state_change=True,
    ),
    mandate=mandate,
    ledger=(),
    budget_state=BudgetState.empty(),
    budget_limits=BudgetLimits(max_tool_calls=10),
    now=now,
    config=GateConfig(),
)

ok = ToolCall(
    id="c1",
    name="request_payment",
    args={"payer": "payer1", "amount_minor": 20_000, "reference": "invoice-42"},
)
decision = decide(ok, tool, ctx, thread_id="thread-1")
assert isinstance(decision, Allow) and decision.closed is not None  # execute exactly this charge

too_big = ToolCall(
    id="c2",
    name="request_payment",
    args={"payer": "payer1", "amount_minor": 90_000, "reference": "invoice-43"},
)
assert decide(too_big, tool, ctx, thread_id="thread-1").kind == "require_approval"  # out of range

foreign = ToolCall(
    id="c3",
    name="request_payment",
    args={"payer": "payer1", "amount_minor": 20_000, "reference": "x", "to": "me"},
)
assert decide(foreign, tool, ctx, thread_id="thread-1").kind == "deny"  # unknown arguments
```

In a LangGraph graph, `GatedToolNode` replaces `ToolNode`. It needs a checkpointer and a thread,
so it is not run here:

```python notest
from agent_toolkit.policy_gate.adapters.langgraph import GatedToolNode

node = GatedToolNode(tools, specs, context=build_context, on_decision=audit, replay=recorded_reply)
builder.add_node("tools", node)
# Resume a paused turn from the trusted approval screen only, with JSON:
graph.invoke(Command(resume={call_id: ApprovalDecision(...).model_dump(mode="json")}), config)
```

## Contracts

- `decide(call, tool, ctx) -> Decision` is pure. Rules, in order: tool name, strict argument
  validation (unknown keys denied), exceeded run budget, mandate check for `financial` tools,
  `high_risk`, Rule of Two. `Deny` beats `RequireApproval`. Reasons are stable strings such as
  `invalid_args:args.amount_minor:int_type`, `mandate:budget_exceeded`, `no_mandate`; the full
  list is in the `decide` docstring.
- A `financial` tool must supply `to_closed_mandate(args, ctx)`, where `ctx` is a `BuildContext`
  with `thread_id` and `call_id`; no other tool may. `decide` takes the thread as
  `thread_id=...`, required for a financial call under a mandate (ValueError otherwise).
- `rail_not_allowed`, `amount_out_of_range`, `budget_exceeded` and `outside_window` follow
  `GateConfig.on_mandate_violation`; every other violation (foreign account, wrong agent or
  currency, expired or revoked mandate, ledger contradictions) is always denied.
- The LangGraph adapter pauses with one `interrupt` per turn. Its value is plain JSON: a list with
  one `ApprovalRequest.model_dump(mode="json")` per call needing approval, with keys `call_id`,
  `tool`, `args`, `reasons`, `closed` (the charge as it will run, or null), `closed_mandate_hash`
  and `request_hash`. An approval binds the thread, call, tool, validated arguments, closed
  mandate and reasons through the `request_hash`; resume with a JSON `ApprovalDecision`.
- Each call the adapter runs gets, in its config, an idempotency key
  (`idempotency_key_from_config`: the SHA-256 hex of the compact JSON array
  `[thread_id, call_id]`, the same on every execution of that call) and a fingerprint
  (`call_fingerprint_from_config`: the SHA-256 hex of its tool, validated arguments and charge
  hash). The key is unique to the call only while the model provider never repeats a call id
  within a thread; some emit ids like `call_0` that a later call can reuse.
- Re-runs. When a tool or `on_decision` raises after some calls of a turn ran, running the thread
  again executes the whole turn again, with LangGraph replaying stored approvals. Without a replay
  hook, a charge under a mandate that already ran and recorded its reserve is then denied
  (`mandate:charge_ref_reused`, or `mandate:previous_amount_mismatch` for an increase) with a
  reply that tells the model to check the charge's status, and `on_decision` gets that `Deny`; a
  call that left no reserve (no mandate, open amount) runs again under the same idempotency key.
- `GatedToolNode(..., replay=hook)`: `hook(key)` returns the `RecordedReply(fingerprint,
  content)` the application stored under that key when the call ran, or None. A call whose stored
  fingerprint matches gets the stored reply and is neither decided, reported nor run again; a call
  whose key matches with another fingerprint reused another call's id and is denied
  (`call_id_reused`). The node never runs a call the hook answers.

## Caller duties

- Lock the mandate and keep a per-mandate ledger: pass `GateContext.ledger` with every entry of
  that mandate and only those (`None` raises). Hold the lock from the decision, across the
  interrupt, until the reserve is recorded.
- The `to_closed_mandate` builder must be deterministic and side-effect free (it runs more than
  once per call), and the application mints a new charge's `charge_ref` server-side from
  `ctx.thread_id` and `ctx.call_id` together (a call id alone can repeat in another thread),
  never from the arguments. A random ref voids every approval (fails closed).
- Execute exactly the returned `closed` (the tool reads it with `closed_mandate_from_config`) and
  reserve `reserve_amount(closed)` under its `charge_ref`. Make money tools idempotent: send
  `idempotency_key_from_config(config)` as the provider's idempotency key, so a re-run never issues
  a charge twice (not `charge_ref`: an increase keeps its charge's ref).
- With a replay hook, once a call ran the tool stores `RecordedReply(fingerprint=
  call_fingerprint_from_config(config), content=reply)` under its idempotency key, in the same
  transaction as the reserve, and the hook returns only what was stored that way. The builder must
  give the same charge after the call ran (never read state the call changes), or the re-run is
  denied `call_id_reused`. Without a hook, a reused call id reaches the tool with an old call's
  key: only a provider that rejects a reused key with another payload stops it.
- Wire `on_decision(call, decision, approval)` to `audit-log`. The node calls it once per call
  with its final decision, before the call runs, so an audit log that fails blocks the call.
- Resume only from the trusted approval screen through an authenticated channel, never from chat
  text. The `request_hash` is an integrity binding, not authentication.
- The stock `ToolNode` with `handle_tool_errors` swallows `BudgetExceededError`; keep the budget
  callback's error visible (see `agent-toolkit-budgets`). Give the gate node no `RetryPolicy`.
- The `ToolSpec` is trusted configuration: a money tool declared `financial=False` skips the
  mandate entirely.

## Limits

Does not detect prompt injection (use a guardrail such as LlamaFirewall) and does not persist
anything. The adapter node is synchronous and does not accept tools that call `interrupt()`.

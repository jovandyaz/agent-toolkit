# agent-toolkit-mandates

What an owner authorized an agent to charge for (an open mandate), one concrete charge the agent
wants to issue (a closed mandate), and a pure check of the second against the first.

**Why.** AP2 v0.2 models agent authority as open mandates (constraints) and closed mandates (one
concrete operation) approved on a deterministic Trusted Surface. Here the money flows the other
way: the owner authorizes the agent to *issue payment requests* (CoDi), and each payer approves in
their own banking app. Amounts are integers in minor units plus an ISO 4217 code, as in AP2.

```
agent-toolkit-mandates @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/mandates
```

Not on PyPI: see the root README, "Install from a git tag".

## Example

```python
from datetime import datetime, timedelta, timezone

from agent_toolkit.mandates import (
    Amount,
    AmountRange,
    Budget,
    ClosedMandate,
    ConsumptionEntry,
    OpenMandate,
    check,
    consumed_in_period,
    ref_state,
    reserve_amount,
)

now = datetime(2026, 12, 7, 10, 0, tzinfo=timezone.utc)
mxn = lambda minor: Amount(minor=minor, currency="MXN")

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
charge = ClosedMandate(
    open_mandate_id="m1",
    agent_id="agent1",
    kind="new_charge",
    charge_ref="chg_1",
    payer="payer1",
    receiving_account="acct1",
    rail="codi",
    amount=mxn(20_000),
    previous_amount=None,
    reference="invoice-42",
)

ledger: list[ConsumptionEntry] = []  # this mandate's entries, and only its entries
verdict = check(
    charge,
    mandate,
    consumed_in_period(ledger, mandate, now),
    now,
    ref_state=ref_state(ledger, charge.charge_ref),
)
assert verdict.ok

# After issuing the charge, record what it reserved under its charge_ref.
reserved = reserve_amount(charge)
assert reserved is not None
ledger.append(ConsumptionEntry(charge_ref="chg_1", kind="reserve", amount=reserved, at=now))
assert consumed_in_period(ledger, mandate, now) == mxn(20_000)
```

## Contracts

- `check(closed, open, consumed, now, *, ref_state) -> Verdict` is pure. It returns every
  violation (`expired`, `revoked`, `agent_mismatch`, `account_not_allowed`, `budget_exceeded`,
  `charge_ref_reused`, ...) and every approval reason (`payer_not_allowed`, `open_amount`). An
  open amount is never bounded, so it always needs approval.
- Budget counts what was **requested**, not paid. `consumed_in_period` sums unreleased reserves
  in the mandate's time zone (day, week from Monday, month, or total). A charge that expired
  unpaid is released only once the provider confirms it. Payments never consume.
- An `amount_increase` consumes only its delta, under the same `charge_ref`, and `check` requires
  the ledger to hold exactly its claimed `previous_amount`.
- `ClosedMandate.content_hash` (SHA-256 of canonical JSON) identifies a charge for signatures
  (`Signer`/`Verifier`, `HmacSha256Signer`) and `Receipt`s.

## Caller duties

- Lock the mandate from reading its ledger until the reserve is recorded. Two checks against one
  snapshot can both pass and together overspend.
- Keep one ledger per mandate. Pass `consumed` and `ref_state` from that ledger, never from tool
  arguments.
- Mint a new charge's `charge_ref` server-side. `agent_id` comes from the session, not the model.
- For an `amount_increase`, load `payer`, `receiving_account` and `rail` from the charge you
  stored under its `charge_ref`; only the amounts come from tool arguments. `check` binds
  `previous_amount` to the ledger, but cannot tell the payer, account or rail from the ones the
  charge was issued with, so an increase naming an allowed payer would skip the payer approval.
- The `HmacSha256Signer` key (32 bytes or more) is an application secret: keep it in a secrets
  manager, never in the repo.

## Limits

No persistence, no execution, no knowledge of any payment provider. SD-JWT/AP2 wire compatibility
is out of scope; signing sits behind the `Signer`/`Verifier` ports.

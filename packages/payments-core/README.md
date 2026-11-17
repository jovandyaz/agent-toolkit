# agent-toolkit-payments-core

Request-to-pay payment domain without I/O: money, payment intents, attempts on a rail, and the
policies that reconcile what a provider reports. Built for push or request-to-pay rails (CoDi; by
extension Pix, UPI).

**Why.** Money moves wrong in a few predictable ways: duplicates, late payments, orphans and
retries that double-charge. The policies encode those cases as pure functions, with acceptance
cases drawn from a prior collections project (see the design spec, section 7.6). Money is an
integer in minor units plus an ISO 4217 code, never a float (AP2 v0.2 does the same).

```
agent-toolkit-payments-core @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/payments-core
```

Not on PyPI: see the root README, "Install from a git tag".

## Example

```python
from datetime import datetime, timedelta, timezone

from agent_toolkit.payments_core import (
    Attempt,
    Money,
    PaymentIntent,
    ProviderPayment,
    apply_provider_payment,
    may_retry_request,
)

now = datetime(2026, 12, 7, 10, 0, tzinfo=timezone.utc)
mxn = lambda minor: Money(amount_minor=minor, currency="MXN")

intent = PaymentIntent(
    id="pi_1",
    principal_id="owner1",
    reference="invoice-42",
    amount=mxn(20_000),
    idempotency_key="idem-1",
    status="pending",
    paid_total=mxn(0),
    late=False,
    applied_event_ids=frozenset(),
    attempts=(
        Attempt(
            id="att_1",
            rail="codi",
            provider_ref="ref_1",
            amount=mxn(20_000),
            created_at=now,
            expires_at=now + timedelta(hours=1),
            status="requested",
        ),
    ),
)
payment = ProviderPayment(
    provider_event_id="evt_1",
    provider_ref="ref_1",
    amount=mxn(20_000),
    received_at=now + timedelta(minutes=5),
    rail="codi",
)

outcome = apply_provider_payment(intent, payment)
assert outcome.kind == "confirmed" and outcome.intent is not None
assert outcome.intent.status == "paid"

replay = apply_provider_payment(outcome.intent, payment)  # same provider_event_id
assert replay.kind == "already_applied" and not replay.refund_flags

assert may_retry_request("not_sent") and not may_retry_request("ambiguous")
```

## Contracts

- Policies are pure and never mutate their input: `apply_provider_payment(intent, payment)`,
  `reconcile_attempt(intent, attempt_id, status, now)`, `may_retry_request(outcome)`.
- In time: `confirmed`. Late: `confirmed_late`, marked. Duplicate or no intent: a `RefundFlag`
  (the rail cannot refund by API). Excess over the amount owed: flagged `excess`, never applied.
- Ports: `RailConnector` (`request`, `query`), `IntentRepository`, `UnitOfWork`. In-memory doubles
  live in `agent_toolkit.payments_core.testing`, for tests only.
- A POST to the provider is retried only when the failure proves it was not processed
  (`not_sent`); an `ambiguous` attempt is reconciled, never resent.

## Caller duties

- Read the intent, run the policy and save the outcome in one `UnitOfWork` that locks the intent.
- Deduplicate `orphan` refund flags by `provider_event_id`.
- Assign an attempt's `provider_ref` before sending it, so an ambiguous attempt can be queried.
- The webhook and the provider query must report the same `provider_event_id` for the same money.

## Limits

No I/O and no provider: the connector lives in the application. Ports are synchronous Protocols
(wrap them for async). Knows nothing about bookings or scheduling.

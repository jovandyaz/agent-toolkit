# agent-toolkit-audit-log

Append-only, hash-chained log of agent decisions, approvals, receipts and external effects, with
the identity of the actor (agent, human or system).

**Why.** Microsoft's agentic failure-mode taxonomy asks for a distinct identity per agent and
audit of which component did what; OWASP Agentic ASI02 and ASI09 ask for immutable logs; AP2
asks for non-repudiable evidence. Each event stores the previous event's hash, so changing,
removing or reordering one breaks the chain at that point.

```
agent-toolkit-audit-log[postgres] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/audit-log
```

Not on PyPI: see the root README, "Install from a git tag".

## Example

```python
from datetime import datetime, timezone

from agent_toolkit.audit_log import append, hash_payload, verify_chain
from agent_toolkit.audit_log.testing import InMemoryAuditStore

key = bytes(range(32))  # demo only: load the real key from your secrets manager
store = InMemoryAuditStore()

append(
    store,
    actor_id="agent1",
    actor_kind="agent",
    action="gate.decision",
    subject="chg_c1",
    payload_hash=hash_payload(b'{"payer":"5512345678"}', key),  # a keyed hash, never the payload
    decision="allow",
    at=datetime.now(timezone.utc),
)
append(
    store,
    actor_id="human1",
    actor_kind="human",
    action="approval.granted",
    subject="chg_c1",
    payload_hash=hash_payload(b"approval-77", key),
    decision=None,
    at=datetime.now(timezone.utc),
)
assert verify_chain(store.iter_all()).ok
```

With Postgres (needs a database and the `schema.sql` migration, so not run here):

```python notest
import psycopg
from agent_toolkit.audit_log import append
from agent_toolkit.audit_log.adapters.postgres import PostgresAuditStore

with psycopg.connect(DATABASE_URL) as conn:  # the app role has SELECT and INSERT only
    append(PostgresAuditStore(conn), ...)  # same transaction as the effect it records
```

## Contracts

- `AuditEvent`: `seq`, `actor_id`, `actor_kind`, `action`, `subject`, `payload_hash`, `decision`,
  `at`, `prev_hash`, `hash` (SHA-256 of the canonical JSON of every other field).
- `append(store, ...)` builds the event after the store's tip; `ChainConflict` means another
  append won, so roll back and call it again. `verify_chain(events)` reports the first bad `seq`
  and never raises. Storage sits behind the `AuditStore` protocol.
- `payload_hash` is `hash_payload(payload, key)`, an HMAC-SHA256, so low-entropy payloads (phone,
  account numbers) cannot be recovered by guessing.

## Caller duties

- The HMAC key (32 bytes or more) is an application secret in a secrets manager or the Keychain,
  never in the repo. Rotating it stops old hashes from being recomputed.
- Store only references in `actor_id` and `subject`, never names or account numbers.
- Append in the same transaction as the effect it records. Run the app as a role without
  `UPDATE` or `DELETE` on the table (see `schema.sql`).
- Wire the gate's `on_decision` callback to `append`.
- A hash chain cannot show that its newest events were removed: periodically record `last().hash`
  outside the log.

## Limits

Not telemetry (use OTel or Langfuse traces for that) and not a PII store. `InMemoryAuditStore` is
for tests only.

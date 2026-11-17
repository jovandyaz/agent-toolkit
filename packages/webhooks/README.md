# agent-toolkit-webhooks

Standard Webhooks signature verification over the raw request body, and an idempotent inbox that
processes each webhook id once.

**Why.** Webhooks are retried and replayed, so the effect must be recorded in the same
transaction as the webhook id. Signature checking uses the `standardwebhooks` library, the one
the provider's Python example uses (design spec, section 7.7).

```
agent-toolkit-webhooks @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/webhooks
```

Not on PyPI: see the root README, "Install from a git tag".

## Example

```python
from datetime import datetime, timezone

from agent_toolkit.webhooks import process_once, verify
from agent_toolkit.webhooks.testing import InMemoryInbox, sign

secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"  # a public documentation value, not a secret
body = b'{"type":"payment.paid"}'
headers = sign(body, secret, "msg_1", datetime.now(timezone.utc))

webhook = verify(body, headers, secret)  # raises InvalidSignatureError when it does not verify
inbox = InMemoryInbox()
handled = []

assert process_once(inbox, webhook, lambda w: handled.append(w.id) or "done") == "done"
assert process_once(inbox, webhook, lambda w: handled.append(w.id)) is None  # replay
assert handled == ["msg_1"]
```

## Contracts

- `verify(body, headers, secret) -> VerifiedWebhook` or `InvalidSignatureError`. Header names are
  case-insensitive; the library's 5-minute timestamp tolerance applies.
- `process_once(inbox, webhook, handler)` records the id, then runs the handler only if the id is
  new; a replay returns `None`. Handler errors propagate.

## Caller duties

- Pass `verify` the raw body exactly as received, never re-serialized JSON.
- Run `inbox.record` and the handler in one transaction: if the handler fails, the rollback undoes
  the record and the provider's retry is processed again. The real `Inbox` needs a unique key on
  the webhook id.
- The webhook secret is an application secret: keep it in a secrets manager, not the repo.

## Limits

Bodies must be UTF-8 (`standardwebhooks` 1.1 decodes them). No storage: `InMemoryInbox` is for
tests only (single-threaded, not persisted).

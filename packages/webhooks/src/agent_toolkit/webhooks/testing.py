"""Test helpers: an in-memory inbox and a signer that uses the library itself."""

from datetime import datetime

from standardwebhooks.webhooks import Webhook


class InMemoryInbox:
    """`Inbox` backed by a dict. For tests only: single-threaded, nothing persisted."""

    def __init__(self) -> None:
        self._seen: dict[str, datetime] = {}

    def record(self, webhook_id: str, received_at: datetime) -> bool:
        if webhook_id in self._seen:
            return False
        self._seen[webhook_id] = received_at
        return True


def sign(body: bytes, secret: str, msg_id: str, ts: datetime) -> dict[str, str]:
    """Return valid Standard Webhooks headers for `body` (UTF-8 only, like the library)."""
    signature = Webhook(secret).sign(msg_id, ts, body.decode())
    return {
        "webhook-id": msg_id,
        "webhook-timestamp": str(int(ts.timestamp())),
        "webhook-signature": signature,
    }

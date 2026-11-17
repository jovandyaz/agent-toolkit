"""Idempotent inbox: process each verified webhook exactly once."""

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from agent_toolkit.webhooks.verify import VerifiedWebhook


class Inbox(Protocol):
    """Remembers which webhook ids were seen."""

    def record(self, webhook_id: str, received_at: datetime) -> bool:
        """Record the id. Return False if it already existed.

        A real implementation uses a unique key, in the same transaction as the effect.
        """
        ...


def process_once[T](
    inbox: Inbox, webhook: VerifiedWebhook, handler: Callable[[VerifiedWebhook], T]
) -> T | None:
    """Record the webhook, then call `handler` only if it is new; a replay returns None.

    Transaction contract: the caller wraps `inbox.record` and the handler in ONE transaction.
    If the handler raises, the exception propagates (never swallowed), so the caller's rollback
    also undoes the record and the provider's retry is processed again.
    """
    if not inbox.record(webhook.id, webhook.timestamp):
        return None
    return handler(webhook)

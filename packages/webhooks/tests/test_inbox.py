from datetime import UTC, datetime

import pytest

from agent_toolkit.webhooks import VerifiedWebhook, process_once
from agent_toolkit.webhooks.testing import InMemoryInbox

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def _webhook(webhook_id: str) -> VerifiedWebhook:
    return VerifiedWebhook(id=webhook_id, timestamp=NOW, body=b"{}")


def test_process_once_skips_replay() -> None:
    inbox = InMemoryInbox()
    calls: list[str] = []

    def handler(webhook: VerifiedWebhook) -> str:
        calls.append(webhook.id)
        return "done"

    assert process_once(inbox, _webhook("msg_1"), handler) == "done"
    assert process_once(inbox, _webhook("msg_1"), handler) is None
    assert calls == ["msg_1"]


def test_distinct_ids_are_both_processed() -> None:
    inbox = InMemoryInbox()
    assert process_once(inbox, _webhook("a"), lambda w: w.id) == "a"
    assert process_once(inbox, _webhook("b"), lambda w: w.id) == "b"


def test_handler_error_propagates() -> None:
    inbox = InMemoryInbox()

    def handler(webhook: VerifiedWebhook) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        process_once(inbox, _webhook("msg_1"), handler)


def test_inbox_record_reports_first_time_only() -> None:
    inbox = InMemoryInbox()
    assert inbox.record("x", NOW) is True
    assert inbox.record("x", NOW) is False

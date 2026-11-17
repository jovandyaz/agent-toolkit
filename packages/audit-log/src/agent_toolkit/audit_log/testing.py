"""Test helpers: an in-memory audit store."""

from collections.abc import Iterator

from agent_toolkit.audit_log.core import GENESIS_HASH, AuditEvent, ChainConflict


class InMemoryAuditStore:
    """`AuditStore` backed by a list. For tests only: single-threaded, nothing persisted."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def last(self) -> AuditEvent | None:
        return self._events[-1] if self._events else None

    def append(self, event: AuditEvent) -> None:
        if event.hash != event.content_hash:
            raise ValueError(f"event {event.seq} hash does not match its content")
        tip = self.last()
        expected = (1, GENESIS_HASH) if tip is None else (tip.seq + 1, tip.hash)
        if (event.seq, event.prev_hash) != expected:
            raise ChainConflict(f"event {event.seq} does not extend the current tip")
        self._events.append(event)

    def iter_all(self) -> Iterator[AuditEvent]:
        return iter(tuple(self._events))

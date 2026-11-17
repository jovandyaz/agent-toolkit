"""Append-only, hash-chained audit log of agent decisions, approvals and effects.

`append` builds the event that follows a store's tip and appends it; `verify_chain` reports the
first event that was altered, removed or reordered. Events hold hashes and references, never PII
in clear. Storage plugs in through `AuditStore`; `agent_toolkit.audit_log.adapters.postgres`
ships one (extra `postgres`).
"""

from agent_toolkit.audit_log.core import (
    GENESIS_HASH,
    ActorKind,
    AuditEvent,
    AuditStore,
    ChainConflict,
    ChainReport,
    append,
    hash_payload,
    next_event,
    verify_chain,
)

__all__ = [
    "GENESIS_HASH",
    "ActorKind",
    "AuditEvent",
    "AuditStore",
    "ChainConflict",
    "ChainReport",
    "append",
    "hash_payload",
    "next_event",
    "verify_chain",
]

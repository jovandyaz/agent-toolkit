"""Append-only, hash-chained audit events with actor identity.

Each event records who did what to which subject: gate decisions, mandates, approvals, receipts
and external effects (Microsoft agentic failure-mode taxonomy: identity and audit; OWASP ASI02
and ASI09: immutable logs). Event `seq` starts at 1 and grows by one; each event stores the
`hash` of the one before it in `prev_hash` (the genesis event stores `GENESIS_HASH`), and its own
`hash` is the SHA-256 of the canonical JSON of every other field. Changing, removing or reordering
any event therefore breaks the chain at that point, and `verify_chain` reports where.

No PII in clear: the payload itself is never stored, only `payload_hash`, a keyed hash (see
`hash_payload`) that nobody without the key can recompute or test guesses against.
`actor_id` and `subject` are references (ids such as an agent id or a charge id), never names,
phone numbers or account numbers.

A hash chain cannot show that its newest events were removed. Stores make the log append-only
(the Postgres adapter documents the `REVOKE` the app applies); to also detect a cut tail, record
`last().hash` somewhere outside the log from time to time.
"""

import hashlib
import hmac
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

ActorKind = Literal["agent", "human", "system"]

GENESIS_HASH = "0" * 64
"""`prev_hash` of the first event (`seq == 1`)."""

_MIN_PAYLOAD_KEY_BYTES = 32

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class ChainConflict(Exception):
    """The event does not extend the store's current tip, usually because another append won.

    Retry by reading `last()` again and building a new event on it (`append` does both).
    """


class _EventFields(BaseModel):
    """Every field of an event except its own hash: the content that `hash` seals."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seq: Annotated[int, Field(ge=1)]
    actor_id: NonEmptyStr
    actor_kind: ActorKind
    action: NonEmptyStr
    subject: NonEmptyStr
    payload_hash: Sha256Hex
    decision: NonEmptyStr | None
    at: AwareDatetime
    prev_hash: Sha256Hex

    @field_validator("at")
    @classmethod
    def _in_utc(cls, value: datetime) -> datetime:
        # One representation per instant, so the hash does not depend on the zone the time
        # was given in or read back in (Postgres returns timestamptz in the session zone).
        return value.astimezone(UTC)

    @property
    def content_hash(self) -> str:
        """SHA-256 hex of the canonical JSON of every field but `hash`: sorted keys, no spaces."""
        fields = self.model_dump(mode="json", exclude={"hash"})
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class AuditEvent(_EventFields):
    """One audit record. `hash` equals `content_hash` unless the event was altered."""

    hash: Sha256Hex


class ChainReport(BaseModel):
    """Result of `verify_chain`: every event before `first_bad_seq` is present and intact."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ok: bool
    first_bad_seq: Annotated[int, Field(ge=1)] | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.ok != (self.first_bad_seq is None):
            raise ValueError("first_bad_seq is set exactly when the chain is not ok")
        return self


class AuditStore(Protocol):
    """Where the chain lives. A store only ever appends."""

    def last(self) -> AuditEvent | None:
        """The event with the highest `seq`, or None when the store is empty."""
        ...

    def append(self, event: AuditEvent) -> None:
        """Store `event` as the new tip.

        Raise ValueError if `event.hash` does not recompute, and `ChainConflict` unless `event`
        extends the current tip: `seq == last.seq + 1` and `prev_hash == last.hash`, or
        `seq == 1` and `prev_hash == GENESIS_HASH` on an empty store.
        """
        ...

    def iter_all(self) -> Iterator[AuditEvent]:
        """Every event in `seq` order."""
        ...


def hash_payload(payload: bytes, key: bytes) -> str:
    """HMAC-SHA256 hex of `payload` under `key`: what an event stores instead of the payload.

    Keyed, so a low-entropy payload (a phone or account number) cannot be recovered by hashing
    every candidate. `key` is an app secret of at least 32 bytes (ValueError otherwise), kept in
    the secrets manager or the Keychain and never in the repo. Rotating it means hashes made with
    the old key can no longer be recomputed, so payloads cannot be matched to old events.
    """
    if len(key) < _MIN_PAYLOAD_KEY_BYTES:
        raise ValueError(f"payload key must be at least {_MIN_PAYLOAD_KEY_BYTES} bytes")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def next_event(
    prev: AuditEvent | None,
    *,
    actor_id: str,
    actor_kind: ActorKind,
    action: str,
    subject: str,
    payload_hash: str,
    decision: str | None,
    at: datetime,
) -> AuditEvent:
    """Build the event that follows `prev` (the genesis event when `prev` is None)."""
    fields = _EventFields(
        seq=1 if prev is None else prev.seq + 1,
        actor_id=actor_id,
        actor_kind=actor_kind,
        action=action,
        subject=subject,
        payload_hash=payload_hash,
        decision=decision,
        at=at,
        prev_hash=GENESIS_HASH if prev is None else prev.hash,
    )
    return AuditEvent.model_validate(fields.model_dump() | {"hash": fields.content_hash})


def append(
    store: AuditStore,
    *,
    actor_id: str,
    actor_kind: ActorKind,
    action: str,
    subject: str,
    payload_hash: str,
    decision: str | None,
    at: datetime,
) -> AuditEvent:
    """Build the event that follows the store's tip, append it and return it.

    Raises `ChainConflict` when another append won the race; calling `append` again retries on
    the new tip (see the store's documentation for its transaction rules first).
    """
    event = next_event(
        store.last(),
        actor_id=actor_id,
        actor_kind=actor_kind,
        action=action,
        subject=subject,
        payload_hash=payload_hash,
        decision=decision,
        at=at,
    )
    store.append(event)
    return event


def verify_chain(events: Iterable[AuditEvent]) -> ChainReport:
    """Check `events` in order and report the first seq that is not verified. Never raises.

    For each position: `seq` is contiguous from 1, `prev_hash` is the previous event's `hash`
    (`GENESIS_HASH` for the first), and `hash` recomputes. `first_bad_seq` is the seq expected
    at the first failing position, so a missing or reordered event reports the seq that is
    missing there. Stops at the first failure, so it can stream a store's `iter_all()`.
    """
    expected_seq, expected_prev = 1, GENESIS_HASH
    for event in events:
        if (
            event.seq != expected_seq
            or event.prev_hash != expected_prev
            or event.hash != event.content_hash
        ):
            return ChainReport(ok=False, first_bad_seq=expected_seq)
        expected_seq, expected_prev = event.seq + 1, event.hash
    return ChainReport(ok=True, first_bad_seq=None)

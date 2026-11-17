import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from agent_toolkit.audit_log import (
    GENESIS_HASH,
    AuditEvent,
    ChainConflict,
    ChainReport,
    append,
    hash_payload,
    next_event,
    verify_chain,
)
from agent_toolkit.audit_log.testing import InMemoryAuditStore

T0 = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
PAYLOAD_KEY = bytes(range(32))  # test-only key; real keys live in a secrets manager
PAYLOAD_HASH = hash_payload(b'{"charge_id":"ch_1","amount_minor":35000}', PAYLOAD_KEY)


def _event(prev: AuditEvent | None, n: int, decision: str | None = "allow") -> AuditEvent:
    return next_event(
        prev,
        actor_id="agent-collections",
        actor_kind="agent",
        action=f"tool.call.{n}",
        subject=f"charge-{n}",
        payload_hash=PAYLOAD_HASH,
        decision=decision,
        at=T0 + timedelta(minutes=n),
    )


def _chain(length: int) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    for n in range(1, length + 1):
        events.append(_event(events[-1] if events else None, n))
    return events


# --- building events ---------------------------------------------------------------------


def test_genesis_starts_at_seq_one_with_zero_prev_hash() -> None:
    genesis = _event(None, 1)
    assert genesis.seq == 1
    assert genesis.prev_hash == GENESIS_HASH == "0" * 64


def test_next_event_links_to_previous() -> None:
    first, second = _chain(2)
    assert second.seq == 2
    assert second.prev_hash == first.hash
    assert second.hash != first.hash


def test_hash_is_sha256_of_canonical_json_without_hash() -> None:
    event = _event(None, 1)
    canonical = (
        '{"action":"tool.call.1","actor_id":"agent-collections","actor_kind":"agent",'
        '"at":"2026-11-17T10:01:00Z","decision":"allow",'
        f'"payload_hash":"{PAYLOAD_HASH}","prev_hash":"{GENESIS_HASH}","seq":1,'
        '"subject":"charge-1"}'
    )
    assert event.hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert event.content_hash == event.hash


def test_at_is_stored_in_utc_so_the_hash_ignores_the_zone() -> None:
    local = (T0 + timedelta(minutes=1)).astimezone(ZoneInfo("America/Mexico_City"))
    event = next_event(
        None,
        actor_id="agent-collections",
        actor_kind="agent",
        action="tool.call.1",
        subject="charge-1",
        payload_hash=PAYLOAD_HASH,
        decision="allow",
        at=local,
    )
    assert event.at.tzinfo is UTC
    assert event.hash == _event(None, 1).hash


def test_hash_payload_is_hmac_sha256_hex_without_the_payload() -> None:
    phone = "5512345678"
    payload = f'{{"payer_phone":"{phone}","amount_minor":35000}}'.encode()
    digest = hash_payload(payload, PAYLOAD_KEY)
    assert digest == hmac.new(PAYLOAD_KEY, payload, hashlib.sha256).hexdigest()
    assert phone not in digest
    assert digest != hashlib.sha256(payload).hexdigest()  # not recomputable without the key


def test_same_payload_with_different_keys_hashes_differently() -> None:
    payload = b'{"payer_phone":"5512345678"}'
    assert hash_payload(payload, b"a" * 32) != hash_payload(payload, b"b" * 32)


@pytest.mark.parametrize("length", [0, 16, 31])
def test_hash_payload_rejects_a_short_key(length: int) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        hash_payload(b"{}", b"k" * length)


@pytest.mark.parametrize(
    "change",
    [
        {"at": datetime(2026, 11, 17, 10, 0)},  # naive
        {"actor_kind": "robot"},
        {"payload_hash": "not-a-hash"},
        {"prev_hash": "A" * 64},  # uppercase hex is not canonical
        {"seq": 0},
        {"actor_id": ""},
        {"decision": ""},
    ],
)
def test_event_rejects_malformed_fields(change: dict[str, object]) -> None:
    fields = _event(None, 1).model_dump() | change
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(fields)


# --- verify_chain ------------------------------------------------------------------------


def test_three_event_chain_verifies() -> None:
    assert verify_chain(_chain(3)) == ChainReport(ok=True, first_bad_seq=None)


def test_empty_chain_verifies() -> None:
    assert verify_chain([]) == ChainReport(ok=True, first_bad_seq=None)


def test_altered_decision_reports_its_seq() -> None:
    first, second, third = _chain(3)
    tampered = second.model_copy(update={"decision": "deny"})
    assert verify_chain([first, tampered, third]) == ChainReport(ok=False, first_bad_seq=2)


def test_altered_event_with_recomputed_hash_breaks_the_next_link() -> None:
    first, second, third = _chain(3)
    altered = second.model_copy(update={"decision": "deny"})
    resealed = altered.model_copy(update={"hash": altered.content_hash})
    assert verify_chain([first, resealed, third]) == ChainReport(ok=False, first_bad_seq=3)


def test_reordered_chain_fails() -> None:
    first, second, third = _chain(3)
    report = verify_chain([first, third, second])
    assert report == ChainReport(ok=False, first_bad_seq=2)


def test_missing_event_reports_the_missing_seq() -> None:
    first, _, third = _chain(3)
    assert verify_chain([first, third]) == ChainReport(ok=False, first_bad_seq=2)


def test_chain_must_start_at_genesis() -> None:
    _, second, third = _chain(3)
    assert verify_chain([second, third]) == ChainReport(ok=False, first_bad_seq=1)


def test_genesis_with_another_prev_hash_fails() -> None:
    first, second = _chain(2)
    altered = first.model_copy(update={"prev_hash": "f" * 64})
    forged = altered.model_copy(update={"hash": altered.content_hash})
    assert verify_chain([forged, second]) == ChainReport(ok=False, first_bad_seq=1)


def test_chain_report_is_consistent() -> None:
    with pytest.raises(ValidationError):
        ChainReport(ok=True, first_bad_seq=2)
    with pytest.raises(ValidationError):
        ChainReport(ok=False, first_bad_seq=None)


# --- store contract (in-memory double) ---------------------------------------------------


def _append(store: InMemoryAuditStore, n: int) -> AuditEvent:
    return append(
        store,
        actor_id="agent-collections",
        actor_kind="agent",
        action=f"tool.call.{n}",
        subject=f"charge-{n}",
        payload_hash=PAYLOAD_HASH,
        decision="allow",
        at=T0 + timedelta(minutes=n),
    )


def test_append_extends_the_store_tip() -> None:
    store = InMemoryAuditStore()
    assert store.last() is None
    events = [_append(store, n) for n in range(1, 4)]
    assert events == _chain(3)
    assert store.last() == events[-1]
    assert list(store.iter_all()) == events
    assert verify_chain(store.iter_all()).ok


def test_store_rejects_an_event_built_on_a_stale_tip() -> None:
    store = InMemoryAuditStore()
    genesis = _append(store, 1)
    _append(store, 2)
    with pytest.raises(ChainConflict):
        store.append(_event(genesis, 2))
    assert [event.seq for event in store.iter_all()] == [1, 2]


def test_store_rejects_an_event_that_does_not_link_to_the_tip() -> None:
    store = InMemoryAuditStore()
    _append(store, 1)
    foreign_second = _event(_event(None, 7), 8)  # seq 2, prev_hash of another chain
    for event in (foreign_second, _event(foreign_second, 9)):
        with pytest.raises(ChainConflict):
            store.append(event)
    assert [event.seq for event in store.iter_all()] == [1]


def test_store_rejects_a_second_genesis() -> None:
    store = InMemoryAuditStore()
    _append(store, 1)
    with pytest.raises(ChainConflict):
        store.append(_event(None, 9))


def test_store_rejects_an_event_whose_hash_does_not_recompute() -> None:
    store = InMemoryAuditStore()
    forged = _event(None, 1).model_copy(update={"decision": "deny"})
    with pytest.raises(ValueError, match="hash"):
        store.append(forged)
    assert store.last() is None

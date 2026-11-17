import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from importlib import resources

import psycopg
import pytest
from psycopg import errors, sql
from psycopg.rows import TupleRow

from agent_toolkit.audit_log import (
    AuditEvent,
    ChainConflict,
    ChainReport,
    append,
    hash_payload,
    next_event,
    verify_chain,
)
from agent_toolkit.audit_log.adapters import postgres as postgres_adapter
from agent_toolkit.audit_log.adapters.postgres import PostgresAuditStore

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not DATABASE_URL,
        reason="DATABASE_URL is not set: the Postgres audit-log tests need a disposable Postgres",
    ),
]

SCHEMA_SQL = resources.files("agent_toolkit.audit_log.adapters").joinpath("schema.sql").read_text()
REVOKE = "REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM {}"

T0 = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
PAYLOAD_HASH = hash_payload(b'{"charge_id":"ch_1","amount_minor":35000}')


def _connect(schema: str, *, autocommit: bool = True) -> psycopg.Connection[TupleRow]:
    # Set at connection start, so no rollback undoes them. A non-UTC session zone checks that
    # events read back hash the same as the events written.
    options = f"-c search_path={schema} -c TimeZone=America/Mexico_City"
    return psycopg.connect(DATABASE_URL, autocommit=autocommit, options=options)


@pytest.fixture
def schema() -> Iterator[str]:
    """A fresh schema holding `audit_events`, unique per test and dropped afterwards."""
    name = f"audit_log_test_{uuid.uuid4().hex}"
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        try:
            with _connect(name) as conn:
                conn.execute(SCHEMA_SQL)
            yield name
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


def _append(store: PostgresAuditStore, n: int) -> AuditEvent:
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


def _next(prev: AuditEvent | None, actor_id: str) -> AuditEvent:
    return next_event(
        prev,
        actor_id=actor_id,
        actor_kind="agent",
        action="tool.call",
        subject="charge-1",
        payload_hash=PAYLOAD_HASH,
        decision="allow",
        at=T0,
    )


def _wait_until_blocked(pid: int) -> None:
    """Wait until backend `pid` waits on a lock held by another session."""
    deadline = time.monotonic() + 10
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        while True:
            row = admin.execute("SELECT cardinality(pg_blocking_pids(%s)) > 0", (pid,)).fetchone()
            if row is not None and row[0]:
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"backend {pid} never blocked")
            time.sleep(0.01)


def test_append_and_verify_roundtrip(schema: str) -> None:
    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        assert store.last() is None
        written = [_append(store, n) for n in range(1, 4)]

    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        assert list(store.iter_all()) == written
        assert store.last() == written[-1]
        assert verify_chain(store.iter_all()) == ChainReport(ok=True, first_bad_seq=None)


@pytest.mark.parametrize("count", [4, 5])
def test_iter_all_reads_every_page_in_order(
    schema: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(postgres_adapter, "_PAGE_SIZE", 2)
    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        written = [_append(store, n) for n in range(1, count + 1)]
        read = []
        for event in store.iter_all():
            assert store.last() == written[-1]  # the connection is free between pages
            read.append(event)
        assert read == written


def test_append_joins_the_callers_transaction(schema: str) -> None:
    with _connect(schema, autocommit=False) as conn:
        store = PostgresAuditStore(conn)
        _append(store, 1)
        conn.rollback()
        assert store.last() is None
        kept = _append(store, 1)
        conn.commit()

    with _connect(schema) as conn:
        assert PostgresAuditStore(conn).last() == kept


def test_append_on_a_tip_already_extended_conflicts(schema: str) -> None:
    with _connect(schema) as conn_a, _connect(schema) as conn_b:
        store_a, store_b = PostgresAuditStore(conn_a), PostgresAuditStore(conn_b)
        _append(store_a, 1)
        tip_a, tip_b = store_a.last(), store_b.last()
        assert tip_a == tip_b

        store_a.append(_next(tip_a, "agent-a"))
        with pytest.raises(ChainConflict):
            store_b.append(_next(tip_b, "agent-b"))

        assert [event.actor_id for event in store_b.iter_all()] == ["agent-collections", "agent-a"]


def test_append_rejects_an_event_that_does_not_link_to_the_tip(schema: str) -> None:
    # The unique constraints alone would accept these: seq 2 and 3 are free, and no stored
    # event has their prev_hash.
    foreign_genesis = _next(None, "agent-x")
    foreign_second = _next(foreign_genesis, "agent-x")
    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        genesis = _append(store, 1)
        for event in (foreign_second, _next(foreign_second, "agent-x")):
            with pytest.raises(ChainConflict):
                store.append(event)
        assert list(store.iter_all()) == [genesis]


def test_concurrent_append_conflicts(schema: str) -> None:
    """Review Focus 5: two appends on the same tip; the second fails and the chain never forks."""
    with _connect(schema) as conn:
        genesis = _append(PostgresAuditStore(conn), 1)

    with (
        _connect(schema, autocommit=False) as conn_a,
        _connect(schema, autocommit=False) as conn_b,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        store_a, store_b = PostgresAuditStore(conn_a), PostgresAuditStore(conn_b)
        tip_a, tip_b = store_a.last(), store_b.last()
        assert tip_a == tip_b == genesis
        event_a, event_b = _next(tip_a, "agent-a"), _next(tip_b, "agent-b")

        store_a.append(event_a)  # inserted, not committed yet
        pending_b = pool.submit(store_b.append, event_b)
        try:
            _wait_until_blocked(conn_b.info.backend_pid)  # B waits on A's uncommitted row
        finally:
            conn_a.commit()
        with pytest.raises(ChainConflict) as conflict:
            pending_b.result(timeout=10)
        assert isinstance(conflict.value.__cause__, errors.UniqueViolation)

        conn_b.rollback()  # the failed insert aborted B's transaction
        retried = _append(store_b, 3)  # reads the new tip
        conn_b.commit()
        assert (retried.seq, retried.prev_hash) == (3, event_a.hash)

    with _connect(schema) as conn:
        events = list(PostgresAuditStore(conn).iter_all())
    assert events == [genesis, event_a, retried]
    assert verify_chain(events).ok


def test_append_rejects_an_event_whose_hash_does_not_recompute(schema: str) -> None:
    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        forged = _next(None, "agent-a").model_copy(update={"decision": "deny"})
        with pytest.raises(ValueError, match="hash"):
            store.append(forged)
        assert store.last() is None


def test_altered_row_is_reported_by_verify_chain(schema: str) -> None:
    with _connect(schema) as conn:
        store = PostgresAuditStore(conn)
        for n in range(1, 4):
            _append(store, n)
        conn.execute("UPDATE audit_events SET decision = 'deny' WHERE seq = 2")  # as the owner
        assert verify_chain(store.iter_all()) == ChainReport(ok=False, first_bad_seq=2)


def test_app_role_cannot_update_or_delete(schema: str) -> None:
    assert f"-- {REVOKE.format('<app_role>')};" in SCHEMA_SQL.splitlines()
    role = sql.Identifier(f"audit_app_{uuid.uuid4().hex}")
    with _connect(schema) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(role))
        try:
            admin.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), role)
            )
            admin.execute(sql.SQL("GRANT SELECT, INSERT ON audit_events TO {}").format(role))
            admin.execute(sql.SQL(REVOKE).format(role))

            with _connect(schema) as app:
                app.execute(sql.SQL("SET ROLE {}").format(role))
                store = PostgresAuditStore(app)
                event = _append(store, 1)  # the app can still read and append
                assert store.last() == event
                for statement in (
                    "UPDATE audit_events SET decision = 'deny'",
                    "DELETE FROM audit_events",
                    "TRUNCATE audit_events",
                ):
                    with pytest.raises(errors.InsufficientPrivilege):
                        app.execute(statement)
                assert list(store.iter_all()) == [event]
        finally:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(role))
            admin.execute(sql.SQL("DROP ROLE {}").format(role))

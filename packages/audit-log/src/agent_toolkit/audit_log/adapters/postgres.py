"""Postgres store for the audit log, on a psycopg 3 connection the caller owns.

Install with the `postgres` extra and create the table with `schema.sql` (shipped next to this
module) from a migration. Queries name `audit_events` unqualified, so the connection's
`search_path` picks the schema.

Transactions: the store never commits or rolls back. Call `append` inside the same transaction
as the effect it records (the gate decision, the charge row, the approval), so both commit or
neither does.

Concurrency: `append` is one `INSERT ... SELECT` that inserts only when the event extends the
current tip (`seq == last.seq + 1` and `prev_hash == last.hash`). When two appends build on the
same tip and the first is not committed yet, the second insert waits for it and, once it commits,
hits the unique `seq` or `prev_hash` constraint; on a tip that is already stale it inserts
nothing. Both raise `ChainConflict`, so the chain never forks. The unique violation also aborts
the caller's open transaction, so after any `ChainConflict` roll back, then call `append` again,
which re-reads `last()`. To keep the rest of a larger transaction instead, wrap the call in a
savepoint (`with conn.transaction():` while that transaction is open) and retry inside it; that
retry sees the new tip only under READ COMMITTED (the default isolation level).

Least privilege: run the app as a role that does not own the table, with only `SELECT` and
`INSERT` on it, and apply the `REVOKE` in `schema.sql`.
"""

from collections.abc import Iterator

import psycopg
from psycopg import errors, sql
from psycopg.rows import class_row

from agent_toolkit.audit_log.core import GENESIS_HASH, AuditEvent, ChainConflict

_PAGE_SIZE = 1000

_FIELDS = tuple(AuditEvent.model_fields)
_COLUMNS = sql.SQL(", ").join(map(sql.Identifier, _FIELDS))

# Inserts nothing unless the event extends the tip; the unique seq and prev_hash constraints
# catch a concurrent append that read the same tip.
_INSERT = sql.SQL(
    """
    INSERT INTO audit_events ({columns})
    SELECT {values}
    WHERE coalesce((SELECT max(seq) FROM audit_events), 0) = %(seq)s - 1
      AND coalesce((SELECT hash FROM audit_events WHERE seq = %(seq)s - 1), %(genesis)s)
          = %(prev_hash)s
    """
).format(columns=_COLUMNS, values=sql.SQL(", ").join(map(sql.Placeholder, _FIELDS)))

_LAST = sql.SQL("SELECT {} FROM audit_events ORDER BY seq DESC LIMIT 1").format(_COLUMNS)

_PAGE = sql.SQL("SELECT {} FROM audit_events WHERE seq > %s ORDER BY seq LIMIT %s").format(_COLUMNS)


class PostgresAuditStore:
    """`AuditStore` on the `audit_events` table, through `conn` (any row factory)."""

    def __init__(self, conn: psycopg.Connection[object]) -> None:
        self._conn = conn

    def last(self) -> AuditEvent | None:
        with self._conn.cursor(row_factory=class_row(AuditEvent)) as cur:
            return cur.execute(_LAST).fetchone()

    def append(self, event: AuditEvent) -> None:
        if event.hash != event.content_hash:
            raise ValueError(f"event {event.seq} hash does not match its content")
        params = event.model_dump() | {"genesis": GENESIS_HASH}
        try:
            with self._conn.cursor() as cur:
                inserted = cur.execute(_INSERT, params).rowcount
        except errors.UniqueViolation as exc:
            constraint = exc.diag.constraint_name
            raise ChainConflict(
                f"event {event.seq} lost a concurrent append ({constraint})"
            ) from exc
        if inserted != 1:
            raise ChainConflict(f"event {event.seq} does not extend the current tip")

    def iter_all(self) -> Iterator[AuditEvent]:
        """Every event in `seq` order, read in pages so the log never sits in memory at once.

        The connection is free between pages, so the caller may use it while iterating.
        """
        after = 0
        while True:
            with self._conn.cursor(row_factory=class_row(AuditEvent)) as cur:
                page = cur.execute(_PAGE, (after, _PAGE_SIZE)).fetchall()
            yield from page
            if len(page) < _PAGE_SIZE:
                return
            after = page[-1].seq

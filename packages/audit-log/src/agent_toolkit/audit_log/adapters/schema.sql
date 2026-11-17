-- agent-toolkit audit-log: append-only, hash-chained audit events (PostgresAuditStore).
-- Apply once, from a migration, in the schema the app's search_path resolves.
--
-- seq and prev_hash are unique, so two appends built on the same tip cannot both land and the
-- chain cannot fork. Columns hold hashes and references only, never PII in clear.
CREATE TABLE audit_events (
    seq          bigint      PRIMARY KEY CHECK (seq >= 1),
    actor_id     text        NOT NULL CHECK (actor_id <> ''),
    actor_kind   text        NOT NULL CHECK (actor_kind IN ('agent', 'human', 'system')),
    action       text        NOT NULL CHECK (action <> ''),
    subject      text        NOT NULL CHECK (subject <> ''),
    payload_hash text        NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    decision     text                 CHECK (decision <> ''),
    at           timestamptz NOT NULL,
    prev_hash    text        NOT NULL UNIQUE CHECK (prev_hash ~ '^[0-9a-f]{64}$'),
    hash         text        NOT NULL UNIQUE CHECK (hash ~ '^[0-9a-f]{64}$')
);

-- The app reads and appends, nothing else. Run it as a role that does not own this table,
-- grant that role SELECT and INSERT, and apply (with the role's name):
-- REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM <app_role>;

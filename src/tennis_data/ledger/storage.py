"""SQLite storage with database-level append-only protection.

Durability: WAL journal, ``synchronous=FULL`` (every commit is fsynced), ``foreign_keys=ON``. Writers use
``BEGIN IMMEDIATE`` so only one transaction at a time can read the chain head and append; others wait up to
``busy_timeout``. A transaction either commits a whole event or nothing (SQLite atomic commit), so a crash cannot
leave a half-written event; on reopen SQLite rolls back or replays the WAL automatically.

Database-level protection (ordinary SQL, not a security boundary against a privileged user):

- ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers on ``events``, ``ledger_meta`` and ``idempotency_conflicts`` abort;
- a ``BEFORE INSERT`` trigger rejects any event whose sequence is not ``max(seq) + 1`` or whose ``previous_hash`` is
  not the current head hash (or the genesis hash for the first event);
- CHECK constraints on event types, schema version, timestamps, hash formats and probability ranges; UNIQUE event ids,
  sequences, hashes and prediction idempotency keys; references to earlier events are foreign keys.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = "ledger-1"
EVENT_TYPES = (
    "PREDICTION_COMMITTED", "MATCH_METADATA_CORRECTED", "MATCH_CANCELLED", "MATCH_POSTPONED",
    "MATCH_RESULT_RECORDED", "PREDICTION_VOIDED", "LEDGER_CHECKPOINT_CREATED",
)  # fmt: skip
MODES = ("SYNTHETIC", "HISTORICAL_REPLAY", "PROSPECTIVE")

_types = ", ".join(f"'{t}'" for t in EVENT_TYPES)
_modes = ", ".join(f"'{m}'" for m in MODES)
DDL = f"""
CREATE TABLE IF NOT EXISTS ledger_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    ledger_id TEXT NOT NULL CHECK (length(ledger_id) BETWEEN 1 AND 128),
    schema_version TEXT NOT NULL CHECK (schema_version = '{SCHEMA_VERSION}'),
    genesis_hash TEXT NOT NULL CHECK (length(genesis_hash) = 64),
    created_at TEXT NOT NULL CHECK (created_at GLOB '????-??-??T??:??:??.??????Z')
) STRICT;

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY CHECK (seq >= 1),
    event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) BETWEEN 8 AND 128),
    event_type TEXT NOT NULL CHECK (event_type IN ({_types})),
    schema_version TEXT NOT NULL CHECK (schema_version = '{SCHEMA_VERSION}'),
    recorded_at TEXT NOT NULL CHECK (recorded_at GLOB '????-??-??T??:??:??.??????Z'),
    match_id TEXT,
    mode TEXT CHECK (mode IS NULL OR mode IN ({_modes})),
    idempotency_key TEXT UNIQUE CHECK (idempotency_key IS NULL OR length(idempotency_key) = 64),
    probability_a REAL CHECK (probability_a IS NULL OR (probability_a >= 0.0 AND probability_a <= 1.0)),
    probability_b REAL CHECK (probability_b IS NULL OR (probability_b >= 0.0 AND probability_b <= 1.0)),
    ref_event_id TEXT REFERENCES events(event_id),
    payload TEXT NOT NULL CHECK (json_valid(payload)),
    refs TEXT NOT NULL CHECK (json_valid(refs)),
    previous_hash TEXT NOT NULL CHECK (length(previous_hash) = 64),
    event_hash TEXT NOT NULL UNIQUE CHECK (length(event_hash) = 64),
    CHECK ((event_type = 'PREDICTION_COMMITTED') = (idempotency_key IS NOT NULL)),
    CHECK ((event_type = 'PREDICTION_COMMITTED') = (mode IS NOT NULL)),
    CHECK ((event_type = 'PREDICTION_COMMITTED') = (probability_a IS NOT NULL AND probability_b IS NOT NULL))
) STRICT;

CREATE INDEX IF NOT EXISTS events_match ON events(match_id, seq);

CREATE TABLE IF NOT EXISTS idempotency_conflicts (
    id INTEGER PRIMARY KEY,
    detected_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    existing_event_id TEXT NOT NULL REFERENCES events(event_id),
    existing_content_hash TEXT NOT NULL,
    rejected_content_hash TEXT NOT NULL,
    rejected_request TEXT NOT NULL CHECK (json_valid(rejected_request))
) STRICT;

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'ledger events are append-only: UPDATE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'ledger events are append-only: DELETE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS meta_no_update BEFORE UPDATE ON ledger_meta
BEGIN SELECT RAISE(ABORT, 'ledger metadata is immutable'); END;
CREATE TRIGGER IF NOT EXISTS meta_no_delete BEFORE DELETE ON ledger_meta
BEGIN SELECT RAISE(ABORT, 'ledger metadata is immutable'); END;
CREATE TRIGGER IF NOT EXISTS conflicts_no_update BEFORE UPDATE ON idempotency_conflicts
BEGIN SELECT RAISE(ABORT, 'conflict log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS conflicts_no_delete BEFORE DELETE ON idempotency_conflicts
BEGIN SELECT RAISE(ABORT, 'conflict log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS events_chain BEFORE INSERT ON events
BEGIN
    SELECT CASE
        WHEN NEW.seq != COALESCE((SELECT max(seq) FROM events), 0) + 1
        THEN RAISE(ABORT, 'ledger sequence must be the next integer')
        WHEN NEW.previous_hash != COALESCE((SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1),
                                           (SELECT genesis_hash FROM ledger_meta))
        THEN RAISE(ABORT, 'previous_hash must equal the current chain head')
    END;
END;
"""

COLUMNS = ("seq", "event_id", "event_type", "schema_version", "recorded_at", "match_id", "mode", "idempotency_key",
           "probability_a", "probability_b", "ref_event_id", "payload", "refs", "previous_hash", "event_hash")  # fmt: skip


class LedgerStorageError(RuntimeError):
    pass


def connect(path: Path | str, busy_timeout_ms: int = 30000, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None,
                              timeout=busy_timeout_ms / 1000)  # fmt: skip
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, isolation_level=None, timeout=busy_timeout_ms / 1000)
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    con.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = FULL")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(DDL)


def meta(con: sqlite3.Connection) -> dict:
    row = con.execute("SELECT * FROM ledger_meta").fetchone()
    if row is None:
        raise LedgerStorageError("ledger is not initialized")
    return dict(row)


def head(con: sqlite3.Connection) -> tuple[int, str]:
    row = con.execute("SELECT seq, event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    if row is None:
        return 0, meta(con)["genesis_hash"]
    return row["seq"], row["event_hash"]


def all_events(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute(f"SELECT {', '.join(COLUMNS)} FROM events ORDER BY seq")]

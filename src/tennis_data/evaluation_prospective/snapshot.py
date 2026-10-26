"""Content-addressed, read-only ledger snapshots.

A snapshot is the Phase 4B ``events.jsonl`` export (``ledger.export``), identified by the SHA-256 of its bytes. The
evaluation never opens the SQLite ledger for writing: ``snapshot_from_ledger`` exports through a read-only connection
into ``<snapshot_dir>/<sha256[:16]>/`` and everything after that reads the file only. Loading is strict: every line
must already be canonical JSON (as the exporter writes it), keys must be exactly the stored event columns, and size
limits apply. ``assert_unchanged`` re-hashes the file so a snapshot modified during an evaluation is detected.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tennis_data.evaluation_prospective.config import DEFAULT_LIMITS, Limits
from tennis_data.evaluation_prospective.safeio import UntrustedInputError, read_bounded, strict_json
from tennis_data.ledger import storage
from tennis_data.ledger.export import HEADER, export
from tennis_data.ledger.hashing import CanonicalizationError, canonical

META_KEYS = {"singleton", "ledger_id", "schema_version", "genesis_hash", "created_at"}


class SnapshotChanged(RuntimeError):
    """An input snapshot changed while it was being evaluated."""


@dataclass(frozen=True)
class LedgerSnapshot:
    path: Path
    sha256: str
    meta: dict
    events: tuple  # stored form: payload and refs as canonical JSON text, exactly as in the database

    @property
    def ledger_id(self) -> str:
        return self.meta["ledger_id"]

    @property
    def head_seq(self) -> int:
        return self.events[-1]["seq"] if self.events else 0

    @property
    def head_hash(self) -> str:
        return self.events[-1]["event_hash"] if self.events else self.meta["genesis_hash"]

    def identity(self) -> dict:
        return {"ledger_id": self.ledger_id, "head_seq": self.head_seq, "head_hash": self.head_hash,
                "snapshot_sha256": self.sha256, "events": len(self.events)}  # fmt: skip


def parse_export(data: bytes, limits: Limits = DEFAULT_LIMITS) -> tuple[dict, list[dict]]:
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    if not lines:
        raise UntrustedInputError("empty ledger export")
    out = []
    for i, raw in enumerate(lines):
        if len(raw) > limits.max_line_bytes:
            raise UntrustedInputError(f"line {i + 1}: longer than {limits.max_line_bytes} bytes")
        obj = strict_json(raw)
        if not isinstance(obj, dict):
            raise UntrustedInputError(f"line {i + 1}: not a JSON object")
        try:
            if canonical(obj) != raw:
                raise UntrustedInputError(f"line {i + 1}: not canonical JSON (export altered or re-serialized)")
        except CanonicalizationError as e:
            raise UntrustedInputError(f"line {i + 1}: {e}") from e
        out.append(obj)
    head = out[0]
    if set(head) != {"format", "meta"} or head["format"] != HEADER or not isinstance(head["meta"], dict):
        raise UntrustedInputError("not a tennis-ledger-export-1 file")
    meta = head["meta"]
    if not {"ledger_id", "schema_version", "genesis_hash", "created_at"} <= set(meta) <= META_KEYS:
        raise UntrustedInputError("ledger metadata has unexpected fields")
    events = []
    for i, e in enumerate(out[1:], start=2):
        if set(e) != set(storage.COLUMNS):
            raise UntrustedInputError(f"line {i}: event fields differ from the ledger-1 columns")
        if not isinstance(e["payload"], dict) or not isinstance(e["refs"], dict):
            raise UntrustedInputError(f"line {i}: payload and refs must be objects")
        if not isinstance(e["seq"], int) or isinstance(e["seq"], bool):
            raise UntrustedInputError(f"line {i}: seq must be an integer")
        for k in ("event_id", "event_type", "schema_version", "recorded_at", "previous_hash", "event_hash"):
            if not isinstance(e[k], str):
                raise UntrustedInputError(f"line {i}: {k} must be a string")
        events.append({**e, "payload": canonical(e["payload"]).decode("utf-8"),
                       "refs": canonical(e["refs"]).decode("utf-8")})  # fmt: skip
    return meta, events


def load_export(path: Path, limits: Limits = DEFAULT_LIMITS) -> LedgerSnapshot:
    data = read_bounded(path, limits.max_export_bytes)
    meta, events = parse_export(data, limits)
    return LedgerSnapshot(Path(path), hashlib.sha256(data).hexdigest(), meta, tuple(events))


def assert_unchanged(snap: LedgerSnapshot) -> None:
    h = hashlib.sha256(Path(snap.path).read_bytes()).hexdigest()
    if h != snap.sha256:
        raise SnapshotChanged(f"{snap.path} changed during evaluation ({snap.sha256[:12]} -> {h[:12]})")


def snapshot_from_ledger(db_path: Path, snapshot_dir: Path, limits: Limits = DEFAULT_LIMITS) -> LedgerSnapshot:
    """Export the ledger through a read-only connection into a content-addressed directory and load it back."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"ledger not found: {db_path}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=snapshot_dir))
    try:
        export(db_path, tmp)
        h = hashlib.sha256((tmp / "events.jsonl").read_bytes()).hexdigest()
        final = snapshot_dir / h[:16]
        if final.exists():
            existing = load_export(final / "events.jsonl", limits)
            if existing.sha256 != h:
                raise SnapshotChanged(f"snapshot directory {final} holds different content")
            return existing
        tmp.rename(final)
        tmp = None
        return load_export(final / "events.jsonl", limits)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

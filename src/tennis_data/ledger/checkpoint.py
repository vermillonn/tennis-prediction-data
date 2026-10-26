"""Checkpoints: a small, deterministic statement of the chain head that can be stored outside the ledger.

    checkpoint_hash = SHA256("tennis-ledger/checkpoint/v1\\0" + canonical({ledger_id, schema_version, sequence,
                                                                           head_hash, previous_checkpoint_hash}))

The export time is operational metadata kept outside the hashed body. A local hash chain is tamper-evident only
relative to a head that was recorded somewhere the local writer cannot change. Publishing checkpoints to an
independent system (a second repository, a timestamping service, a signed e-mail) makes later history rewriting
detectable; Phase 4B implements export and verification only and no external publication has been configured.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tennis_data.ledger import storage
from tennis_data.ledger.hashing import CHECKPOINT_TAG, canonical, sha256_hex, ts
from tennis_data.ledger.verification import verify_events

FORMAT = "tennis-ledger-checkpoint-1"


def make_checkpoint(con, previous: dict | None = None) -> dict:
    m = storage.meta(con)
    seq, head = storage.head(con)
    body = {"format": FORMAT, "ledger_id": m["ledger_id"], "schema_version": storage.SCHEMA_VERSION, "sequence": seq,
            "head_hash": head, "previous_checkpoint_hash": previous["checkpoint_hash"] if previous else None}  # fmt: skip
    return {**body, "checkpoint_hash": checkpoint_hash(body)}


def checkpoint_hash(body: dict) -> str:
    keys = ("format", "ledger_id", "schema_version", "sequence", "head_hash", "previous_checkpoint_hash")
    return sha256_hex(CHECKPOINT_TAG + canonical({k: body[k] for k in keys}))


def write_checkpoint(cp: dict, path: Path, exported_at: datetime | None = None) -> Path:
    """Deterministic checkpoint file plus a separate ``.meta.json`` holding the operational export time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(cp) + b"\n")
    meta = {"checkpoint_hash": cp["checkpoint_hash"], "exported_at": ts(exported_at or datetime.now(UTC))}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, sort_keys=True) + "\n")
    return path


def verify_checkpoint(path_or_con, cp: dict, tol: float = 1e-12) -> dict:
    """Does the local ledger contain, unchanged, the history the checkpoint commits to?"""
    problems = []
    if cp.get("format") != FORMAT:
        problems.append("unknown checkpoint format")
    if checkpoint_hash(cp) != cp.get("checkpoint_hash"):
        problems.append("checkpoint_hash does not match its body (checkpoint altered)")
    con = storage.connect(path_or_con, read_only=True) if not hasattr(path_or_con, "execute") else path_or_con
    m = storage.meta(con)
    events = storage.all_events(con)
    if m["ledger_id"] != cp.get("ledger_id"):
        problems.append("ledger_id differs")
    seq = cp.get("sequence", -1)
    if seq > len(events):
        problems.append(f"ledger has {len(events)} events, fewer than the checkpoint's {seq} (history truncated)")
    else:
        rep = verify_events(m, events[:seq], tol) if seq else None
        head = rep.head_hash if rep else m["genesis_hash"]
        if rep is not None and not rep.ok:
            problems.append(f"chain up to the checkpoint is invalid: {rep.issues[:3]}")
        if head != cp.get("head_hash"):
            problems.append("head hash at the checkpoint sequence differs (history rewritten)")
    full = verify_events(m, events, tol)
    if not full.ok:
        problems.append(f"chain after the checkpoint is invalid: {full.issues[:3]}")
    return {"ok": not problems, "problems": problems, "checkpoint_sequence": seq, "ledger_events": len(events),
            "ledger_head_hash": full.head_hash}  # fmt: skip

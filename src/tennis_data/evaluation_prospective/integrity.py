"""Integrity of a ledger snapshot: the Phase 4B verifier, checkpoint consistency and in-chain checkpoint events.

Reuses ``ledger.verification.verify_events`` (hash chain, payloads, lifecycle) and ``ledger.checkpoint.checkpoint_hash``.
A snapshot with any integrity issue is INVALID as a whole: the evaluation then scores nothing (fail closed).
"""

from __future__ import annotations

import json

from tennis_data.evaluation_prospective.snapshot import LedgerSnapshot
from tennis_data.ledger.checkpoint import FORMAT as CHECKPOINT_FORMAT
from tennis_data.ledger.checkpoint import checkpoint_hash
from tennis_data.ledger.verification import verify_events

CHECKPOINT_KEYS = {"format", "ledger_id", "schema_version", "sequence", "head_hash", "previous_checkpoint_hash",
                   "checkpoint_hash"}  # fmt: skip


def verify_snapshot(snap: LedgerSnapshot, tol: float = 1e-12) -> dict:
    rep = verify_events(snap.meta, list(snap.events), tol).as_dict()
    rep["in_chain_checkpoints"] = in_chain_checkpoints(snap)
    bad_cp = [c for c in rep["in_chain_checkpoints"] if not c["ok"]]
    if bad_cp:
        rep["ok"] = False
        rep["issues"].append({"seq": bad_cp[0]["seq"], "code": "in_chain_checkpoint_mismatch",
                              "detail": bad_cp[0]["problem"]})  # fmt: skip
    rep["snapshot_head_hash"] = snap.head_hash
    if rep["ok"] and rep["head_hash"] != snap.head_hash:
        rep["ok"] = False
        rep["issues"].append({"seq": snap.head_seq, "code": "head_hash_mismatch", "detail": ""})
    return rep


def in_chain_checkpoints(snap: LedgerSnapshot) -> list[dict]:
    """``LEDGER_CHECKPOINT_CREATED`` events must name the true head hash at their covered sequence."""
    out = []
    for e in snap.events:
        if e["event_type"] != "LEDGER_CHECKPOINT_CREATED":
            continue
        p = json.loads(e["payload"])
        seq = p.get("covered_seq")
        problem = None
        if not isinstance(seq, int) or seq < 0 or seq >= e["seq"]:
            problem = "covered_seq must precede the checkpoint event"
        else:
            head = snap.events[seq - 1]["event_hash"] if seq else snap.meta["genesis_hash"]
            if head != p.get("covered_head_hash"):
                problem = "covered_head_hash differs from the chain"
        out.append({"seq": e["seq"], "covered_seq": seq, "checkpoint_hash": p.get("checkpoint_hash"),
                    "ok": problem is None, "problem": problem})  # fmt: skip
    return out


def check_checkpoint(cp: dict, snap: LedgerSnapshot) -> list[str]:
    """Problems of a checkpoint body relative to the snapshot (empty list: consistent).

    A checkpoint beyond the snapshot head means the snapshot is missing history that was already committed
    (truncated export, or an older snapshot presented as current): a problem.
    """
    if not isinstance(cp, dict) or set(cp) != CHECKPOINT_KEYS:
        return ["checkpoint fields differ from tennis-ledger-checkpoint-1"]
    problems = []
    if cp["format"] != CHECKPOINT_FORMAT:
        problems.append("unknown checkpoint format")
    if checkpoint_hash(cp) != cp["checkpoint_hash"]:
        problems.append("checkpoint_hash does not match its body (checkpoint altered)")
    if cp["ledger_id"] != snap.ledger_id:
        problems.append("ledger_id differs from the snapshot")
    seq = cp["sequence"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        problems.append("invalid checkpoint sequence")
    elif seq > snap.head_seq:
        problems.append(f"snapshot ends at {snap.head_seq}, before checkpoint sequence {seq} (history truncated)")
    else:
        head = snap.events[seq - 1]["event_hash"] if seq else snap.meta["genesis_hash"]
        if head != cp["head_hash"]:
            problems.append(f"head hash at sequence {seq} differs from the snapshot (history rewritten or forked)")
    return problems

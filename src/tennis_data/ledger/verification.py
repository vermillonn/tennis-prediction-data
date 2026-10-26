"""Independent integrity verification of a ledger (database or exported event list).

Detects: modified payloads or fields (hash mismatch), non-canonical stored payloads, deleted or reordered events
(sequence gaps, predecessor mismatches), wrong predecessor hashes, invalid event hashes, duplicate event ids or hashes,
duplicate idempotency keys, inconsistent denormalized columns, invalid payloads, commit-time regressions and invalid
lifecycle transitions. The result names the verified chain head, which can be compared with a trusted checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tennis_data.ledger import storage
from tennis_data.ledger.events import EventValidationError, validate_payload
from tennis_data.ledger.hashing import canonical, event_hash, genesis_hash, parse_ts
from tennis_data.ledger.lifecycle import TransitionError, replay


@dataclass
class VerificationReport:
    ok: bool
    ledger_id: str
    events: int
    head_seq: int
    head_hash: str
    issues: list = field(default_factory=list)

    def as_dict(self):
        return {"ok": self.ok, "ledger_id": self.ledger_id, "events": self.events, "head_seq": self.head_seq,
                "head_hash": self.head_hash, "issues": self.issues}  # fmt: skip


def body_of(e: dict) -> dict:
    payload = e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])
    refs = e["refs"] if isinstance(e["refs"], dict) else json.loads(e["refs"])
    return {k: e[k] for k in ("seq", "event_id", "event_type", "schema_version", "recorded_at", "match_id", "mode",
                              "idempotency_key", "probability_a", "probability_b", "ref_event_id", "previous_hash")} | {
        "payload": payload, "refs": refs}  # fmt: skip


def verify_events(meta: dict, events: list[dict], tol: float = 1e-12, max_issues: int = 1000) -> VerificationReport:
    issues: list[dict] = []

    def bad(seq, code, detail=""):
        if len(issues) < max_issues:
            issues.append({"seq": seq, "code": code, "detail": str(detail)})

    gh = genesis_hash(meta["ledger_id"])
    if meta.get("genesis_hash") != gh:
        bad(0, "genesis_hash_mismatch")
    if meta.get("schema_version") != storage.SCHEMA_VERSION:
        bad(0, "schema_version_mismatch", meta.get("schema_version"))
    prev, seen_ids, seen_hashes, seen_keys, last_t = gh, set(), set(), set(), None
    parsed = []
    for i, e in enumerate(events):
        seq = e["seq"]
        if seq != i + 1:
            bad(seq, "sequence_gap_or_reorder", f"expected {i + 1}")
        if e["previous_hash"] != prev:
            bad(seq, "previous_hash_mismatch")
        try:
            body = body_of(e)
            if isinstance(e["payload"], str) and canonical(body["payload"]).decode("utf-8") != e["payload"]:
                bad(seq, "payload_not_canonical")
            if event_hash(body, e["previous_hash"]) != e["event_hash"]:
                bad(seq, "event_hash_mismatch")
        except (ValueError, TypeError) as ex:
            bad(seq, "unreadable_event", ex)
            prev = e["event_hash"]
            continue
        if e["event_id"] in seen_ids:
            bad(seq, "duplicate_event_id", e["event_id"])
        if e["event_hash"] in seen_hashes:
            bad(seq, "duplicate_event_hash")
        seen_ids.add(e["event_id"])
        seen_hashes.add(e["event_hash"])
        if e["schema_version"] != storage.SCHEMA_VERSION:
            bad(seq, "schema_version_mismatch", e["schema_version"])
        if e["event_type"] not in storage.EVENT_TYPES:
            bad(seq, "unknown_event_type", e["event_type"])
        p = body["payload"]
        try:
            validate_payload(e["event_type"], p, tol)
        except (EventValidationError, KeyError, TypeError) as ex:
            bad(seq, "invalid_payload", ex)
        if e["event_type"] == "PREDICTION_COMMITTED":
            k = e["idempotency_key"]
            if k in seen_keys:
                bad(seq, "duplicate_idempotency_key")
            seen_keys.add(k)
            a, pr = p.get("audit", {}), p.get("prediction", {})
            if a.get("idempotency_key") != k or e["event_id"] != f"pred-{(k or '')[:32]}":
                bad(seq, "idempotency_key_mismatch")
            if (e["probability_a"], e["probability_b"], e["mode"]) != (pr.get("probability_a"), pr.get("probability_b"),
                                                                       a.get("mode")):  # fmt: skip
                bad(seq, "denormalized_column_mismatch")
        try:
            t = parse_ts(e["recorded_at"])
            if last_t is not None and t < last_t:
                bad(seq, "commit_time_regression")
            last_t = t
        except ValueError as ex:
            bad(seq, "invalid_timestamp", ex)
        parsed.append({**e, "payload": p})
        prev = e["event_hash"]
    try:
        replay(parsed)
    except (TransitionError, KeyError, TypeError) as ex:
        bad(None, "invalid_lifecycle_transition", ex)
    head_seq = events[-1]["seq"] if events else 0
    return VerificationReport(not issues, meta["ledger_id"], len(events), head_seq, prev, issues)


def verify_ledger(path, tol: float = 1e-12) -> VerificationReport:
    con = storage.connect(path, read_only=True)
    try:
        return verify_events(storage.meta(con), storage.all_events(con), tol)
    finally:
        con.close()

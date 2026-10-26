"""Match outcomes and actual-start evidence, reconciled from the ledger and an outcome snapshot (policy ``unanimous``).

Inputs

- the ledger's own lifecycle (``ledger.lifecycle.replay``): current result after corrections, cancellations, player
  changes, and every sourced ``actual_start`` metadata event;
- an *outcome snapshot* (``tennis-outcome-snapshot-1`` JSONL): validated final match records with provenance, one
  line per provider record version. It is hashed and treated as untrusted input.

Policy (the same principles as Phase 4A ``rec-1``):

1. Per (provider, provider_record_id) the highest ``source_version`` is current; two different contents under one
   version are a conflict (``version_reissued``) and make the match DISPUTED.
2. Every current assertion (the ledger's current result and each provider's current record) must agree on status
   and winner. Any disagreement makes the match DISPUTED: no source is silently preferred. A dispute is resolved
   only by a newer source version or a ledger result correction, which produces a new evaluation snapshot.
3. A score whose own status marker (Phase 1 ``match_status``: RET, W/O, DEF, ABD) contradicts the stated status is a
   dispute. The provider's player pair must equal the predicted pair and a winner must be one of them, otherwise
   IDENTITY_MISMATCH.
4. Statuses are kept apart: COMPLETED, RETIRED, WALKOVER, DEFAULT, ABANDONED and UNKNOWN are different outcomes; a
   walkover, default or abandonment is never turned into an ordinary win.
5. The actual start is the *earliest* sourced actual start (conservative for pre-match checks); disagreement between
   sources is flagged.

Records retrieved, and ledger events recorded, after the evaluation cutoff are ignored (not yet known).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tennis_data.evaluation_prospective.config import DEFAULT_LIMITS, Limits
from tennis_data.evaluation_prospective.safeio import UntrustedInputError, read_bounded, strict_json
from tennis_data.ledger.events import RESULT_STATUSES, WINNER_REQUIRED
from tennis_data.ledger.hashing import CanonicalizationError, parse_ts
from tennis_data.ledger.lifecycle import MatchState
from tennis_data.normalization.score import match_status

FORMAT = "tennis-outcome-snapshot-1"
HEADER_KEYS = {"format", "snapshot_id", "created_at", "description"}
RECORD_KEYS = {"match_id", "provider", "provider_record_id", "source_version", "status", "winner_id", "player_ids",
               "score", "actual_start_at", "tournament_id", "tournament_level", "retrieved_at"}  # fmt: skip
OUTCOMES = {"completed": "COMPLETED", "retired": "RETIRED", "walkover": "WALKOVER", "default": "DEFAULT",
            "abandoned": "ABANDONED", "unknown": "UNKNOWN"}  # fmt: skip
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class OutcomeSnapshot:
    path: Path | None
    sha256: str
    header: dict
    records: tuple

    def identity(self) -> dict:
        return {"outcome_snapshot_sha256": self.sha256, "snapshot_id": self.header.get("snapshot_id"),
                "records": len(self.records)}  # fmt: skip


def empty_outcomes() -> OutcomeSnapshot:
    return OutcomeSnapshot(None, EMPTY_SHA, {"format": FORMAT, "snapshot_id": "none", "created_at": None,
                                             "description": "no outcome snapshot supplied"}, ())  # fmt: skip


def _check_record(r, i):
    def bad(msg):
        raise UntrustedInputError(f"outcome line {i}: {msg}")

    if not isinstance(r, dict) or set(r) != RECORD_KEYS:
        bad("fields differ from tennis-outcome-snapshot-1")
    for k in ("match_id", "provider", "provider_record_id"):
        if not isinstance(r[k], str) or not r[k] or len(r[k]) > 256:
            bad(f"{k} must be a non-empty string")
    if not isinstance(r["source_version"], int) or isinstance(r["source_version"], bool) or r["source_version"] < 0:
        bad("source_version must be a non-negative integer")
    if r["status"] not in RESULT_STATUSES:
        bad(f"status must be one of {RESULT_STATUSES}")
    if r["winner_id"] is not None and (not isinstance(r["winner_id"], int) or isinstance(r["winner_id"], bool)):
        bad("winner_id must be an integer or null")
    p = r["player_ids"]
    if not isinstance(p, list) or len(p) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in p):
        bad("player_ids must be two integers")
    for k in ("score", "tournament_id", "tournament_level"):
        if r[k] is not None and (not isinstance(r[k], str) or len(r[k]) > 256):
            bad(f"{k} must be a short string or null")
    try:
        parse_ts(r["retrieved_at"])
        if r["actual_start_at"] is not None:
            parse_ts(r["actual_start_at"])
    except CanonicalizationError as e:
        bad(str(e))


def load_outcomes(path: Path, limits: Limits = DEFAULT_LIMITS) -> OutcomeSnapshot:
    data = read_bounded(path, limits.max_export_bytes)
    lines = [x for x in data.split(b"\n") if x.strip()]
    if not lines:
        raise UntrustedInputError("empty outcome snapshot")
    if len(lines) - 1 > limits.max_outcome_records:
        raise UntrustedInputError("too many outcome records")
    head = strict_json(lines[0])
    if not isinstance(head, dict) or set(head) != HEADER_KEYS or head["format"] != FORMAT:
        raise UntrustedInputError("not a tennis-outcome-snapshot-1 file")
    recs = []
    for i, raw in enumerate(lines[1:], start=2):
        if len(raw) > limits.max_line_bytes:
            raise UntrustedInputError(f"outcome line {i}: too long")
        r = strict_json(raw)
        _check_record(r, i)
        recs.append(r)
    recs.sort(key=lambda r: (r["match_id"], r["provider"], r["provider_record_id"], r["source_version"],
                             json.dumps(r, sort_keys=True)))  # fmt: skip
    return OutcomeSnapshot(Path(path), hashlib.sha256(data).hexdigest(), head, tuple(recs))


def assert_outcomes_unchanged(o: OutcomeSnapshot) -> None:
    from tennis_data.evaluation_prospective.snapshot import SnapshotChanged

    if o.path is not None and hashlib.sha256(Path(o.path).read_bytes()).hexdigest() != o.sha256:
        raise SnapshotChanged(f"{o.path} changed during evaluation")


def current_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Latest version per provider record, plus conflict notes for reissued versions."""
    by: dict = {}
    for r in records:
        by.setdefault((r["provider"], r["provider_record_id"]), []).append(r)
    cur, notes = [], []
    for key, vs in sorted(by.items()):
        top = max(v["source_version"] for v in vs)
        tops = {json.dumps(v, sort_keys=True) for v in vs if v["source_version"] == top}
        if len(tops) > 1:
            notes.append(f"version_reissued:{key[0]}:{key[1]}:{top}")
        cur.append(next(v for v in vs if v["source_version"] == top))
    return cur, notes


def _actual_starts(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if e["event_type"] == "MATCH_METADATA_CORRECTED" and e["payload"].get("kind") == "actual_start":
            out.append({"origin": "ledger", "event_id": e["event_id"],
                        "provider": e["payload"]["source"].get("provider"),
                        "actual_start_at": e["payload"]["changes"]["actual_start_at"]})  # fmt: skip
    return out


def reconcile_match(st: MatchState, events: list[dict], records: list[dict], players: frozenset | None) -> dict:
    """Outcome of one match. ``events``: the match's ledger events (parsed payloads) up to the cutoff."""
    cur, flags = current_records(records)
    starts = _actual_starts(events) + [
        {"origin": "outcome_snapshot", "provider": r["provider"], "record": r["provider_record_id"],
         "actual_start_at": r["actual_start_at"]} for r in cur if r["actual_start_at"] is not None
    ]  # fmt: skip
    start_times = sorted({s["actual_start_at"] for s in starts}, key=parse_ts)
    if len(start_times) > 1:
        flags.append("actual_start_sources_disagree")
    assertions = []
    if st.result is not None:
        res = st.result
        assertions.append({"origin": "ledger", "event_id": st.result_event, "provider": res["source"].get("provider"),
                           "status": res["status"], "winner_id": res["winner_id"], "score": res["score"],
                           "player_ids": None})  # fmt: skip
    for r in cur:
        assertions.append({"origin": "outcome_snapshot", "provider": r["provider"], "record": r["provider_record_id"],
                           "source_version": r["source_version"], "status": r["status"], "winner_id": r["winner_id"],
                           "score": r["score"], "player_ids": sorted(r["player_ids"])})  # fmt: skip
    level = next((r["tournament_level"] for r in cur if r["tournament_level"]), None)
    out = {"match_id": st.match_id, "outcome": None, "status": None, "winner_id": None, "assertions": assertions,
           "flags": flags, "corrected": len(st.result_history) > 1, "result_history": list(st.result_history),
           "actual_start_at": start_times[0] if start_times else None, "actual_start_sources": starts,
           "tournament_level": level, "match_status": st.status}  # fmt: skip
    if st.status == "superseded":
        out["outcome"] = "SUPERSEDED"
        return out
    if st.status == "cancelled":
        played = [a for a in assertions if a["status"] in WINNER_REQUIRED or a["status"] == "abandoned"]
        out["outcome"] = "DISPUTED" if played else "CANCELLED"
        if played:
            flags.append("cancelled_but_result_reported")
        return out
    if any(f.startswith("version_reissued") for f in flags):
        out["outcome"] = "DISPUTED"
        return out
    if not assertions:
        out["outcome"] = "PENDING"
        return out
    for a in assertions:
        if a["score"]:
            ms = match_status(a["score"])
            if ms not in ("unknown", "unfinished") and ms != a["status"]:
                flags.append(f"score_status_inconsistent:{a['provider']}")
        if players is not None:
            if a["player_ids"] is not None and frozenset(a["player_ids"]) != players:
                flags.append(f"players_differ:{a['provider']}")
            if a["winner_id"] is not None and a["winner_id"] not in players:
                flags.append(f"winner_not_in_match:{a['provider']}")
    if any(f.startswith(("players_differ", "winner_not_in_match")) for f in flags):
        out["outcome"] = "IDENTITY_MISMATCH"
        return out
    keys = {(a["status"], a["winner_id"]) for a in assertions}
    if len(keys) > 1 or any(f.startswith("score_status_inconsistent") for f in flags):
        out["outcome"] = "DISPUTED"
        if len(keys) > 1:
            flags.append("sources_disagree")
        return out
    status, winner = next(iter(keys))
    out.update(outcome=OUTCOMES[status], status=status, winner_id=winner)
    return out


def reconcile_all(states: dict[str, MatchState], events_by_match: dict[str, list[dict]], outcomes: OutcomeSnapshot,
                  players: dict[str, frozenset], cutoff: str | None) -> dict[str, dict]:  # fmt: skip
    recs: dict = {}
    for r in outcomes.records:
        if cutoff is not None and parse_ts(r["retrieved_at"]) > parse_ts(cutoff):
            continue
        recs.setdefault(r["match_id"], []).append(r)
    out = {}
    for mid in sorted(set(states) | set(recs)):
        st = states.get(mid) or MatchState(mid)
        out[mid] = reconcile_match(st, events_by_match.get(mid, []), recs.get(mid, []), players.get(mid))
    return out

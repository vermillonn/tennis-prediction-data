"""Deterministic lifecycle rules: current match and prediction status derived from the ordered event history.

Per match (events in sequence order):

| event | allowed when | effect |
|---|---|---|
| PREDICTION_COMMITTED | match not cancelled or superseded; no result yet (except HISTORICAL_REPLAY) | adds a prediction |
| MATCH_POSTPONED | not cancelled, superseded or decided | new scheduled start (the original is kept) |
| MATCH_CANCELLED | not cancelled, superseded or decided | status cancelled (terminal) |
| MATCH_METADATA_CORRECTED | not superseded | schedule / actual start / other; ``players`` supersedes the match (terminal) |
| MATCH_RESULT_RECORDED | not cancelled or superseded; first result has no ``supersedes``; a correction must supersede the current result | current result |
| PREDICTION_VOIDED | the referenced prediction exists, is not voided, and the reason's evidence exists | prediction voided (still listed) |

Nothing is ever removed: superseded results, voided predictions and old schedules stay in the history and in exports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tennis_data.ledger.events import VOID_REASONS
from tennis_data.ledger.hashing import parse_ts


class TransitionError(ValueError):
    pass


@dataclass
class MatchState:
    match_id: str
    status: str = "scheduled"  # scheduled | postponed | cancelled | superseded | decided
    scheduled_start_history: list = field(default_factory=list)  # (seq, timestamp or None, source)
    actual_start_at: str | None = None
    actual_start_event: str | None = None
    result_event: str | None = None
    result: dict | None = None
    result_history: list = field(default_factory=list)
    superseded_by: str | None = None
    predictions: dict = field(default_factory=dict)  # event_id -> dict(mode, recorded_at, voided, ...)
    events: list = field(default_factory=list)  # (seq, event_type, event_id)


def _payload(e: dict) -> dict:
    return e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])


def apply(st: MatchState, e: dict, all_by_id: dict | None = None) -> MatchState:
    """Apply one event to a match state, raising TransitionError if the transition is not allowed."""
    t, p = e["event_type"], _payload(e)
    closed = st.status in ("cancelled", "superseded")
    if t == "PREDICTION_COMMITTED":
        if closed:
            raise TransitionError(f"{st.match_id}: cannot predict a {st.status} match")
        mode = p["audit"]["mode"]
        if st.status == "decided" and mode != "HISTORICAL_REPLAY":
            raise TransitionError(f"{st.match_id}: a result is already recorded; only HISTORICAL_REPLAY may follow")
        if not st.scheduled_start_history:
            st.scheduled_start_history.append((e["seq"], p["match"]["scheduled_start_at"], "prediction"))
        st.predictions[e["event_id"]] = {
            "mode": mode, "recorded_at": e["recorded_at"], "voided": None, "seq": e["seq"],
            "scheduled_start_at": p["match"]["scheduled_start_at"],
            "scheduled_start_verified": p["match"]["scheduled_start_verified"],
            "prediction_created_at": p["prediction"]["prediction_created_at"],
        }  # fmt: skip
    elif t == "MATCH_POSTPONED":
        if closed or st.status == "decided":
            raise TransitionError(f"{st.match_id}: cannot postpone a {st.status} match")
        st.status = "postponed"
        st.scheduled_start_history.append((e["seq"], p["new_scheduled_start_at"], p["source"].get("provider")))
    elif t == "MATCH_CANCELLED":
        if closed or st.status == "decided":
            raise TransitionError(f"{st.match_id}: cannot cancel a {st.status} match")
        st.status = "cancelled"
    elif t == "MATCH_METADATA_CORRECTED":
        if st.status == "superseded":
            raise TransitionError(f"{st.match_id}: match is superseded by {st.superseded_by}")
        kind, ch = p["kind"], p["changes"]
        if kind == "schedule":
            st.scheduled_start_history.append((e["seq"], ch["scheduled_start_at"], p["source"].get("provider")))
        elif kind == "actual_start":
            st.actual_start_at, st.actual_start_event = ch["actual_start_at"], e["event_id"]
        elif kind == "players":
            if st.status == "decided":
                raise TransitionError(f"{st.match_id}: players cannot change after a result")
            st.status, st.superseded_by = "superseded", ch["superseded_by_match_id"]
    elif t == "MATCH_RESULT_RECORDED":
        if closed:
            raise TransitionError(f"{st.match_id}: cannot record a result for a {st.status} match")
        if st.result_event is None and p["supersedes_event_id"] is not None:
            raise TransitionError(f"{st.match_id}: correction without an earlier result")
        if st.result_event is not None and p["supersedes_event_id"] != st.result_event:
            raise TransitionError(f"{st.match_id}: a second result must supersede the current one ({st.result_event})")
        st.result_history.append(e["event_id"])
        st.result_event, st.result = e["event_id"], p
        st.status = "decided"
    elif t == "PREDICTION_VOIDED":
        ref = e["ref_event_id"]
        if ref not in st.predictions:
            raise TransitionError(f"{st.match_id}: void references unknown prediction {ref}")
        if st.predictions[ref]["voided"] is not None:
            raise TransitionError(f"{st.match_id}: prediction {ref} is already voided")
        _check_void_evidence(st, p, ref, all_by_id or {})
        st.predictions[ref]["voided"] = {
            "event_id": e["event_id"],
            "reason_code": p["reason_code"],
            "actor": p["actor"],
        }
    elif t == "LEDGER_CHECKPOINT_CREATED":
        raise TransitionError("checkpoint events do not belong to a match")
    st.events.append((e["seq"], t, e["event_id"]))
    return st


def _check_void_evidence(st: MatchState, p: dict, ref: str, by_id: dict) -> None:
    need = VOID_REASONS[p["reason_code"]]
    own = {eid for _, _, eid in st.events}
    for eid in p["evidence_event_ids"]:
        if eid not in own:
            raise TransitionError(f"void evidence {eid} is not an earlier event of {st.match_id}")
    ev = [by_id[eid] for eid in p["evidence_event_ids"] if eid in by_id]
    ok = False
    for x in ev:
        xp = _payload(x)
        if x["event_type"] != need:
            continue
        r = p["reason_code"]
        if r == "match_cancelled":
            ok = True
        elif r == "players_changed" and xp["kind"] == "players":
            ok = True
        elif r == "walkover_no_play" and xp["status"] == "walkover":
            ok = True
        elif r == "committed_after_actual_start" and xp["kind"] == "actual_start":
            ok = parse_ts(xp["changes"]["actual_start_at"]) <= parse_ts(st.predictions[ref]["recorded_at"])
    if not ok:
        raise TransitionError(f"void reason {p['reason_code']} is not supported by its evidence")


def replay(events: list[dict]) -> dict[str, MatchState]:
    """Rebuild every match state from the full ordered history (raises on the first invalid transition)."""
    by_id = {e["event_id"]: e for e in events}
    states: dict[str, MatchState] = {}
    for e in events:
        if e["match_id"] is None:
            continue
        st = states.setdefault(e["match_id"], MatchState(e["match_id"]))
        apply(st, e, by_id)
    return states


def timing(st: MatchState, pred_event_id: str) -> dict:
    """Pre-match status of one prediction, derived (never rewritten) from the history.

    - generated_before_scheduled_start: creation time < scheduled start recorded *in the prediction* (None if absent);
    - committed_before_scheduled_start: ledger commit time < that scheduled start (None if absent);
    - scheduled_start_verified: whether that start came from a verified source;
    - verified_before_actual_start: commit time < actual start, once a sourced actual start exists; else None.
    """
    p = st.predictions[pred_event_id]
    sched = parse_ts(p["scheduled_start_at"]) if p["scheduled_start_at"] else None
    rec = parse_ts(p["recorded_at"])
    created = parse_ts(p["prediction_created_at"])
    actual = parse_ts(st.actual_start_at) if st.actual_start_at else None
    return {
        "generated_before_scheduled_start": None if sched is None else created < sched,
        "committed_before_scheduled_start": None if sched is None else rec < sched,
        "scheduled_start_verified": bool(p["scheduled_start_verified"]),
        "verified_before_actual_start": None if actual is None else rec < actual,
        "actual_start_event": st.actual_start_event,
    }

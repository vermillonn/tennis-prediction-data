"""Prediction classification: one mutually exclusive category per committed prediction, plus overlapping flags.

Categories are assigned by the first rule that applies, in this fixed order (protocol ``pe-1``):

| order | category | rule |
|---|---|---|
| 1 | INVALID_CHAIN | the snapshot fails integrity verification (nothing is scored) |
| 2 | INVALID_EVIDENCE | an anchor or retained trusted checkpoint contradicts the snapshot |
| 3 | SYNTHETIC / HISTORICAL_REPLAY / PROSPECTIVE | the record's mode is not the protocol's cohort mode (never upgraded) |
| 4 | MODEL_NOT_APPROVED | model id, frozen hash or configuration hash is not the approved frozen M4A |
| 5 | MISSING_BASELINE | no committed Elo baseline probability |
| 6 | NOT_OPERATIONALLY_READY | the recorded readiness evidence fails (see ``readiness``) |
| 7 | NOT_PRIMARY_SLOT | committed in another forecast slot |
| 8 | DUPLICATE_SUPERSEDED | an earlier admissible forecast of the match in the primary slot exists (selection) |
| 9 | VOIDED | voided by an objective, evidence-backed reason (visible in the exclusion report) |
| 10 | MATCH_CANCELLED | the match was cancelled |
| 11 | MATCH_SUPERSEDED | players changed; the match id was superseded |
| 12 | LATE_COMMITMENT | committed at or after the earliest sourced actual start |
| 13 | OUTCOME_PENDING | admissible forecast, no outcome yet (stays in the cohort) |
| 14 | TIMING_UNRESOLVED | outcome known but no actual-start evidence (pre-start not verified) |
| 15 | OUTCOME_IDENTITY_MISMATCH | outcome players or winner do not match the forecast |
| 16 | OUTCOME_DISPUTED | sources disagree or a version was reissued |
| 17 | OUTCOME_RETIRED | retirement (secondary analysis only, per ``retirement_policy``) |
| 18 | OUTCOME_WALKOVER / _DEFAULT / _ABANDONED / _UNKNOWN | not a played, completed match |
| 19 | SCORABLE | everything above passed and the outcome is COMPLETED with a binary winner |

Selection (rule 8) happens before any outcome is consulted: per match, the forecast with the lowest sequence among
those passing rules 3–7 in the primary slot is selected. If it is voided or late, the match drops out; a later
forecast never takes its place, so a losing forecast cannot be replaced by a better one.

Evidence level of each scorable forecast is the strongest *verified* anchor covering its sequence. It is
``INDEPENDENTLY_TIMESTAMPED`` only if the attested time precedes the actual start; otherwise the timestamp proves
nothing about pre-match commitment and the level falls back. The primary cohort is SCORABLE ∧
INDEPENDENTLY_TIMESTAMPED; the operational cohort is every SCORABLE forecast, reported with its weaker evidence.
"""

from __future__ import annotations

import json
import math

from tennis_data.evaluation_prospective.anchoring import evidence_for
from tennis_data.evaluation_prospective.outcomes import reconcile_all
from tennis_data.ledger.hashing import parse_ts
from tennis_data.ledger.lifecycle import TransitionError, replay

CATEGORIES = (
    "INVALID_CHAIN", "INVALID_EVIDENCE", "SYNTHETIC", "HISTORICAL_REPLAY", "PROSPECTIVE", "MODEL_NOT_APPROVED",
    "MISSING_BASELINE", "NOT_OPERATIONALLY_READY", "NOT_PRIMARY_SLOT", "DUPLICATE_SUPERSEDED", "VOIDED",
    "MATCH_CANCELLED", "MATCH_SUPERSEDED", "LATE_COMMITMENT", "OUTCOME_PENDING", "TIMING_UNRESOLVED",
    "OUTCOME_IDENTITY_MISMATCH", "OUTCOME_DISPUTED", "OUTCOME_RETIRED", "OUTCOME_WALKOVER", "OUTCOME_DEFAULT",
    "OUTCOME_ABANDONED", "OUTCOME_UNKNOWN", "SCORABLE",
)  # fmt: skip
OUTCOME_CATEGORY = {"IDENTITY_MISMATCH": "OUTCOME_IDENTITY_MISMATCH", "DISPUTED": "OUTCOME_DISPUTED",
                    "RETIRED": "OUTCOME_RETIRED", "WALKOVER": "OUTCOME_WALKOVER", "DEFAULT": "OUTCOME_DEFAULT",
                    "ABANDONED": "OUTCOME_ABANDONED", "UNKNOWN": "OUTCOME_UNKNOWN"}  # fmt: skip


def readiness(p: dict, mode: str) -> list[str]:
    """Readiness failures recorded in the prediction itself. PROSPECTIVE adds the Phase 4B commit-time gates."""
    m, pr, a = p["match"], p["prediction"], p["audit"]
    bad = []
    if pr.get("freshness_status") == "fail":
        bad.append("freshness_fail")
    if not m.get("scheduled_start_verified"):
        bad.append("schedule_unverified")
    if mode == "PROSPECTIVE":
        if a.get("clock") != "system":
            bad.append("untrusted_clock")
        if a.get("prospective_gate") != ["COMPLETE", "ENABLED"]:
            bad.append("status_gate_not_enabled")
        if not pr.get("prediction_cutoff"):
            bad.append("no_prediction_cutoff")
        if (a.get("timing") or {}).get("committed_before_scheduled_start") is not True:
            bad.append("not_committed_before_scheduled_start")
    return bad


def _bucket(v, edges, unit):
    if v is None:
        return "(unavailable)"
    if v < edges[0]:
        return f"0: <{edges[0]} {unit}"
    for i in range(len(edges) - 1):
        if v < edges[i + 1]:
            return f"{i + 1}: {edges[i]}-{edges[i + 1] - 1} {unit}"
    return f"{len(edges)}: >={edges[-1]} {unit}"


def _min_pair(fa: dict, a: str, b: str):
    va, vb = fa.get(a), fa.get(b)
    if not isinstance(va, int | float) or not isinstance(vb, int | float):
        return None
    return min(va, vb)


def classify(events: list[dict], protocol: dict, outcomes, *, integrity_ok: bool, evidence_ok: bool,
             anchor_cov: dict, cutoff: str | None = None) -> dict:  # fmt: skip
    """Classify every PREDICTION_COMMITTED event of ``events`` (stored form, sequence order)."""
    cut = parse_ts(cutoff) if cutoff else None
    parsed = []
    for e in events:
        if cut is not None and parse_ts(e["recorded_at"]) > cut:
            break  # recorded_at is non-decreasing in a verified chain: the rest is after the cutoff
        p = e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"])
        parsed.append({**e, "payload": p})
    lifecycle_ok = integrity_ok
    states = {}
    if integrity_ok:
        try:
            states = replay(parsed)
        except (TransitionError, KeyError, TypeError):
            lifecycle_ok = False
    by_match: dict = {}
    for e in parsed:
        if e["match_id"] is not None:
            by_match.setdefault(e["match_id"], []).append(e)
    preds = [e for e in parsed if e["event_type"] == "PREDICTION_COMMITTED"]
    players = {e["match_id"]: frozenset((e["payload"]["match"]["player_a_id"], e["payload"]["match"]["player_b_id"]))
               for e in preds}  # fmt: skip
    oc = reconcile_all(states, by_match, outcomes, players, cutoff) if lifecycle_ok else {}
    mode_target, slot = protocol["cohort_mode"], protocol["primary_forecast_slot"]

    rows = []
    for e in preds:
        p = e["payload"]
        m, pr, a = p["match"], p["prediction"], p["audit"]
        fa = pr.get("feature_availability") or {}
        st = states.get(e["match_id"])
        info = st.predictions.get(e["event_id"]) if st else None
        o = oc.get(e["match_id"], {})
        flags = []
        row = {
            "seq": e["seq"], "event_id": e["event_id"], "match_id": e["match_id"], "mode": e["mode"],
            "forecast_slot": a.get("forecast_slot"), "recorded_at": e["recorded_at"], "model_id": pr.get("model_id"),
            "frozen_hash": pr.get("frozen_hash"), "configuration_hash": pr.get("configuration_hash"),
            "probability_a": pr.get("probability_a"), "baseline_elo_probability_a": pr.get("baseline_elo_probability_a"),
            "player_a_id": m.get("player_a_id"), "player_b_id": m.get("player_b_id"), "surface": m.get("surface"),
            "best_of": m.get("best_of"), "tournament_id": m.get("tournament_id"),
            "scheduled_start_at": m.get("scheduled_start_at"), "freshness_status": pr.get("freshness_status"),
            "prior_service_points": _min_pair(fa, "a_service_points_observed", "b_service_points_observed"),
            "elo_matches": _min_pair(fa, "global_match_count_a", "global_match_count_b"),
            "voided": (info or {}).get("voided"), "outcome": o.get("outcome"), "outcome_status": o.get("status"),
            "winner_id": o.get("winner_id"), "actual_start_at": o.get("actual_start_at"),
            "tournament_level": o.get("tournament_level"), "outcome_corrected": bool(o.get("corrected")),
            "timing": None, "evidence_level": None, "attested_time": None, "selected": False, "admissible": False,
            "cohort": None, "y": None, "category": None, "flags": flags,
        }  # fmt: skip
        if row["winner_id"] is not None and row["winner_id"] in (row["player_a_id"], row["player_b_id"]):
            row["y"] = 1 if row["winner_id"] == row["player_a_id"] else 0
        # timing evidence (host-clock commit time vs earliest sourced actual start)
        if row["actual_start_at"] is None:
            row["timing"] = "UNRESOLVED"
        elif parse_ts(e["recorded_at"]) < parse_ts(row["actual_start_at"]):
            row["timing"] = "PRE_START"
        else:
            row["timing"] = "LATE"
            flags.append("committed_at_or_after_actual_start")
        # evidence level of the commitment
        if not (integrity_ok and lifecycle_ok and evidence_ok):
            row["evidence_level"] = "INVALID"
        else:
            lvl, att = evidence_for(e["seq"], anchor_cov)
            if lvl == "INDEPENDENTLY_TIMESTAMPED" and (row["actual_start_at"] is None or parse_ts(att) >= parse_ts(
                    row["actual_start_at"])):  # fmt: skip
                flags.append("timestamp_not_before_actual_start" if row["actual_start_at"] else
                             "timestamp_without_actual_start")  # fmt: skip
                lvl = "EXTERNALLY_PUBLISHED" if anchor_cov["published_through_seq"] >= e["seq"] else "LOCAL_ONLY"
                att = None if row["actual_start_at"] else att
            row["evidence_level"], row["attested_time"] = lvl, att
        if e["mode"] != mode_target:
            flags.append(f"mode_{e['mode'].lower()}")
        model_ok = (pr.get("model_id") == protocol["approved_model_id"]
                    and str(pr.get("frozen_hash", "")).startswith(protocol["approved_frozen_hash"])
                    and pr.get("configuration_hash") == protocol["approved_configuration_hash"])  # fmt: skip
        if not model_ok:
            flags.append("model_not_approved")
        b = row["baseline_elo_probability_a"]
        baseline_ok = isinstance(b, float) and math.isfinite(b) and 0.0 <= b <= 1.0
        if not baseline_ok:
            flags.append("missing_baseline")
        ready = readiness(p, e["mode"])
        flags.extend(f"readiness:{x}" for x in ready)
        if row["voided"]:
            flags.append(f"voided:{row['voided']['reason_code']}")
        row["_pre_ok"] = e["mode"] == mode_target and model_ok and baseline_ok and not ready
        rows.append(row)

    # selection: first admissible forecast per match in the primary slot, decided before outcomes are used
    first: dict = {}
    for r in rows:
        if r["_pre_ok"] and r["forecast_slot"] == slot and r["match_id"] not in first:
            first[r["match_id"]] = r["seq"]
    for r in rows:
        r["selected"] = first.get(r["match_id"]) == r["seq"]
        r["category"] = _category(r, protocol, integrity_ok and lifecycle_ok, evidence_ok)
        r["admissible"] = r["selected"] and r["category"] not in (
            "VOIDED",
            "MATCH_CANCELLED",
            "MATCH_SUPERSEDED",
            "LATE_COMMITMENT",
            "TIMING_UNRESOLVED",
        )
        if r["category"] == "SCORABLE":
            r["cohort"] = "primary" if r["evidence_level"] == "INDEPENDENTLY_TIMESTAMPED" else "operational"
        elif r["category"] == "OUTCOME_RETIRED" and protocol["retirement_policy"] == "separate":
            r["cohort"] = "retirement_secondary"
        del r["_pre_ok"]
    return {"rows": rows, "outcomes": oc, "lifecycle_ok": lifecycle_ok, "events_considered": len(parsed),
            "matches": len(by_match)}  # fmt: skip


def _category(r: dict, protocol: dict, chain_ok: bool, evidence_ok: bool) -> str:
    f = r["flags"]
    if not chain_ok:
        return "INVALID_CHAIN"
    if not evidence_ok:
        return "INVALID_EVIDENCE"
    if r["mode"] != protocol["cohort_mode"]:
        return r["mode"]
    if "model_not_approved" in f:
        return "MODEL_NOT_APPROVED"
    if "missing_baseline" in f:
        return "MISSING_BASELINE"
    if any(x.startswith("readiness:") for x in f):
        return "NOT_OPERATIONALLY_READY"
    if r["forecast_slot"] != protocol["primary_forecast_slot"]:
        return "NOT_PRIMARY_SLOT"
    if not r["selected"]:
        return "DUPLICATE_SUPERSEDED"
    if r["voided"]:
        return "VOIDED"
    if r["outcome"] == "CANCELLED":
        return "MATCH_CANCELLED"
    if r["outcome"] == "SUPERSEDED":
        return "MATCH_SUPERSEDED"
    if r["timing"] == "LATE":
        return "LATE_COMMITMENT"
    if r["outcome"] in (None, "PENDING"):
        return "OUTCOME_PENDING"
    if r["timing"] == "UNRESOLVED" and protocol["require_actual_start_evidence"]:
        return "TIMING_UNRESOLVED"
    if r["outcome"] in OUTCOME_CATEGORY:
        return OUTCOME_CATEGORY[r["outcome"]]
    if r["outcome"] in protocol["scored_outcomes"] and r["y"] is not None:
        return "SCORABLE"
    return "OUTCOME_UNKNOWN"


def subgroup_values(r: dict, protocol: dict) -> dict:
    """Pre-declared subgroup labels for one row (all computed; small groups are marked, never dropped)."""
    p = r["probability_a"]
    conf = max(p, 1 - p) if isinstance(p, float) else None
    cb = None if conf is None else min(9, max(5, int(conf * 10)))
    t = r["actual_start_at"] or r["scheduled_start_at"]
    return {
        "surface": r["surface"] or "(missing)",
        "tournament_level": r["tournament_level"] or "(unavailable)",
        "best_of": f"best of {r['best_of']}" if r["best_of"] else "(missing)",
        "prior_service_points": _bucket(r["prior_service_points"], protocol["sample_size_edges"], "points"),
        "elo_experience": _bucket(r["elo_matches"], protocol["elo_experience_edges"], "matches"),
        "probability_range": "(missing)" if cb is None else f"{cb * 10}-{cb * 10 + 10}% favourite",
        "freshness": r["freshness_status"] or "(missing)",
        "period": t[:7] if t else "(missing)",
    }  # fmt: skip

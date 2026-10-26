"""Event payload schemas (``ledger-1``) and their validation.

All timestamps inside payloads use the canonical form of ``hashing.ts`` (UTC, microseconds, ``Z``). External provider
match identifiers are kept in ``match.provider_match_ids`` and never replace the canonical ``match_id``.
"""

from __future__ import annotations

import math

from tennis_data.ledger.hashing import CanonicalizationError, parse_ts
from tennis_data.ledger.storage import MODES

RESULT_STATUSES = ("completed", "retired", "walkover", "default", "abandoned", "unknown")
WINNER_REQUIRED = ("completed", "retired", "walkover", "default")
METADATA_KINDS = ("schedule", "actual_start", "players", "other")
# Objective, predefined voiding criteria. Each needs evidence events of the listed type for the same match, so a
# forecast cannot be voided merely because it lost (there is no "outcome" reason).
VOID_REASONS = {
    "match_cancelled": "MATCH_CANCELLED",
    "players_changed": "MATCH_METADATA_CORRECTED",  # kind = players
    "committed_after_actual_start": "MATCH_METADATA_CORRECTED",  # kind = actual_start, start <= commit time
    "walkover_no_play": "MATCH_RESULT_RECORDED",  # status = walkover
}
FRESHNESS = ("ok", "warning", "fail", "unknown")
NAMESPACES = {"SYNTHETIC": "synthetic:", "HISTORICAL_REPLAY": "replay:"}
ORIENTATION_RULE = "phase1-sha256-match-id"  # tennis_data.normalization.ordering.assign_players


class EventValidationError(ValueError):
    pass


def _req(d: dict, keys, where):
    missing = [k for k in keys if k not in d]
    if missing:
        raise EventValidationError(f"{where}: missing {missing}")


def _ts_or_none(v, where):
    if v is None:
        return None
    try:
        return parse_ts(v)
    except CanonicalizationError as e:
        raise EventValidationError(f"{where}: {e}") from e


def _source(src, where):
    if not isinstance(src, dict) or not src.get("provider"):
        raise EventValidationError(f"{where}: a source with a provider is required for provenance")
    _ts_or_none(src.get("retrieved_at"), f"{where}.retrieved_at")


def check_probabilities(pa, pb, tol: float) -> None:
    for name, p in (("probability_a", pa), ("probability_b", pb)):
        if not isinstance(p, float) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise EventValidationError(f"{name} must be a finite float in [0, 1], got {p!r}")
    if abs(pa + pb - 1.0) > tol:
        raise EventValidationError(f"P(A) + P(B) = {pa + pb!r} differs from 1 by more than {tol}")


def check_mode_namespace(mode: str, match_id: str) -> None:
    """Modes cannot be confused: SYNTHETIC and HISTORICAL_REPLAY records live in their own match-id namespaces."""
    if mode not in MODES:
        raise EventValidationError(f"mode must be one of {MODES}")
    for m, prefix in NAMESPACES.items():
        if mode == m and not match_id.startswith(prefix):
            raise EventValidationError(f"{mode} predictions need match ids starting with {prefix!r}")
        if mode != m and match_id.startswith(prefix):
            raise EventValidationError(f"match id {match_id!r} belongs to the {m} namespace, not {mode}")


def validate_prediction(p: dict, tol: float) -> None:
    _req(p, ("match", "prediction", "audit"), "PREDICTION_COMMITTED")
    m, pr, a = p["match"], p["prediction"], p["audit"]
    _req(m, ("match_id", "provider_match_ids", "player_a_id", "player_b_id", "tournament_id", "surface", "round",
             "best_of", "scheduled_start_at", "scheduled_start_verified", "schedule_snapshot_id"), "match")  # fmt: skip
    _req(pr, ("model_id", "model_version", "frozen_hash", "configuration_hash", "feature_schema_version",
              "state_snapshot_id", "feature_state_hash", "source_snapshots", "prediction_created_at",
              "prediction_cutoff", "probability_a", "probability_b", "baseline_elo_probability_a",
              "feature_availability", "freshness_status"), "prediction")  # fmt: skip
    _req(a, ("mode", "forecast_slot", "inference_engine_version", "git_commit", "ledger_schema_version",
             "idempotency_key", "content_hash", "orientation", "clock", "timing"), "audit")  # fmt: skip
    if (
        not isinstance(m["player_a_id"], int)
        or not isinstance(m["player_b_id"], int)
        or m["player_a_id"] == m["player_b_id"]
    ):
        raise EventValidationError("player ids must be two different integers")
    if m["best_of"] not in (None, 3, 5):
        raise EventValidationError("best_of must be 3, 5 or null")
    if not isinstance(m["provider_match_ids"], dict):
        raise EventValidationError("provider_match_ids must map provider -> external match id")
    if m["scheduled_start_verified"] and m["scheduled_start_at"] is None:
        raise EventValidationError("a verified scheduled start needs a timestamp")
    _ts_or_none(m["scheduled_start_at"], "scheduled_start_at")
    _ts_or_none(pr["prediction_created_at"], "prediction_created_at")
    _ts_or_none(pr["prediction_cutoff"], "prediction_cutoff")
    check_probabilities(pr["probability_a"], pr["probability_b"], tol)
    b = pr["baseline_elo_probability_a"]
    if not isinstance(b, float) or not 0.0 <= b <= 1.0:
        raise EventValidationError("baseline_elo_probability_a must be a float in [0, 1]")
    if pr["freshness_status"] not in FRESHNESS:
        raise EventValidationError(f"freshness_status must be one of {FRESHNESS}")
    if a["orientation"] != ORIENTATION_RULE:
        raise EventValidationError("unknown player-orientation rule")
    check_mode_namespace(a["mode"], m["match_id"])


def validate_payload(event_type: str, payload: dict, tol: float = 1e-12) -> None:
    if not isinstance(payload, dict):
        raise EventValidationError("payload must be an object")
    if event_type == "PREDICTION_COMMITTED":
        validate_prediction(payload, tol)
    elif event_type == "MATCH_METADATA_CORRECTED":
        _req(payload, ("kind", "changes", "reason", "source"), event_type)
        if payload["kind"] not in METADATA_KINDS:
            raise EventValidationError(f"kind must be one of {METADATA_KINDS}")
        _source(payload["source"], event_type)
        ch = payload["changes"]
        if payload["kind"] == "schedule":
            _req(ch, ("scheduled_start_at",), "changes")
            _ts_or_none(ch["scheduled_start_at"], "changes.scheduled_start_at")
        if payload["kind"] == "actual_start":
            _req(ch, ("actual_start_at",), "changes")
            if _ts_or_none(ch["actual_start_at"], "changes.actual_start_at") is None:
                raise EventValidationError("actual_start_at is required")
        if payload["kind"] == "players":
            _req(ch, ("superseded_by_match_id", "player_a_id", "player_b_id"), "changes")
    elif event_type == "MATCH_POSTPONED":
        _req(payload, ("new_scheduled_start_at", "reason", "source"), event_type)
        _ts_or_none(payload["new_scheduled_start_at"], "new_scheduled_start_at")
        _source(payload["source"], event_type)
    elif event_type == "MATCH_CANCELLED":
        _req(payload, ("reason", "source"), event_type)
        _source(payload["source"], event_type)
    elif event_type == "MATCH_RESULT_RECORDED":
        _req(
            payload, ("status", "winner_id", "score", "source", "supersedes_event_id", "correction_reason"), event_type
        )
        if payload["status"] not in RESULT_STATUSES:
            raise EventValidationError(f"result status must be one of {RESULT_STATUSES}")
        if payload["status"] in WINNER_REQUIRED and not isinstance(payload["winner_id"], int):
            raise EventValidationError(f"status {payload['status']} needs a winner_id")
        if payload["status"] not in WINNER_REQUIRED and payload["winner_id"] is not None:
            raise EventValidationError(f"status {payload['status']} has no winner (not forced into win/loss)")
        _source(payload["source"], event_type)
        if (payload["supersedes_event_id"] is None) != (payload["correction_reason"] is None):
            raise EventValidationError("a result correction needs both supersedes_event_id and correction_reason")
    elif event_type == "PREDICTION_VOIDED":
        _req(payload, ("reason_code", "evidence_event_ids", "actor", "note"), event_type)
        if payload["reason_code"] not in VOID_REASONS:
            raise EventValidationError(f"void reason must be one of {sorted(VOID_REASONS)}")
        if not payload["actor"]:
            raise EventValidationError("void needs the authorized actor or process identity")
        if not payload["evidence_event_ids"]:
            raise EventValidationError("void needs evidence events")
    elif event_type == "LEDGER_CHECKPOINT_CREATED":
        _req(payload, ("checkpoint_hash", "covered_seq", "covered_head_hash", "previous_checkpoint_hash"), event_type)
    else:
        raise EventValidationError(f"unknown event type {event_type}")

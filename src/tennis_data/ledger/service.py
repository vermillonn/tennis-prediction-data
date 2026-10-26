"""Trusted application service: validated upcoming match → freshness → frozen M4A inference → ledger commit → receipt.

Uses the Phase 4A path unchanged (``load_state_snapshot``, ``compute_state``, ``assess``, ``load_production``,
``predict_state``); no feature or model code is duplicated here. A prediction is "recorded" only when the ledger
transaction has committed and a receipt exists; any failure raises and no receipt is returned.

PROSPECTIVE requests are refused here *and* in ``LedgerWriter`` while the Phase 4A status gate is not ENABLED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from tennis_data.ledger import storage
from tennis_data.ledger.hashing import ts
from tennis_data.ledger.verification import verify_events
from tennis_data.ledger.writer import LedgerWriter, PredictionRejected, PredictionRequest, Receipt, orient
from tennis_data.prospective.config import STATE_ENGINE_VERSION
from tennis_data.prospective.inference import ModelArtifact, load_production, predict_state

PHASE4A_MODE = {"SYNTHETIC": "exploratory_retrospective", "HISTORICAL_REPLAY": "historical_reconstruction",
                "PROSPECTIVE": "prospective"}  # fmt: skip


@dataclass(frozen=True)
class Schedule:
    """Schedule facts for one match, as supplied by a schedule source snapshot (never invented)."""

    scheduled_start_at: datetime | None
    verified: bool
    snapshot_id: str | None
    retrieved_at: datetime | None
    provider_match_ids: dict


def _gate(writer: LedgerWriter, mode: str) -> None:
    p = writer.cfg.prospective
    if mode == "PROSPECTIVE" and (p.prospective_predictions != "ENABLED" or p.backfill_status != "COMPLETE"):
        raise PredictionRejected(f"PROSPECTIVE predictions are {p.prospective_predictions} "
                                 f"(backfill {p.backfill_status})")  # fmt: skip


def commit_state_rows(writer: LedgerWriter, pending: pl.DataFrame, art: ModelArtifact, *, mode: str, forecast_slot: str,
                      snapshot_id: str, components: dict, schedules: dict[str, Schedule], tournament_ids: dict,
                      rounds: dict, best_of: dict, freshness: dict | None = None, cutoff: datetime | None = None,
                      git_commit: str | None = None) -> list[tuple[Receipt, PredictionRequest]]:  # fmt: skip
    """Price pending state rows with the frozen model and commit each prediction (one transaction each)."""
    _gate(writer, mode)
    pcfg = writer.cfg.prospective
    created = writer.clock.now()  # creation time from the ledger's clock, not from the caller
    preds = predict_state(pending, art, pcfg, prediction_mode=PHASE4A_MODE[mode], snapshot_id=snapshot_id,
                          components=components, freshness=freshness, generated_at=created)  # fmt: skip
    out = []
    for r in preds.iter_rows(named=True):
        mid = r["match_id"]
        sch = schedules[mid]
        o = orient(mid, r["player_a_id"], r["player_b_id"], r["probability_a"], r["baseline_elo_probability_a"])
        if o["player_a_id"] != r["player_a_id"]:  # use the model's own B probability (exactly 1 - P(A))
            o["probability_a"], o["probability_b"] = r["probability_b"], r["probability_a"]
        fresh = json.loads(r["freshness"]) if r["freshness"] else {}
        req = PredictionRequest(
            mode=mode, forecast_slot=forecast_slot, match_id=mid, tournament_id=tournament_ids[mid],
            surface=r["surface"], round=rounds.get(mid), best_of=best_of.get(mid),
            scheduled_start_at=ts(sch.scheduled_start_at) if sch.scheduled_start_at else None,
            scheduled_start_verified=sch.verified, schedule_snapshot_id=sch.snapshot_id,
            schedule_retrieved_at=ts(sch.retrieved_at) if sch.retrieved_at else None,
            model_id=art.model_id, model_version=art.model_version, frozen_hash=art.frozen_hash,
            configuration_hash=pcfg.modeling.configuration_hash(), feature_schema_version=pcfg.features.full_version,
            state_snapshot_id=snapshot_id, feature_state_hash=r["feature_state_hash"],
            source_snapshots={k: str(v) for k, v in components.items()}, prediction_created_at=ts(created),
            prediction_cutoff=ts(cutoff) if cutoff else None, feature_availability=json.loads(r["feature_availability"]),
            freshness_status=fresh.get("status", "unknown"), freshness=fresh,
            provider_match_ids=dict(sch.provider_match_ids),
            inference_engine_version=f"{STATE_ENGINE_VERSION}/{art.model_version}/{art.source}",
            git_commit=git_commit, **o,
        )  # fmt: skip
        out.append((writer.commit_prediction(req), req))
    return out


def predict_and_commit(writer: LedgerWriter, fixtures, *, mode: str, forecast_slot: str, schedules: dict,
                       cutoff: datetime | None = None, git_commit: str | None = None):  # fmt: skip
    """Full Phase 4A path for UpcomingMatch fixtures (state snapshot from the archive + accepted backfill)."""
    from tennis_data.prospective.backfill import load_state_snapshot
    from tennis_data.prospective.freshness import assess, state_coverage
    from tennis_data.prospective.state import compute_state

    _gate(writer, mode)
    pcfg = writer.cfg.prospective
    snapshot = load_state_snapshot(pcfg)
    art = load_production(pcfg)
    st = compute_state(snapshot, list(fixtures), pcfg, keep_history=True)
    cover = state_coverage(st["frame"].filter(~pl.col("match_id").str.starts_with("pending:")))
    fresh = assess(st["pending"], cover, pcfg, now=writer.clock.now())
    return commit_state_rows(
        writer, st["pending"], art, mode=mode, forecast_slot=forecast_slot, snapshot_id=snapshot.snapshot_id,
        components=snapshot.components, schedules=schedules, tournament_ids={f.match_id: f.tournament_id for f in fixtures},
        rounds={f.match_id: f.round for f in fixtures}, best_of={f.match_id: f.best_of for f in fixtures},
        freshness=fresh, cutoff=cutoff, git_commit=git_commit,
    )  # fmt: skip


def verify_receipt(path, receipt: Receipt) -> dict:
    """Independent check that a receipt's event is in the ledger, unchanged, on a valid chain."""
    con = storage.connect(path, read_only=True)
    try:
        m, events = storage.meta(con), storage.all_events(con)
    finally:
        con.close()
    problems = []
    if m["ledger_id"] != receipt.ledger_id:
        problems.append("ledger_id differs")
    if receipt.seq > len(events):
        problems.append("receipt sequence beyond the ledger head")
    else:
        e = events[receipt.seq - 1]
        if (e["event_id"], e["event_hash"], e["previous_hash"], e["recorded_at"]) != (
                receipt.event_id, receipt.event_hash, receipt.previous_hash, receipt.recorded_at):  # fmt: skip
            problems.append("event at the receipt's sequence differs from the receipt")
        rep = verify_events(m, events[: receipt.seq])
        if not rep.ok:
            problems.append(f"chain up to the receipt is invalid: {rep.issues[:3]}")
    return {"ok": not problems, "problems": problems}

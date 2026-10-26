"""Synthetic ledger fixtures: deterministic clock, stable ids, valid prediction requests (no live providers)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from tennis_data.ledger.writer import FixedClock, LedgerWriter, PredictionRequest, load_ledger_config, orient

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)


def ledger_cfg(tmp_path, prospective=None):
    cfg = load_ledger_config(prospective=prospective)
    return replace(cfg, ledger_path=tmp_path / "ledger.sqlite", export_dir=tmp_path / "export",
                   checkpoint_dir=tmp_path / "checkpoints", reports_dir=tmp_path / "reports")  # fmt: skip


def writer(tmp_path, ledger_id="ledger-test", start=T0, prospective=None):
    w = LedgerWriter(ledger_cfg(tmp_path, prospective), FixedClock(start))
    w.init(ledger_id)
    return w


def request(cfg, match_id="synthetic:m1", p1=104925, p2=207989, prob_p1=0.62, elo_p1=0.6, mode="SYNTHETIC",
            slot="pre_match", start=T0 + timedelta(days=1), **over) -> PredictionRequest:  # fmt: skip
    o = orient(match_id, p1, p2, prob_p1, elo_p1)
    base = dict(
        mode=mode, forecast_slot=slot, match_id=match_id, tournament_id="synthetic-open", surface="Hard", round="R32",
        best_of=3, scheduled_start_at=start.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if start else None,
        scheduled_start_verified=start is not None, schedule_snapshot_id="sched-1",
        schedule_retrieved_at="2026-11-01T08:00:00.000000Z", model_id="M4A+raw", model_version="m3b-1+engine1",
        frozen_hash=cfg.prospective.expected_frozen_hash + "0" * 48,
        configuration_hash=cfg.prospective.modeling.configuration_hash(), feature_schema_version="sr-1+engine2",
        state_snapshot_id="state-1", feature_state_hash="fs-1", source_snapshots={"archive": "83733587"},
        prediction_created_at="2026-11-01T08:30:00.000000Z", prediction_cutoff=None, feature_availability={},
        freshness_status="unknown", inference_engine_version="state-1/m3b-1+engine1/production", git_commit="test",
    )  # fmt: skip
    return PredictionRequest(**{**base, **o, **over})


def source(provider="fixture"):
    return {"provider": provider, "snapshot_id": "snap-1", "retrieved_at": "2026-11-02T20:00:00.000000Z"}


def result(winner, status="completed", supersedes=None, reason=None, score="6-4 6-4"):
    return {"status": status, "winner_id": winner, "score": score, "source": source(),
            "supersedes_event_id": supersedes, "correction_reason": reason}  # fmt: skip

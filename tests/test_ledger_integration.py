"""Frozen M4A inference → ledger commitment → receipt, end to end on synthetic state (no live data)."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from conftest import FakeArchive
from ledger_helpers import T0, ledger_cfg
from tennis_data.ledger.reader import LedgerReader
from tennis_data.ledger.service import Schedule, commit_state_rows, predict_and_commit, verify_receipt
from tennis_data.ledger.verification import verify_ledger
from tennis_data.ledger.writer import FixedClock, LedgerWriter, PredictionRejected
from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.inference import load_production

FIX = Path(__file__).parent / "fixtures" / "m4a_production"
GOLDEN = json.loads((FIX / "golden.json").read_text())


@pytest.fixture
def prod_writer(tmp_path):
    d = tmp_path / "modeling"
    shutil.copytree(FIX, d)
    pcfg = load_prospective_config()
    pcfg = replace(pcfg, modeling=replace(pcfg.modeling, modeling_dir=d))  # the real frozen 2026 artifacts
    w = LedgerWriter(ledger_cfg(tmp_path, pcfg), FixedClock(T0))
    w.init("ledger-integration")
    return w, pcfg


def _state(match_ids):
    rows = []
    for mid, r in zip(match_ids, GOLDEN["rows"], strict=False):
        rows.append({"match_id": f"pending:{mid}", "player_a_id": 104925, "player_b_id": 207989, "surface": "Hard",
                     "prediction_batch": date(2026, 10, 31), "predicted_probability_a": r["p"], "spw_diff": r["spw"],
                     "rpw_diff": r["rpw"], "a_spw_decay52w": r["asd"], "b_spw_decay52w": r["bsd"],
                     "a_rpw_decay52w": r["ard"], "b_rpw_decay52w": r["brd"], "a_has_stats_history": True,
                     "b_has_stats_history": True, "a_weeks_since_last_stats": 1, "b_weeks_since_last_stats": 1,
                     "a_service_points_observed": 5000, "b_service_points_observed": 5000,
                     "a_service_ess_decay52w": 900.0, "b_service_ess_decay52w": 900.0,
                     "adjustment_fit_batch": date(2026, 10, 24), "global_match_count_a": 300,
                     "global_match_count_b": 300})  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)


def _schedules(ids, start=T0 + timedelta(days=1)):
    return {m: Schedule(start, True, "sched-snap-1", T0 - timedelta(hours=1), {"fixture": f"ext-{m}"}) for m in ids}


def _commit(w, pcfg, ids, mode="SYNTHETIC"):
    return commit_state_rows(w, _state(ids), load_production(pcfg), mode=mode, forecast_slot="pre_match",
                             snapshot_id="synthetic-state-1", components={"fixture": "m4a_production"},
                             schedules=_schedules(ids), tournament_ids=dict.fromkeys(ids, "synthetic-open"),
                             rounds=dict.fromkeys(ids, "R32"), best_of=dict.fromkeys(ids, 3), git_commit="test")  # fmt: skip


def test_production_m4a_predictions_are_committed_with_receipts(prod_writer):
    w, pcfg = prod_writer
    ids = [f"synthetic:{r['match_id']}" for r in GOLDEN["rows"]]
    out = _commit(w, pcfg, ids)
    assert len(out) == 6 and all(r.status == "committed" for r, _ in out)
    rd = LedgerReader(w.path)
    for (rec, req), g in zip(out, GOLDEN["rows"], strict=True):
        exp = GOLDEN["expected"][g["match_id"]]["probability_a"]  # P(player 104925 wins)
        pa = req.probability_a if req.player_a_id == 104925 else req.probability_b
        assert pa == pytest.approx(exp, abs=1e-12)
        assert abs(req.probability_a + req.probability_b - 1) <= 1e-12
        e = rd.event(rec.event_id)
        assert e["payload"]["prediction"]["frozen_hash"].startswith("bc4e5c870c6e21b9")
        assert e["payload"]["prediction"]["configuration_hash"] == "5e72010100065889"
        assert e["payload"]["match"]["provider_match_ids"] == {"fixture": f"ext-{req.match_id}"}
        assert e["payload"]["audit"]["mode"] == "SYNTHETIC" and e["match_id"].startswith("synthetic:")
        assert verify_receipt(w.path, rec)["ok"]
    again = _commit(w, pcfg, ids)  # identical request later: the existing records are returned
    assert all(r.status == "existing" for r, _ in again)
    assert verify_ledger(w.path).ok and verify_ledger(w.path).events == 6


def test_prospective_requests_are_refused_before_inference(prod_writer):
    w, pcfg = prod_writer
    with pytest.raises(PredictionRejected, match="DISABLED"):
        _commit(w, pcfg, ["tour:2026-9999:1"], mode="PROSPECTIVE")
    assert LedgerReader(w.path).head()[0] == 0


def test_no_receipt_without_a_committed_transaction(prod_writer):
    w, pcfg = prod_writer
    real = w.con

    class Fail:
        def execute(self, sql, *a):
            if sql == "COMMIT":
                raise OSError("disk full")
            return real.execute(sql, *a)

    w.con = Fail()
    with pytest.raises(OSError):
        _commit(w, pcfg, ["synthetic:x"])
    w.con = real
    assert LedgerReader(w.path).head()[0] == 0


def test_tampered_receipt_does_not_verify(prod_writer):
    w, pcfg = prod_writer
    rec, _ = _commit(w, pcfg, ["synthetic:r"])[0]
    assert not verify_receipt(w.path, replace(rec, event_hash="0" * 64))["ok"]
    assert not verify_receipt(w.path, replace(rec, seq=9))["ok"]


def test_full_phase4a_path_on_a_synthetic_world(tmp_path_factory):
    """Synthetic Phase 1-3B world + Phase 4A state snapshot → compute_state → M4A → ledger."""
    from prospective_helpers import build_world
    from tennis_data.prospective.state import UpcomingMatch

    tmp = tmp_path_factory.mktemp("ledger_world")
    (tmp / "archive").mkdir()
    pcfg, _ = build_world(tmp, FakeArchive(tmp / "archive"))
    w = LedgerWriter(ledger_cfg(tmp, pcfg), FixedClock(datetime(2020, 7, 1, tzinfo=UTC)))
    w.init("ledger-world")
    fx = [UpcomingMatch("synthetic:u1", 100, 200, "synthetic-open", date(2020, 8, 3), "A", "Hard", "R32", 32, 3),
          UpcomingMatch("synthetic:u2", 300, 400, "synthetic-open", date(2020, 8, 3), "A", "Clay", "R32", 32, 3)]  # fmt: skip
    sched = {f.match_id: Schedule(datetime(2020, 8, 3, 12, tzinfo=UTC), True, "s1", datetime(2020, 7, 1, tzinfo=UTC), {})
             for f in fx}  # fmt: skip
    out = predict_and_commit(w, fx, mode="SYNTHETIC", forecast_slot="pre_match", schedules=sched, git_commit="test")
    assert len(out) == 2
    for rec, req in out:
        e = LedgerReader(w.path).event(rec.event_id)["payload"]
        assert e["prediction"]["state_snapshot_id"] == req.state_snapshot_id
        assert e["prediction"]["freshness_status"] in ("ok", "warning", "fail")
        assert e["audit"]["timing"]["committed_before_scheduled_start"] is True
    replay = [UpcomingMatch("replay:tour:2019-0005:1", 100, 200, "2019-0005", date(2019, 2, 11), "A", "Hard", "R32", 32,
                            3)]  # fmt: skip
    r = predict_and_commit(w, replay, mode="HISTORICAL_REPLAY", forecast_slot="pre_match",
                           schedules={"replay:tour:2019-0005:1": Schedule(None, False, None, None, {})})  # fmt: skip
    assert r[0][1].mode == "HISTORICAL_REPLAY" and r[0][1].scheduled_start_verified is False
    with pytest.raises(PredictionRejected, match="DISABLED"):
        predict_and_commit(w, fx, mode="PROSPECTIVE", forecast_slot="pre_match", schedules=sched)
    assert verify_ledger(w.path).ok

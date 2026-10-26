"""Prediction controls: idempotency, conflicts, orientation, probabilities, model identity, modes, timing gates."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from ledger_helpers import T0, request, result, writer
from tennis_data.ledger.reader import LedgerReader
from tennis_data.ledger.verification import verify_ledger
from tennis_data.ledger.writer import (
    IdempotencyConflict,
    PredictionRejected,
    SystemClock,
    check_prospective_timing,
    idempotency_key,
)


def test_identical_resubmission_returns_the_existing_record(tmp_path):
    w = writer(tmp_path)
    r1 = w.commit_prediction(request(w.cfg))
    later = request(w.cfg, prediction_created_at="2026-11-01T08:45:00.000000Z", git_commit="other")
    r2 = w.commit_prediction(later)  # only creation time and git commit differ: same forecast
    assert r2.status == "existing" and r2.event_id == r1.event_id and r2.seq == r1.seq
    assert LedgerReader(w.path).head()[0] == 1


def test_conflicting_resubmission_is_rejected_and_logged(tmp_path):
    w = writer(tmp_path)
    r1 = w.commit_prediction(request(w.cfg, prob_p1=0.62))
    with pytest.raises(IdempotencyConflict):
        w.commit_prediction(request(w.cfg, prob_p1=0.70))  # a "better" number for the same slot
    rd = LedgerReader(w.path)
    assert rd.head()[0] == 1
    c = rd.conflicts()
    assert len(c) == 1 and c[0]["existing_event_id"] == r1.event_id
    assert rd.event(r1.event_id)["payload"]["prediction"]["probability_a"] in (0.62, pytest.approx(0.38))


def test_separate_forecast_slots_are_separate_records(tmp_path):
    w = writer(tmp_path)
    a = w.commit_prediction(request(w.cfg, slot="t_minus_24h"))
    b = w.commit_prediction(request(w.cfg, slot="pre_match", prob_p1=0.64))
    assert a.event_id != b.event_id and b.seq == 2
    with pytest.raises(PredictionRejected, match="forecast slot"):
        w.commit_prediction(request(w.cfg, slot="whenever_i_like"))
    assert idempotency_key(request(w.cfg, slot="pre_match")) != idempotency_key(request(w.cfg, slot="t_minus_24h"))


def test_player_orientation_is_enforced(tmp_path):
    w = writer(tmp_path)
    good = request(w.cfg)
    swapped = replace(good, player_a_id=good.player_b_id, player_b_id=good.player_a_id,
                      probability_a=good.probability_b, probability_b=good.probability_a)  # fmt: skip
    with pytest.raises(PredictionRejected, match="orientation"):
        w.commit_prediction(swapped)
    r = w.commit_prediction(good)
    p = LedgerReader(w.path).event(r.event_id)["payload"]
    assert p["audit"]["orientation"] == "phase1-sha256-match-id"


@pytest.mark.parametrize("pa, pb", [(float("nan"), 0.5), (1.2, -0.2), (0.6, 0.5), (0.6, 0.4 + 1e-9), (1, 0)])
def test_invalid_probabilities_are_rejected(tmp_path, pa, pb):
    w = writer(tmp_path)
    with pytest.raises(PredictionRejected):
        w.commit_prediction(replace(request(w.cfg), probability_a=pa, probability_b=pb))
    assert LedgerReader(w.path).head()[0] == 0


def test_model_identity_and_state_are_required(tmp_path):
    w = writer(tmp_path)
    with pytest.raises(PredictionRejected, match="frozen hash"):
        w.commit_prediction(request(w.cfg, frozen_hash="deadbeef" * 8))
    with pytest.raises(PredictionRejected, match="configuration hash"):
        w.commit_prediction(request(w.cfg, configuration_hash="0" * 16))
    with pytest.raises(PredictionRejected, match="state"):
        w.commit_prediction(request(w.cfg, state_snapshot_id=""))


def test_prospective_mode_is_disabled_in_the_writer(tmp_path):
    w = writer(tmp_path)
    assert (w.cfg.prospective.backfill_status, w.cfg.prospective.prospective_predictions) == ("INCOMPLETE", "DISABLED")
    for clock in (w.clock, SystemClock()):
        w.clock = clock
        with pytest.raises(PredictionRejected, match="DISABLED"):
            w.commit_prediction(request(w.cfg, mode="PROSPECTIVE", match_id="tour:2026-9999:1"))
    assert LedgerReader(w.path).head()[0] == 0


@pytest.mark.parametrize("mode, mid", [
    ("SYNTHETIC", "replay:m1"), ("SYNTHETIC", "tour:2026-1:1"), ("HISTORICAL_REPLAY", "synthetic:m1"),
    ("HISTORICAL_REPLAY", "tour:2026-1:1"),
])  # fmt: skip
def test_modes_live_in_separate_namespaces(tmp_path, mode, mid):
    w = writer(tmp_path)
    with pytest.raises(PredictionRejected, match="namespace|match ids"):
        w.commit_prediction(request(w.cfg, mode=mode, match_id=mid))


def test_result_known_blocks_new_synthetic_predictions_but_not_replays(tmp_path):
    w = writer(tmp_path)
    w.commit_prediction(request(w.cfg, match_id="synthetic:x"))
    w.record("MATCH_RESULT_RECORDED", "synthetic:x", result(104925))
    with pytest.raises(PredictionRejected, match="result is already recorded"):
        w.commit_prediction(request(w.cfg, match_id="synthetic:x", slot="t_minus_1h"))
    w.commit_prediction(request(w.cfg, match_id="replay:tour:2010-1:1", mode="HISTORICAL_REPLAY"))
    w.record("MATCH_RESULT_RECORDED", "replay:tour:2010-1:1", result(104925))
    w.commit_prediction(request(w.cfg, match_id="replay:tour:2010-1:1", mode="HISTORICAL_REPLAY", slot="t_minus_1h"))
    assert verify_ledger(w.path).ok


def test_commit_time_comes_from_the_ledger_clock_not_the_caller(tmp_path):
    w = writer(tmp_path)
    r = w.commit_prediction(request(w.cfg, prediction_created_at="2000-01-01T00:00:00.000000Z"))
    e = LedgerReader(w.path).event(r.event_id)
    assert e["recorded_at"] == "2026-11-01T09:00:01.000000Z"
    assert e["payload"]["audit"]["clock"] == "injected"  # test clock: never acceptable for PROSPECTIVE


# ------------------------------------------------------------------- prospective timing (behind the gate)


def _pros(cfg, **over):
    kw = {"prediction_cutoff": "2026-11-02T08:00:00.000000Z", "freshness_status": "ok", **over}
    return request(cfg, mode="PROSPECTIVE", match_id="tour:2026-9999:1", **kw)


def test_prospective_timing_rules(tmp_path):
    cfg = writer(tmp_path).cfg
    now = T0  # 2026-11-01 09:00; scheduled start 2026-11-02 09:00; cutoff 2026-11-02 08:00
    check_prospective_timing(_pros(cfg), now, cfg)
    cases = [
        (_pros(cfg, start=None), now, "no verified scheduled start"),
        (_pros(cfg, scheduled_start_verified=False), now, "no verified scheduled start"),
        (_pros(cfg, prediction_cutoff=None), now, "cutoff"),
        (_pros(cfg, prediction_cutoff="2026-11-02T10:00:00.000000Z"), now, "after the scheduled start"),
        (_pros(cfg), now + timedelta(hours=23, minutes=30), "before the prediction cutoff"),
        (_pros(cfg, prediction_cutoff="2026-11-02T09:00:00.000000Z"), now + timedelta(days=1, minutes=1), "cutoff"),
        (_pros(cfg, prediction_created_at="2026-11-01T10:00:00.000000Z"), now, "future"),
        (_pros(cfg, schedule_retrieved_at="2026-10-30T00:00:00.000000Z"), now, "too old"),
        (_pros(cfg, freshness_status="fail"), now, "freshness"),
    ]  # fmt: skip
    for req, t, msg in cases:
        with pytest.raises(PredictionRejected, match=msg):
            check_prospective_timing(req, t, cfg)

"""Append-only corrections: postponement, cancellation, player replacement, result correction, voiding, timing."""

from __future__ import annotations

import pytest

from ledger_helpers import request, result, source, writer
from tennis_data.ledger.events import EventValidationError
from tennis_data.ledger.export import export, read_jsonl
from tennis_data.ledger.lifecycle import TransitionError
from tennis_data.ledger.reader import LedgerReader
from tennis_data.ledger.verification import verify_ledger

M = "synthetic:life"


def _setup(tmp_path):
    w = writer(tmp_path)
    r = w.commit_prediction(request(w.cfg, match_id=M))
    return w, r, LedgerReader(w.path)


def test_postponement_keeps_the_original_forecast_and_schedule(tmp_path):
    w, r, rd = _setup(tmp_path)
    w.record("MATCH_POSTPONED", M, {"new_scheduled_start_at": "2026-11-03T12:00:00.000000Z", "reason": "rain",
                                    "source": source()})  # fmt: skip
    st = rd.match_state(M)
    assert st.status == "postponed"
    assert [s[1] for s in st.scheduled_start_history] == ["2026-11-02T09:00:00.000000Z", "2026-11-03T12:00:00.000000Z"]
    pred = rd.predictions_for_match(M)[0]
    assert pred["event"]["event_hash"] == r.event_hash and pred["voided"] is None
    assert pred["event"]["payload"]["match"]["scheduled_start_at"] == "2026-11-02T09:00:00.000000Z"


def test_cancellation_is_terminal_and_voiding_needs_evidence(tmp_path):
    w, r, rd = _setup(tmp_path)
    with pytest.raises(TransitionError, match="not supported by its evidence"):
        w.record("PREDICTION_VOIDED", M, {"reason_code": "match_cancelled", "evidence_event_ids": [r.event_id],
                                          "actor": "ops", "note": ""}, ref_event_id=r.event_id)  # fmt: skip
    c = w.record("MATCH_CANCELLED", M, {"reason": "withdrawal before draw", "source": source()})
    with pytest.raises(TransitionError):
        w.record("MATCH_RESULT_RECORDED", M, result(104925))
    with pytest.raises(Exception, match="cancelled"):
        w.commit_prediction(request(w.cfg, match_id=M, slot="t_minus_1h"))
    v = w.record("PREDICTION_VOIDED", M, {"reason_code": "match_cancelled", "evidence_event_ids": [c.event_id],
                                          "actor": "ops-process", "note": "cancelled"}, ref_event_id=r.event_id)  # fmt: skip
    pred = rd.predictions_for_match(M)[0]
    assert pred["voided"]["event_id"] == v.event_id and pred["voided"]["actor"] == "ops-process"
    with pytest.raises(TransitionError, match="already voided"):
        w.record("PREDICTION_VOIDED", M, {"reason_code": "match_cancelled", "evidence_event_ids": [c.event_id],
                                          "actor": "x", "note": ""}, ref_event_id=r.event_id)  # fmt: skip
    export(w.path, tmp_path / "exp")
    _, events = read_jsonl(tmp_path / "exp" / "events.jsonl")
    assert [e["event_type"] for e in events] == ["PREDICTION_COMMITTED", "MATCH_CANCELLED", "PREDICTION_VOIDED"]
    assert verify_ledger(w.path).ok


def test_losing_is_not_a_void_reason(tmp_path):
    w, r, _ = _setup(tmp_path)
    res = w.record("MATCH_RESULT_RECORDED", M, result(207989))
    with pytest.raises(EventValidationError, match="void reason"):
        w.record("PREDICTION_VOIDED", M, {"reason_code": "prediction_lost", "evidence_event_ids": [res.event_id],
                                          "actor": "x", "note": ""}, ref_event_id=r.event_id)  # fmt: skip
    with pytest.raises(TransitionError, match="evidence"):  # a completed result is no evidence of a walkover
        w.record("PREDICTION_VOIDED", M, {"reason_code": "walkover_no_play", "evidence_event_ids": [res.event_id],
                                          "actor": "x", "note": ""}, ref_event_id=r.event_id)  # fmt: skip


def test_player_replacement_supersedes_without_moving_the_forecast(tmp_path):
    w, r, rd = _setup(tmp_path)
    new = "synthetic:life-replacement"
    w.record("MATCH_METADATA_CORRECTED", M, {"kind": "players", "reason": "lucky loser replaced player B",
                                             "source": source(), "changes": {"superseded_by_match_id": new,
                                                                              "player_a_id": 104925,
                                                                              "player_b_id": 200000}})  # fmt: skip
    assert rd.match_state(M).status == "superseded"
    with pytest.raises(Exception, match="superseded"):
        w.commit_prediction(request(w.cfg, match_id=M, slot="t_minus_1h"))
    w.commit_prediction(request(w.cfg, match_id=new, p2=200000))
    assert len(rd.predictions_for_match(M)) == 1 and len(rd.predictions_for_match(new)) == 1
    assert rd.predictions_for_match(M)[0]["event"]["event_hash"] == r.event_hash


def test_result_corrections_preserve_the_original(tmp_path):
    w, r, rd = _setup(tmp_path)
    first = w.record("MATCH_RESULT_RECORDED", M, result(104925, score="6-4 6-4"))
    with pytest.raises(TransitionError, match="supersede"):
        w.record("MATCH_RESULT_RECORDED", M, result(207989))
    with pytest.raises(EventValidationError, match="correction"):
        w.record("MATCH_RESULT_RECORDED", M, result(207989, supersedes=first.event_id))
    fixed = w.record("MATCH_RESULT_RECORDED", M, result(207989, "retired", first.event_id, "source issued correction",
                                                        score="6-4 2-1 RET"))  # fmt: skip
    st = rd.match_state(M)
    assert st.result_event == fixed.event_id and st.result_history == [first.event_id, fixed.event_id]
    assert rd.event(first.event_id)["payload"]["winner_id"] == 104925  # original kept
    assert rd.event(fixed.event_id)["payload"]["source"]["provider"] == "fixture"
    pred = rd.event(r.event_id)["payload"]["prediction"]
    assert pred["probability_a"] in (0.62, pytest.approx(0.38))  # results never touch the prediction


@pytest.mark.parametrize("status, winner, ok", [
    ("abandoned", None, True), ("unknown", None, True), ("walkover", 104925, True), ("default", 104925, True),
    ("abandoned", 104925, False), ("completed", None, False), ("retired", None, False),
])  # fmt: skip
def test_non_binary_outcomes(tmp_path, status, winner, ok):
    w, _, _ = _setup(tmp_path)
    if ok:
        w.record("MATCH_RESULT_RECORDED", M, result(winner, status, score=None))
    else:
        with pytest.raises(EventValidationError):
            w.record("MATCH_RESULT_RECORDED", M, result(winner, status, score=None))


def test_timing_statuses_and_late_actual_start(tmp_path):
    w, r, rd = _setup(tmp_path)  # committed 2026-11-01 09:00:01, scheduled 2026-11-02 09:00
    t = rd.predictions_for_match(M)[0]["timing"]
    assert t == {"generated_before_scheduled_start": True, "committed_before_scheduled_start": True,
                 "scheduled_start_verified": True, "verified_before_actual_start": None, "actual_start_event": None}  # fmt: skip
    a = w.record("MATCH_METADATA_CORRECTED", M, {"kind": "actual_start", "reason": "official start time",
                                                 "source": source(), "changes": {"actual_start_at":
                                                                                 "2026-11-01T09:00:00.500000Z"}})  # fmt: skip
    t2 = rd.predictions_for_match(M)[0]["timing"]
    assert t2["verified_before_actual_start"] is False and t2["committed_before_scheduled_start"] is True
    assert rd.event(r.event_id)["payload"]["audit"]["timing"]["committed_before_scheduled_start"] is True  # unchanged
    w.record("PREDICTION_VOIDED", M, {"reason_code": "committed_after_actual_start", "evidence_event_ids": [a.event_id],
                                      "actor": "timing-audit", "note": ""}, ref_event_id=r.event_id)  # fmt: skip
    assert rd.predictions_for_match(M)[0]["voided"]["reason_code"] == "committed_after_actual_start"

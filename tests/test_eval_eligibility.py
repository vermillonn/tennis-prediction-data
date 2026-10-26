"""Phase 4C: prediction eligibility, selection policy, timing evidence, voids, postponements and mode isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from evaluation_helpers import OPENSSL, T0, LocalTSA, World, outcome_record, scored_match, write_outcomes
from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.demo import build_demo, demo_protocol
from tennis_data.evaluation_prospective.outcomes import empty_outcomes, load_outcomes
from tennis_data.evaluation_prospective.pipeline import evaluate
from tennis_data.evaluation_prospective.snapshot import snapshot_from_ledger
from tennis_data.ledger.hashing import ts
from tennis_data.ledger.writer import PredictionRejected

P1, P2 = 104925, 207989


def _eval(wd, outcomes=None, **kw):
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    oc = load_outcomes(outcomes) if outcomes else empty_outcomes()
    res = evaluate(snap, oc, kw.pop("protocol", wd.cfg.protocol), **kw)
    n = {}
    for r in res["classification"]:
        n[r["match_id"]] = n.get(r["match_id"], 0) + 1
    key = {r["seq"]: r["match_id"] if n[r["match_id"]] == 1 or r["selected"] else f"{r['match_id']}#{r['seq']}"
           for r in res["classification"]}  # fmt: skip
    return snap, {key[r["seq"]]: r for r in res["classification"]}, res


def test_prospective_pre_start_forecast_is_operational_not_primary_without_timestamp(tmp_path):
    wd = World(tmp_path)
    scored_match(wd, "m-1", T0 + timedelta(days=1))
    _, rows, res = _eval(wd)
    r = rows["m-1"]
    assert r["category"] == "SCORABLE" and r["timing"] == "PRE_START" and r["evidence_level"] == "LOCAL_ONLY"
    assert r["cohort"] == "operational" and res["scores"]["primary"]["n"] == 0
    assert res["scores"]["operational"]["n"] == 1


@pytest.mark.skipif(OPENSSL is None, reason="openssl binary not installed")
def test_primary_cohort_needs_a_timestamp_before_the_actual_start(tmp_path):
    wd = World(tmp_path)
    scored_match(wd, "m-1", T0 + timedelta(days=1))
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    tsa = LocalTSA(tmp_path / "tsa")
    cp = anchoring.checkpoint_from_snapshot(snap)
    rec, _ = anchoring.publish(cp, anchoring.Rfc3161Publisher("https://tsa.test/", [tsa.ca], tsa.transport),
                               tmp_path / "ev", "ci", now=T0)  # fmt: skip
    common = dict(anchor_records=[("r.json", rec)], evidence_dir=tmp_path / "ev")
    res = evaluate(snap, empty_outcomes(), wd.cfg.protocol, tsa_ca_files=[tsa.ca], **common)
    r = res["classification"][0]
    assert r["evidence_level"] == "INDEPENDENTLY_TIMESTAMPED" and r["cohort"] == "primary"
    assert res["scores"]["primary"]["n"] == 1
    # without the verifier trusting the TSA, the same forecast is only operational
    res = evaluate(snap, empty_outcomes(), wd.cfg.protocol, **common)
    assert res["classification"][0]["cohort"] == "operational" and res["scores"]["primary"]["n"] == 0


@pytest.mark.skipif(OPENSSL is None, reason="openssl binary not installed")
def test_timestamp_after_actual_start_does_not_prove_pre_match(tmp_path):
    from datetime import UTC, datetime

    wd = World(tmp_path)
    past = datetime(2026, 1, 5, 9, tzinfo=UTC)  # the real TSA time (now) is after this synthetic match
    wd.clock.t = past - timedelta(days=2)
    scored_match(wd, "m-old", past)
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    tsa = LocalTSA(tmp_path / "tsa")
    rec, _ = anchoring.publish(anchoring.checkpoint_from_snapshot(snap),
                               anchoring.Rfc3161Publisher("https://tsa.test/", [tsa.ca], tsa.transport),
                               tmp_path / "ev", "ci", now=past)  # fmt: skip
    res = evaluate(snap, empty_outcomes(), wd.cfg.protocol, anchor_records=[("r.json", rec)],
                   evidence_dir=tmp_path / "ev", tsa_ca_files=[tsa.ca])  # fmt: skip
    r = res["classification"][0]
    assert r["evidence_level"] == "LOCAL_ONLY" and "timestamp_not_before_actual_start" in r["flags"]
    assert r["cohort"] == "operational"


def test_synthetic_and_replay_never_enter_the_prospective_cohort(tmp_path):
    from evaluation_helpers import eval_cfg

    cfg = eval_cfg(tmp_path)
    d = build_demo(cfg, tmp_path / "demo", weeks=4, per_week=6, replay_matches=5)
    snap = snapshot_from_ledger(d["ledger"], tmp_path / "snaps")
    oc = load_outcomes(d["outcomes"])
    live = evaluate(snap, oc, cfg.protocol, anchor_records=anchoring.load_records(d["evidence"], 2**20),
                    evidence_dir=d["evidence"], fixture_root=d["fixture_root"])  # fmt: skip
    cats = {r["category"] for r in live["classification"]}
    assert cats == {"SYNTHETIC", "HISTORICAL_REPLAY"}
    assert live["scores"]["primary"]["n"] == live["scores"]["operational"]["n"] == 0
    demo = evaluate(snap, oc, demo_protocol(cfg), anchor_records=anchoring.load_records(d["evidence"], 2**20),
                    evidence_dir=d["evidence"], fixture_root=d["fixture_root"])  # fmt: skip
    assert demo["scores"]["operational"]["n"] > 0 and demo["scores"]["primary"]["n"] == 0
    assert all(r["category"] == "HISTORICAL_REPLAY" for r in demo["classification"] if r["mode"] == "HISTORICAL_REPLAY")
    assert all(r["mode"] == "SYNTHETIC" for r in demo["classification"] if r["cohort"])


def test_gate_disabled_refuses_prospective_records(tmp_path):
    from evaluation_helpers import eval_cfg
    from tennis_data.ledger.writer import LedgerWriter

    cfg = eval_cfg(tmp_path)  # the repository's real gate: DISABLED
    assert cfg.ledger.prospective.prospective_predictions == "DISABLED"
    assert cfg.ledger.prospective.backfill_status == "INCOMPLETE"
    wd = World(tmp_path / "x")
    w = LedgerWriter(cfg.ledger, wd.clock)
    w.init("gate")
    with pytest.raises(PredictionRejected, match="DISABLED"):
        w.commit_prediction(replace(_req(wd), mode="PROSPECTIVE"))


def _req(wd):
    from tennis_data.ledger.writer import PredictionRequest, orient

    o = orient("m-g", P1, P2, 0.6, 0.55)
    return PredictionRequest(mode="PROSPECTIVE", forecast_slot="pre_match", match_id="m-g", tournament_id="t",
                             surface="Hard", round="R32", best_of=3, scheduled_start_at=ts(T0), scheduled_start_verified=True,
                             schedule_snapshot_id="s", schedule_retrieved_at=ts(T0 - timedelta(hours=2)), model_id="M4A+raw",
                             model_version="m", frozen_hash=wd.cfg.ledger.prospective.expected_frozen_hash + "0" * 48,
                             configuration_hash=wd.cfg.protocol["approved_configuration_hash"], feature_schema_version="f",
                             state_snapshot_id="s", feature_state_hash="h", source_snapshots={},
                             prediction_created_at=ts(T0 - timedelta(hours=1)), prediction_cutoff=ts(T0 - timedelta(minutes=5)),
                             feature_availability={}, freshness_status="ok", **o)  # fmt: skip


def test_late_commitment_and_missing_actual_start(tmp_path):
    wd = World(tmp_path)
    s1, s2, s3 = (T0 + timedelta(days=i) for i in (1, 2, 3))
    # committed before the scheduled start, but the match actually began earlier (start brought forward)
    wd.predict("late", s1, s1 - timedelta(hours=2))
    wd.start(s1 - timedelta(hours=1), "late", s1 - timedelta(hours=3))
    wd.result(s1 + timedelta(hours=2), "late", P1)
    wd.predict("nostart", s2, s2 - timedelta(hours=5))  # outcome known, no actual-start evidence
    wd.result(s2 + timedelta(hours=3), "nostart", P1)
    wd.predict("pending", s3, s3 - timedelta(hours=5))  # no outcome yet
    _, rows, res = _eval(wd)
    assert rows["late"]["category"] == "LATE_COMMITMENT" and rows["late"]["timing"] == "LATE"
    assert rows["nostart"]["category"] == "TIMING_UNRESOLVED" and rows["nostart"]["timing"] == "UNRESOLVED"
    assert rows["pending"]["category"] == "OUTCOME_PENDING" and rows["pending"]["admissible"]
    assert res["scores"]["operational"]["n"] == 0
    assert {u["match_id"] for u in res["unresolved_outcomes"]} == {"pending"}


def test_actual_start_from_outcome_snapshot_and_earliest_source_wins(tmp_path):
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    wd.predict("m", s, s - timedelta(hours=5))
    wd.start(s + timedelta(minutes=30), "m", s + timedelta(minutes=20))
    wd.result(s + timedelta(hours=3), "m", P1)
    # another source says the match began before the commitment
    oc = write_outcomes(tmp_path / "o.jsonl", [outcome_record("m", "feed_b", "completed", P1, (P1, P2),
                                                              start=s - timedelta(hours=6))])  # fmt: skip
    _, rows, _ = _eval(wd, oc)
    assert rows["m"]["category"] == "LATE_COMMITMENT" and rows["m"]["actual_start_at"] == ts(s - timedelta(hours=6))


def test_selection_is_first_admissible_forecast_never_the_best(tmp_path):
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    wd.predict("m", s, s - timedelta(hours=10), prob=0.30, sched="v1")  # loses (A wins with p=0.3 -> bad forecast)
    wd.predict("m", s, s - timedelta(hours=5), prob=0.95, sched="v2")  # re-issued schedule; would score better
    wd.predict("m", s, s - timedelta(hours=1, minutes=5), prob=0.97, slot="t_minus_1h")
    wd.start(s + timedelta(minutes=10), "m", s + timedelta(minutes=5))
    wd.result(s + timedelta(hours=3), "m", P1)
    _, rows, res = _eval(wd)
    sel = rows["m"]
    assert sel["probability_a"] in (0.30, 0.70) and sel["category"] == "SCORABLE"
    others = sorted(r["category"] for k, r in rows.items() if k != "m")
    assert others == ["DUPLICATE_SUPERSEDED", "NOT_PRIMARY_SLOT"]
    assert res["scores"]["operational"]["n"] == 1


def test_voided_selection_is_not_replaced(tmp_path):
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    first = wd.predict("m", s, s - timedelta(hours=10), sched="v1")
    wd.predict("m", s, s - timedelta(hours=5), sched="v2")
    canc = wd.record(s - timedelta(hours=2), "MATCH_CANCELLED", "m", {"reason": "rain", "source": {"provider": "f"}})
    wd.record(s - timedelta(hours=1), "PREDICTION_VOIDED", "m", {"reason_code": "match_cancelled",
                                                                 "evidence_event_ids": [canc.event_id], "actor": "op",
                                                                 "note": ""}, ref=first.event_id)  # fmt: skip
    _, rows, res = _eval(wd)
    assert rows["m"]["category"] == "VOIDED" and rows["m"]["voided"]["reason_code"] == "match_cancelled"
    assert [r["category"] for k, r in rows.items() if k != "m"] == ["DUPLICATE_SUPERSEDED"]
    assert res["voids"][0]["event_id"] == first.event_id and res["scores"]["operational"]["n"] == 0


def test_postponed_cancelled_and_players_changed(tmp_path):
    wd = World(tmp_path)
    s1, s2, s3 = (T0 + timedelta(days=3 * i) for i in (1, 2, 3))
    wd.predict("post", s1, s1 - timedelta(hours=10))
    wd.record(s1 - timedelta(hours=1), "MATCH_POSTPONED", "post", {
        "new_scheduled_start_at": ts(s1 + timedelta(days=1)), "reason": "rain", "source": {"provider": "f"}})  # fmt: skip
    wd.start(s1 + timedelta(days=1, minutes=10), "post", s1 + timedelta(days=1, minutes=2))
    wd.result(s1 + timedelta(days=1, hours=3), "post", P2)
    wd.predict("canc", s2, s2 - timedelta(hours=10))
    wd.record(s2 - timedelta(hours=1), "MATCH_CANCELLED", "canc", {"reason": "x", "source": {"provider": "f"}})
    wd.predict("swap", s3, s3 - timedelta(hours=10))
    wd.record(s3 - timedelta(hours=1), "MATCH_METADATA_CORRECTED", "swap", {
        "kind": "players", "changes": {"superseded_by_match_id": "swap-2", "player_a_id": P1, "player_b_id": 5},
        "reason": "lucky loser", "source": {"provider": "f"}})  # fmt: skip
    _, rows, _ = _eval(wd)
    assert rows["post"]["category"] == "SCORABLE" and rows["post"]["scheduled_start_at"] == ts(s1)  # not rewritten
    assert rows["canc"]["category"] == "MATCH_CANCELLED"
    assert rows["swap"]["category"] == "MATCH_SUPERSEDED"


def test_model_identity_readiness_and_cutoff(tmp_path):
    wd = World(tmp_path)
    s1, s2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    scored_match(wd, "other-model", s1, model_id="M3+raw")
    scored_match(wd, "m", s2)
    _, rows, _ = _eval(wd)
    assert rows["other-model"]["category"] == "MODEL_NOT_APPROVED"
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    early = evaluate(snap, empty_outcomes(), wd.cfg.protocol, cutoff=ts(s2 + timedelta(hours=1)))
    r = {x["match_id"]: x for x in early["classification"]}["m"]
    assert r["category"] == "OUTCOME_PENDING"  # the result was recorded after the cutoff


def test_synthetic_readiness_failure(tmp_path):
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    wd.predict("synthetic:fresh-fail", s, s - timedelta(hours=5), mode="SYNTHETIC", freshness_status="fail")
    wd.start(s + timedelta(minutes=10), "synthetic:fresh-fail", s + timedelta(minutes=5))
    wd.result(s + timedelta(hours=3), "synthetic:fresh-fail", P1)
    _, rows, _ = _eval(wd, protocol={**wd.cfg.protocol, "cohort_mode": "SYNTHETIC"})
    r = rows["synthetic:fresh-fail"]
    assert r["category"] == "NOT_OPERATIONALLY_READY" and "readiness:freshness_fail" in r["flags"]

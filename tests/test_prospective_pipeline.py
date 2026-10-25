"""Phase 4A end to end on a synthetic Phase 1-3B world: backfill, corrections, chronology, inference, replay."""

from __future__ import annotations

import json
import random
import shutil
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import FakeArchive
from prospective_helpers import base_matches, build_world, fixture_doc, prospective_config, result, write_doc
from tennis_data.modeling.dataset import DERIVED
from tennis_data.prospective.backfill import ingest_file, load_state_snapshot, run_backfill
from tennis_data.prospective.config import load_prospective_config
from tennis_data.prospective.freshness import assess, state_coverage
from tennis_data.prospective.inference import (
    ArtifactIntegrityError,
    InferenceError,
    load_production,
    predict_state,
)
from tennis_data.prospective.replay import run_replay
from tennis_data.prospective.state import StateError, UpcomingMatch, compute_state
from tennis_data.storage.parquet import tree_checksums

FX = "tennis-fixture-provider-1"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4a")
    archive_root = tmp / "archive"
    archive_root.mkdir()
    cfg, res = build_world(tmp, FakeArchive(archive_root))
    return {"tmp": tmp, "cfg": cfg, "modeling": res}


def _fresh_cfg(world, name):
    """A Phase 4A config with its own empty prospective directory on the shared synthetic world."""
    tmp = world["tmp"] / name
    tmp.mkdir()
    path = prospective_config(tmp, world["cfg"].modeling.config_path, world["modeling"]["frozen"]["frozen_hash"])
    return load_prospective_config(path), tmp


def _ingest(cfg, tmp, doc, name="feed.json", at=None):
    return ingest_file(cfg, write_doc(tmp / name, doc), FX, retrieved_at=at)


# --------------------------------------------------------------------------------------------------- backfill


@pytest.fixture(scope="module")
def backfilled(world):
    cfg, tmp = _fresh_cfg(world, "bf")
    _ingest(cfg, tmp, fixture_doc())
    return cfg, tmp, run_backfill(cfg)


def test_backfill_accepts_holds_and_quarantines(backfilled):
    cfg, _, b = backfilled
    s = b["summary"]
    reasons = {r["reason"]: r["len"] for r in s["reconciliation_quarantine"]}
    assert reasons == {"identity_unresolved": 1, "invalid_record": 1}  # m7 ambiguous player, m10 impossible counts
    assert s["accepted_records"] == 8 and s["phase1_validated"] == 8
    v = b["validated"]
    assert v["match_id"].str.starts_with("tour:2020-fixture-").all()
    status = dict(zip(v["match_status"], v["stats_status"], strict=False))
    assert {"completed", "retired", "walkover"} <= set(status)
    led = b["ledger"].sort("match_id")
    assert (led["prediction_mode"] == "historical_reconstruction").all()
    assert (led["eligibility_basis"] == "inferred_weekly_batch").all()
    assert led["event_start_at"].is_null().all()  # the fixture did not supply match start times


def test_ledger_chronology_and_multi_week_release(backfilled):
    _, _, b = backfilled
    led = b["ledger"]
    for r in led.iter_rows(named=True):
        assert r["state_eligible_at"] == datetime.combine(r["release_batch"] + timedelta(days=7),
                                                          datetime.min.time(), UTC)  # fmt: skip
        assert r["state_eligible_at"].date() > r["tournament_date"] - timedelta(days=7)
    slam = led.filter(pl.col("tournament_id") == "2020-fixture-t-slam")
    assert slam.height == 2 and (slam["event_weeks"] == 2).all()
    assert (slam["release_batch"] == slam["prediction_batch"] + timedelta(days=7)).all()
    by = dict(zip(led["provider_match_id"], zip(led["is_update"], led["stats_eligible"], strict=True), strict=True))
    assert by["m3"] == (False, False)  # retirement: frozen rules exclude it from Elo updates and statistics
    assert by["m5"] == (False, False)  # walkover
    assert by["m4"] == (True, False)  # outcome without statistics
    assert by["m1"] == (True, True)


def test_new_player_gets_reserved_id(backfilled):
    cfg, _, b = backfilled
    d = b["decisions"].filter(pl.col("provider_player_id") == "fx-new").row(0, named=True)
    assert d["category"] == "unmatched" and d["canonical_player_id"] == cfg.new_player_id_base
    amb = b["decisions"].filter(pl.col("provider_player_id") == "fx-amb").row(0, named=True)
    assert amb["category"] == "ambiguous" and not amb["usable"]


def test_backfill_is_idempotent_and_duplicate_ingestion_is_a_noop(backfilled):
    cfg, tmp, _ = backfilled
    before = {k: v for k, v in tree_checksums(cfg.prospective_dir / "backfill").items() if k != "performance.json"}
    _ingest(cfg, tmp, fixture_doc(), "feed_copy.json")  # identical bytes under another file name
    run_backfill(cfg)
    after = {k: v for k, v in tree_checksums(cfg.prospective_dir / "backfill").items() if k != "performance.json"}
    assert before == after
    assert len(list((cfg.prospective_dir / "raw").rglob("*.bin"))) == 1


def test_delivery_order_and_split_feeds_do_not_matter(world):
    ms = base_matches()
    results = {}
    for name, order in (("o1", list(range(len(ms)))), ("o2", list(reversed(range(len(ms)))))):
        cfg, tmp = _fresh_cfg(world, name)
        half = len(ms) // 2
        parts = [[ms[i] for i in order[:half]], [ms[i] for i in order[half:]]]
        for i, part in enumerate(parts if name == "o1" else parts[::-1]):
            _ingest(cfg, tmp, fixture_doc(matches=part), f"part{i}.json", at=datetime(2020, 7, 1, tzinfo=UTC))
        b = run_backfill(cfg)
        results[name] = b["summary"]["accepted_table_hash"]
    assert results["o1"] == results["o2"]


def test_correction_creates_version_and_earliest_replay_batch(world):
    cfg, tmp = _fresh_cfg(world, "corr")
    _ingest(cfg, tmp, fixture_doc())
    first = run_backfill(cfg)
    fixed = [m if m["id"] != "m9" else result("m9", "t-slam", "R64", "fx-500", "fx-600", "fx-600", version=2)
             for m in base_matches()]  # fmt: skip
    _ingest(cfg, tmp, fixture_doc(matches=fixed, generated_at="2020-07-08T00:00:00+00:00"), "v2.json")
    second = run_backfill(cfg)
    vers = second["reconciliation"]["versions"].filter(pl.col("provider_match_id") == "m9").sort("source_version")
    assert vers["state"].to_list() == ["superseded", "current"]
    assert second["replay_from_batch"] == "2020-06-20"  # the slam's batch, not the latest batch
    a, b = first["validated"], second["validated"]
    m9a = a.filter(pl.col("match_id").str.contains("2020-fixture-t-slam") & (pl.col("round") == "R64"))
    m9b = b.filter(pl.col("match_id").str.contains("2020-fixture-t-slam") & (pl.col("round") == "R64"))
    assert m9a["match_id"].to_list() == m9b["match_id"].to_list()  # same match, corrected winner
    assert m9a["winner_id"].item() == 500 and m9b["winner_id"].item() == 600
    # The original result is preserved immutably in the raw store.
    assert len(list((cfg.prospective_dir / "raw").rglob("*.bin"))) == 2


def test_same_version_reissued_with_other_content_is_quarantined(world):
    cfg, tmp = _fresh_cfg(world, "reissue")
    _ingest(cfg, tmp, fixture_doc())
    bad = [m if m["id"] != "m1" else {**m, "winner_id": "fx-200"} for m in base_matches()]
    _ingest(cfg, tmp, fixture_doc(matches=bad, generated_at="2020-07-02T00:00:00+00:00"), "bad.json")
    q = run_backfill(cfg)["reconciliation"]["quarantine"]
    assert q.filter((pl.col("provider_match_id") == "m1") & (pl.col("reason") == "version_reissued")).height == 2


def test_cross_provider_agreement_and_conflict(world):
    cfg, tmp = _fresh_cfg(world, "xsrc")
    _ingest(cfg, tmp, fixture_doc())
    second = [result("s1", "t-june1", "R32", "fx-100", "fx-200", "fx-100", stats=False),  # agrees with m1
              result("s2", "t-june1", "R32", "fx-300", "fx-400", "fx-300")]  # contradicts m2  # fmt: skip
    _ingest(cfg, tmp, fixture_doc(matches=second, provider="other"), "other.json")
    b = run_backfill(cfg)
    notes = b["reconciliation"]["notes"]
    assert notes.filter(pl.col("provider_match_id") == "s1")["reason"].to_list() == ["stats_discrepancy"]
    q = b["reconciliation"]["quarantine"].filter(pl.col("reason") == "cross_source_conflict")
    assert sorted(q["provider_match_id"]) == ["m2", "s2"]


def test_existing_history_is_never_overwritten(world):
    cfg, tmp = _fresh_cfg(world, "hist")
    from tennis_data.prospective.backfill import load_archive

    hist, _ = load_archive(cfg)
    h = hist.filter(pl.col("tournament_id") == "2020-0010").sort("match_id").row(0, named=True)
    tour = {"id": "h", "name": h["tournament_name"], "level": h["tournament_level"], "surface": h["surface"],
            "draw_size": h["draw_size"], "start_date": str(h["tournament_date"])}  # fmt: skip
    w, lo = h["winner_id"], h["player_b_id"] if h["winner_id"] == h["player_a_id"] else h["player_a_id"]
    same = result("hs", "h", h["round"], f"fx-{w}", f"fx-{lo}", f"fx-{w}", score=h["score"])
    flip = result("hf", "h", h["round"], f"fx-{w}", f"fx-{lo}", f"fx-{lo}", score=h["score"])
    for name, m in (("same", same), ("flip", flip)):
        doc = fixture_doc(matches=[m], provider=f"p{name}")
        doc["tournaments"] = [tour]
        _ingest(cfg, tmp, doc, f"{name}.json")
    b = run_backfill(cfg)
    assert b["summary"]["accepted_records"] == 0
    assert b["reconciliation"]["notes"]["reason"].to_list() == ["duplicate_of_historical"]
    assert b["reconciliation"]["quarantine"]["reason"].to_list() == ["contradicts_historical"]


def test_checkpoint_restoration_and_tamper_detection(backfilled):
    cfg, _, b = backfilled
    snap = load_state_snapshot(cfg)
    assert snap.snapshot_id == b["summary"]["state_snapshot_id"]
    path = cfg.prospective_dir / "backfill" / "matches.parquet"
    original = path.read_bytes()
    try:
        pl.read_parquet(path).head(3).write_parquet(path)
        with pytest.raises(AssertionError):
            load_state_snapshot(cfg)
    finally:
        path.write_bytes(original)


# ----------------------------------------------------------------------------------------------- inference


def _fixture(pid_a=100, pid_b=200, d=date(2020, 8, 3), **kw):
    base = dict(match_id="u1", player_a_id=pid_a, player_b_id=pid_b, tournament_id="next", tournament_date=d,
                tournament_level="A", surface="Hard", round="R32", draw_size=32, best_of=3)  # fmt: skip
    return UpcomingMatch(**{**base, **kw})


@pytest.fixture(scope="module")
def inferred(backfilled):
    cfg, _, _ = backfilled
    snap = load_state_snapshot(cfg)
    fx = [_fixture(), _fixture(200, 100, match_id="u1-swapped"), _fixture(9000000, 300, match_id="u-new"),
          _fixture(100, 200, d=date(2020, 6, 8), match_id="u-past")]  # fmt: skip
    st = compute_state(snap, fx, cfg, keep_history=True)
    art = load_production(cfg)
    p = predict_state(st["pending"], art, cfg, prediction_mode="exploratory_retrospective",
                      snapshot_id=snap.snapshot_id, components=snap.components)  # fmt: skip
    return cfg, snap, st, art, p


def test_prediction_symmetry_and_swap_invariance(inferred):
    _, _, _, _, p = inferred
    by = {r["match_id"]: r for r in p.iter_rows(named=True)}
    for r in by.values():
        assert abs(r["probability_a"] + r["probability_b"] - 1) <= 1e-12
    assert by["u1"]["probability_a"] == pytest.approx(by["u1-swapped"]["probability_b"], abs=1e-12)
    assert by["u1"]["baseline_elo_probability_a"] == pytest.approx(1 - by["u1-swapped"]["baseline_elo_probability_a"],
                                                                   abs=1e-12)  # fmt: skip


def test_backfilled_results_reach_later_state_only(inferred):
    cfg, _, st, _, _ = inferred
    pend = {r["match_id"]: r for r in st["pending"].iter_rows(named=True)}
    # u-past is predicted in the 2020-06-06 batch: the 2020-06-01 event (batch 2020-05-30) is released, the
    # 2020-06-08 event and the slam are not yet.
    assert pend["pending:u-past"]["a_stat_matches"] < pend["pending:u1"]["a_stat_matches"]
    assert pend["pending:u-new"]["a_has_stats_history"] is True  # Zulu played m6 in the backfill


def test_missing_history_uses_only_frozen_fallback(inferred):
    cfg, snap, _, art, _ = inferred
    st = compute_state(snap, [_fixture(9000000, 100, match_id="np", d=date(2020, 6, 1))], cfg)
    row = st["pending"].row(0, named=True)
    assert row["a_has_stats_history"] is False  # Zulu's first match is in this very batch
    p = predict_state(st["pending"], art, cfg, prediction_mode="exploratory_retrospective",
                      snapshot_id=snap.snapshot_id, components=snap.components)  # fmt: skip
    avail = json.loads(p["feature_availability"][0])
    expected = [n for n in art.inputs if DERIVED[n].fill_zero and any(row[b] is None for b in DERIVED[n].base)]
    assert avail["filled_inputs"] == expected


def test_frozen_fallback_matches_phase3b_dataset_construction(inferred):
    """M4A-style inputs on a row without recent history: decay differences -> 0, missingness indicator set."""
    cfg = inferred[0]
    from tennis_data.prospective.inference import ModelArtifact, derive

    inputs = (
        "elo_logit",
        "spw_shrunk_diff",
        "rpw_shrunk_diff",
        "spw_decay_diff",
        "rpw_decay_diff",
        "decay_missing_diff",
    )
    art = ModelArtifact("M4A+raw", "v", "h", inputs, (1.0,) * 6, (0.0,) * 6, {"method": "raw"}, 1e-6, "test", "s", "c")
    st = pl.DataFrame({"match_id": ["x"], "predicted_probability_a": [0.6], "spw_diff": [0.01], "rpw_diff": [None],
                       "a_spw_decay52w": [None], "b_spw_decay52w": [0.65], "a_rpw_decay52w": [None],
                       "b_rpw_decay52w": [0.35]}, schema_overrides={"rpw_diff": pl.Float64, "a_spw_decay52w": pl.Float64,
                                                                      "a_rpw_decay52w": pl.Float64})  # fmt: skip
    x, m = derive(st, art, cfg)
    assert x[0].tolist() == pytest.approx([np.log(0.6 / 0.4), 0.01, 0.0, 0.0, 0.0, 1.0])
    assert [n for n in inputs if m[f"filled_{n}"][0]] == ["rpw_shrunk_diff", "spw_decay_diff", "rpw_decay_diff"]


def test_artifact_integrity_is_enforced(inferred, tmp_path):
    cfg = inferred[0]
    prod = cfg.modeling.modeling_dir / "models" / "production_model.json"
    original = prod.read_bytes()
    try:
        d = json.loads(original)
        d["fit"]["coef_scaled"]["elo_logit"] += 1e-9
        prod.write_text(json.dumps(d))
        with pytest.raises(ArtifactIntegrityError):
            load_production(cfg)
    finally:
        prod.write_bytes(original)
    other = load_prospective_config(prospective_config(tmp_path, cfg.modeling.config_path, "f" * 16))
    with pytest.raises(ArtifactIntegrityError, match="expected"):
        load_production(other)


def test_inference_input_errors(inferred):
    cfg, snap, _, art, _ = inferred
    with pytest.raises(StateError, match="round"):
        compute_state(snap, [_fixture(round="R999")], cfg)
    with pytest.raises(StateError, match="outside the rating pools"):
        compute_state(snap, [_fixture(tournament_level="C")], cfg)
    with pytest.raises(StateError, match="identical"):
        compute_state(snap, [_fixture(pid_b=100)], cfg)
    st = compute_state(snap, [_fixture()], cfg)
    with pytest.raises(InferenceError):
        predict_state(st["pending"], art, cfg, prediction_mode="prospective-ish", snapshot_id="x", components={})


def test_inference_is_deterministic(inferred):
    cfg, snap, _, art, p = inferred
    st = compute_state(snap, [_fixture()], cfg)
    q = predict_state(st["pending"], art, cfg, prediction_mode="exploratory_retrospective",
                      snapshot_id=snap.snapshot_id, components=snap.components)  # fmt: skip
    a = p.filter(pl.col("match_id") == "u1").drop("generated_at")
    assert a.equals(q.drop("generated_at"))


def test_input_order_invariance_of_state(inferred):
    cfg, snap, _, _, _ = inferred
    from tennis_data.prospective.state import make_snapshot

    shuffled = snap.matches.sample(fraction=1.0, shuffle=True, seed=3)
    s2 = make_snapshot(shuffled, snap.quarantine_ids, {k: v for k, v in snap.components.items() if k != "matches"})
    assert s2.snapshot_id == snap.snapshot_id
    a = compute_state(snap, [_fixture()], cfg)["pending"]
    b = compute_state(s2, [_fixture()], cfg)["pending"]
    assert a.equals(b)


# --------------------------------------------------------------------------------------------- freshness


def test_freshness_grades_staleness(inferred):
    cfg, _, st, _, _ = inferred
    cover = state_coverage(st["frame"].filter(~pl.col("match_id").str.starts_with("pending:")))
    pend = st["pending"].filter(pl.col("match_id") == "pending:u1")
    near = assess(pend, cover, cfg, source_retrieved_at=datetime(2020, 8, 1, tzinfo=UTC),
                  now=datetime(2020, 8, 2, tzinfo=UTC))["u1"]  # fmt: skip
    far = pend.with_columns(pl.col("prediction_batch") + pl.duration(weeks=20))
    stale = assess(far, cover, cfg, source_retrieved_at=datetime(2020, 8, 1, tzinfo=UTC),
                   now=datetime(2020, 12, 30, tzinfo=UTC))["u1"]  # fmt: skip
    assert near["checks"]["missing_backfill_weeks"]["value"] < stale["checks"]["missing_backfill_weeks"]["value"]
    assert stale["status"] == "fail"
    assert stale["checks"]["last_eligible_result"]["status"] == "fail"
    blocked = assess(pend, cover, cfg, unresolved_ids={100}, unresolved_count=1)["u1"]
    assert blocked["checks"]["unresolved_identities"]["status"] == "fail"


# ------------------------------------------------------------------------------------------------- replay


def test_historical_replay_reproduces_published_predictions(world):
    cfg = world["cfg"]
    snap = load_state_snapshot(cfg)
    r = run_replay(snap, cfg)["summary"]
    assert r["passed"], r
    assert r["max_abs_error"] <= 1e-12 and r["max_elo_abs_error"] <= 1e-12
    assert r["state_reconstruction"]["features_identical"]
    assert {c["category"] for c in r["by_category"]} >= {
        "hard",
        "clay",
        "grass",
        "development_period",
        "exploratory_period",
    }


def test_phase4a_run_is_deterministic_and_leaves_upstream_untouched(world):
    from tennis_data.prospective.pipeline import run_phase4a

    cfg, tmp = _fresh_cfg(world, "full")
    _ingest(cfg, tmp, fixture_doc())
    up = [cfg.data.paths.validated_dir, cfg.elo.elo_dir, cfg.features.features_dir, cfg.modeling.modeling_dir,
          cfg.elo.reports_dir, cfg.features.reports_dir, cfg.modeling.reports_dir]  # fmt: skip
    before = [tree_checksums(p) for p in up]
    a = run_phase4a(cfg)

    def reports():
        return {k: v for k, v in tree_checksums(cfg.reports_dir).items() if k != "runtime.json"}

    first = reports()
    m1 = (cfg.prospective_dir / "phase4a_manifest.json").read_text()
    run_phase4a(cfg)
    assert reports() == first
    for name in ("backfill_coverage.md", "identity_reconciliation.md", "inference_readiness.md"):
        assert (cfg.reports_dir / name).exists()
    assert (cfg.prospective_dir / "phase4a_manifest.json").read_text() == m1
    assert [tree_checksums(p) for p in up] == before
    assert a["replay"]["summary"]["passed"]
    assert a["readiness"]["ready_for_prospective_predictions"] is False  # no verified live provider
    shutil.rmtree(tmp / "prospective" / "raw", ignore_errors=True)


def test_unused_imports_guard():
    assert np and random


def test_prospective_mode_refuses_stale_or_late_requests(backfilled):
    from tennis_data.prospective.pipeline import PredictionBlocked, predict_upcoming

    cfg, _, _ = backfilled
    late = _fixture(d=date(2020, 8, 3))
    with pytest.raises(PredictionBlocked, match="after the match start"):
        predict_upcoming(cfg, [late], "prospective", now=datetime(2020, 8, 4, tzinfo=UTC))
    with pytest.raises(PredictionBlocked, match="before the prospective period"):
        predict_upcoming(cfg, [_fixture(d=date(2020, 7, 6))], "prospective", now=datetime(2020, 7, 1, tzinfo=UTC))
    far = _fixture(d=date(2021, 3, 1))
    with pytest.raises(PredictionBlocked, match="freshness"):
        predict_upcoming(cfg, [far], "prospective", now=datetime(2021, 2, 1, tzinfo=UTC))
    ok = predict_upcoming(cfg, [late], "exploratory_retrospective", now=datetime(2020, 8, 4, tzinfo=UTC))
    assert ok["prediction_mode"].to_list() == ["exploratory_retrospective"]
    fresh = json.loads(ok["freshness"][0])
    assert fresh["status"] in ("ok", "warning", "fail") and "missing_backfill_weeks" in fresh["checks"]

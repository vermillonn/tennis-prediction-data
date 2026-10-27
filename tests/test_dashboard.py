"""Research dashboard: data loading, filters, lookup, symmetry, metric consistency, missing data, licensing,
closed gates and clean start-up (headless ``AppTest``; no browser, private data or providers)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tennis_data.config import PROJECT_ROOT
from tennis_data.dashboard import data as D

APP = str(PROJECT_ROOT / "dashboard" / "streamlit_app.py")
SECTIONS = ["Overview", "Match explorer", "Player comparison", "Model evaluation", "Data & status"]
ARCHIVE_END = date(2026, 5, 25)


@pytest.fixture(scope="module")
def preds() -> pl.DataFrame:
    df = D.load_predictions()
    assert df is not None, "the public dashboard dataset must be committed"
    return df


@pytest.fixture(scope="module")
def snap() -> pl.DataFrame:
    return D.load_snapshots()


def _run(section: str) -> AppTest:
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if section != "Overview":
        at.sidebar.radio(key="section").set_value(section).run()
    return at


# ------------------------------------------------------------------------------------- published consistency
def test_full_sample_metrics_reproduce_the_published_report(preds):
    rows = D.headline(D.published())
    assert [r["phase"] for r in rows] == ["development", "exploratory"]
    for r in rows:
        sub = D.filter_predictions(preds, phases=[r["phase"]])
        assert sub.height == r["matches"]
        m = D.metrics(sub)
        for k in ("log_loss", "brier", "accuracy", "ece"):
            assert m["p_m4a"][k] == pytest.approx(r["m4a"][k], abs=1e-12)
            assert m["p_elo"][k] == pytest.approx(r["elo"][k], abs=1e-12)
        d = D.paired(sub)["d_log_loss"].to_numpy()
        assert float(np.mean(d)) == pytest.approx(r["d_log_loss"], abs=1e-12)


def test_calibration_matches_the_published_tables(preds):
    pub = D.published()["calibration"]
    for phase in ("development", "exploratory"):
        sub = D.filter_predictions(preds, phases=[phase])
        for col, model in (("p_m4a", "M4A+raw"), ("p_elo", "M0")):
            mine = D.calibration_curve(sub, col).sort("bin")
            ref = pub[phase].filter(pl.col("model") == model).sort("bin")
            assert mine["n"].to_list() == ref["n"].to_list()
            assert np.allclose(mine["mean_predicted"].to_numpy(), ref["mean_predicted"].to_numpy(), atol=1e-8)
            assert np.allclose(mine["observed"].to_numpy(), ref["observed"].to_numpy(), atol=1e-8)


def test_published_aggregates_are_deterministic(preds):
    assert D.headline(D.published()) == D.headline(D.published())
    a, b = D.metrics(preds), D.metrics(preds)
    assert a == b
    assert D.verify_manifest() and all(D.verify_manifest().values())


# ------------------------------------------------------------------------------------------------ licensing
def test_public_dataset_is_archive_only_and_licensed():
    man = D.load_manifest()
    assert "CC BY-NC-SA 4.0" in man["licence"] and "Sackmann" in man["licence"]
    assert man["source"]["archive_commits"] == ["83733587353df8a41f2fd4f516147d5aa83f5a8d"]
    assert (
        man["model"]["frozen_hash"] == "bc4e5c870c6e21b9" and man["model"]["configuration_hash"] == "5e72010100065889"
    )
    files = {p.name for p in D.data_dir().iterdir()}
    assert files <= D.ALLOWED_FILES, files - D.ALLOWED_FILES


def test_records_contain_no_backfill_or_licensed_source_data(preds, snap):
    for df in (preds, snap):
        assert df["tournament_date"].max() <= ARCHIVE_END  # nothing after the pinned archive (no 2026 backfill)
        assert df["match_id"].str.starts_with("tour:").all()
    for c in ("player_a_id", "player_b_id"):
        assert preds[c].dtype.is_integer()  # archive IDs, never provider codes
    assert snap["player_id"].dtype.is_integer() and snap["opponent_id"].dtype.is_integer()
    assert set(preds.columns) == {"match_id", "season", "tournament_date", "prediction_batch", "release_batch",
                                  "tournament_name", "tournament_level", "surface", "round", "player_a_id",
                                  "player_a_name", "player_b_id", "player_b_name", "score", "p_m4a", "p_elo", "a_won",
                                  "phase", "training_cutoff"}  # fmt: skip


def test_status_file_holds_aggregates_only():
    s = D.load_status()
    assert set(s) <= {"generated_from", "freshness_probe", "active_state_snapshot_id", "archive_commit", "snapshots",
                      "backfill_runs"}  # fmt: skip
    allowed = {
        "version",
        "accepted_matches",
        "elo_updates",
        "quarantined_matches",
        "unresolved_identities",
        "service_statistics_rows",
        "verification_passed",
        "staged_readiness_blockers",
        "staged_freshness",
    }
    for r in s["backfill_runs"]:
        assert set(r) <= allowed
        assert all(isinstance(r[k], int) for k in ("accepted_matches", "elo_updates", "quarantined_matches"))
    text = json.dumps(s)
    assert "tml" not in text.lower() and "approval" not in text.lower()


def test_app_code_never_reads_private_or_licensed_paths():
    for p in [PROJECT_ROOT / "dashboard" / f for f in ("streamlit_app.py", "sections.py", "charts.py")] + [
        Path(D.__file__)
    ]:
        src = p.read_text()
        for bad in ("data/operations", "private", "exports", "private_backups", "TennisMyLife", "snapshots/"):
            assert bad not in src, (p.name, bad)


# -------------------------------------------------------------------------------------------- filters/lookup
def test_filters_select_exactly_the_requested_rows(preds):
    sub = D.filter_predictions(preds, surfaces=["Clay"], years=(2010, 2012), levels=["G", "M"])
    assert sub.height > 0
    assert set(sub["surface"]) == {"Clay"} and set(sub["tournament_level"]) <= {"G", "M"}
    assert sub["season"].min() >= 2010 and sub["season"].max() <= 2012
    pid = int(preds["player_a_id"][0])
    mine = D.filter_predictions(preds, player_ids=[pid])
    assert ((mine["player_a_id"] == pid) | (mine["player_b_id"] == pid)).all()
    opp = int(mine.filter(pl.col("player_a_id") == pid)["player_b_id"][0])
    pair = D.filter_predictions(preds, player_ids=[pid], opponent_id=opp)
    assert pair.height >= 1 and all({r["player_a_id"], r["player_b_id"]} == {pid, opp} for r in pair.to_dicts())
    assert D.filter_predictions(preds, years=(1900, 1901)).height == 0


def test_lookup_returns_recorded_predictions_only(preds):
    r = preds.row(123, named=True)
    rec = D.lookup(preds, r["match_id"])
    assert rec["p_m4a"] == r["p_m4a"] and rec["p_elo"] == r["p_elo"]
    assert rec["winner"] == (r["player_a_name"] if r["a_won"] else r["player_b_name"])
    assert rec["m4a_correct"] == ((r["p_m4a"] >= 0.5) == r["a_won"])
    assert D.lookup(preds, "tour:not-a-match") is None


def test_player_order_symmetry(preds):
    swapped = preds.with_columns(
        pl.col("player_b_id").alias("player_a_id"),
        pl.col("player_a_id").alias("player_b_id"),
        pl.col("player_b_name").alias("player_a_name"),
        pl.col("player_a_name").alias("player_b_name"),
        (1 - pl.col("p_m4a")).alias("p_m4a"),
        (1 - pl.col("p_elo")).alias("p_elo"),
        (~pl.col("a_won")).alias("a_won"),
    )
    a, b = D.metrics(preds), D.metrics(swapped)
    for model in D.MODELS:
        for k in ("log_loss", "brier", "accuracy"):
            assert a[model][k] == pytest.approx(b[model][k], abs=1e-12)
    pid = int(preds["player_a_id"][0])
    o1 = D.orient(preds, pid).sort("match_id")
    o2 = D.orient(swapped, pid).sort("match_id")
    assert np.allclose(o1["p_m4a_player"].to_numpy(), o2["p_m4a_player"].to_numpy(), atol=1e-12)
    assert o1["player_won"].to_list() == o2["player_won"].to_list()


def test_snapshots_are_point_in_time(snap):
    pid = int(snap["player_id"][snap.height // 2])
    as_of = date(2015, 6, 1)
    s = D.snapshot_asof(snap, pid, as_of)
    if s is not None:
        assert s["tournament_date"] <= as_of
    f = D.recent_form(snap, pid, as_of)
    assert f.height <= 10 and (f["tournament_date"] < as_of).all()
    assert D.snapshot_asof(snap, pid, date(1980, 1, 1)) is None  # before the snapshot window


# --------------------------------------------------------------------------------------------- missing / gates
def test_missing_dataset_and_reports_degrade_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path / "none"))
    assert D.load_predictions() is None and D.load_manifest() is None and D.verify_manifest() == {}
    assert D.headline(D.published())  # the published reports still work
    monkeypatch.setenv("DASHBOARD_REPORTS_DIR", str(tmp_path / "none"))
    assert D.headline(D.published()) == []


def test_gates_are_read_from_config_and_fail_closed(tmp_path, monkeypatch):
    g = D.gates()
    assert (g.backfill, g.prospective_predictions) == ("INCOMPLETE", "DISABLED")
    assert (
        not g.live_predictions_available and not g.availability_policy_enabled and g.external_publishers_enabled == []
    )
    monkeypatch.setenv("DASHBOARD_CONFIG_DIR", str(tmp_path))
    assert D.gates().prospective_predictions == "DISABLED" and not D.gates().live_predictions_available
    (tmp_path / "prospective.toml").write_text(
        '[status]\nbackfill = "INCOMPLETE"\nprospective_predictions = "ENABLED"\n'
    )
    assert not D.gates().live_predictions_available  # both conditions are required


# --------------------------------------------------------------------------------------------- app start-up
@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_starts_from_a_clean_checkout(section):
    at = _run(section)
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    if section == "Data & status":
        assert len(errors) == 1 and "Live predictions are unavailable" in errors[0] and "DISABLED" in errors[0]
    else:
        assert errors == []


def test_comparison_shows_no_win_probability():
    at = _run("Player comparison")
    assert any("No win probability is shown here" in i.value for i in at.info)
    assert not any("P(win)" in m.label or "probability" in m.label.lower() for m in at.metric)


def test_app_runs_in_aggregate_mode_without_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path))
    for section in ("Overview", "Match explorer", "Model evaluation"):
        at = _run(section)
        assert not at.exception, [e.value for e in at.exception]
        if section == "Match explorer":
            assert any("public dashboard dataset" in i.value for i in at.info)
    st.cache_data.clear()

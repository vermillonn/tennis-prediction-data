"""Phase 3B dataset construction, leakage guards and end-to-end reproducibility on a synthetic build."""

from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
import pytest

from conftest import FakeArchive
from modeling_helpers import build_upstream
from tennis_data.features.model_inputs import FORBIDDEN, ModelInputError
from tennis_data.modeling.artifacts import PREDICTION_COLUMNS
from tennis_data.modeling.config import load_modeling_config
from tennis_data.modeling.dataset import (
    DERIVED,
    DatasetError,
    assert_inputs_allowed,
    build_dataset,
    build_from_frames,
    join_tables,
    model_matrix,
)
from tennis_data.modeling.experiments import Data, verify_frozen
from tennis_data.modeling.pipeline import run_modeling
from tennis_data.storage.parquet import tree_checksums
from tennis_data.storage.query import connect


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m3b")
    path = build_upstream(tmp, FakeArchive(_mk(tmp / "archive")))
    return load_modeling_config(path)


def _mk(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="module")
def run(upstream):
    return run_modeling(upstream)


def _frames(cfg):
    elo = pl.read_parquet(cfg.features.elo.elo_dir / "predictions" / "model_name=surface_elo" / "part-0.parquet")
    feats = pl.concat(
        [pl.read_parquet(p) for p in sorted((cfg.features.features_dir / "serve_return").rglob("*.parquet"))]
    )
    vdir = cfg.features.elo.data.paths.validated_dir / "matches"
    labels = pl.concat(
        [
            pl.read_parquet(p, columns=["match_id", "player_a_id", "player_a_won"])
            for p in sorted(vdir.rglob("*.parquet"))
        ]
    )
    return elo, feats, labels  # fmt: skip


# ------------------------------------------------------------------------------------------------- dataset


def test_join_is_one_to_one_and_keeps_every_row(upstream):
    elo, feats, labels = _frames(upstream)
    ds = build_from_frames(elo, feats, labels, upstream)
    assert ds.height == elo.height == feats.height
    assert ds["match_id"].n_unique() == ds.height
    assert ds["baseline_elo_probability_a"].equals(
        ds.select("match_id").join(elo, on="match_id", how="left")["predicted_probability_a"]
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda e, f, l: (pl.concat([e, e.head(1)]), f, l), "duplicate match_id"),
        (lambda e, f, l: (e, f.head(f.height - 1), l), "row counts differ"),
        (lambda e, f, l: (e, f.with_columns(pl.col("player_a_id").shift(1, fill_value=-1)), l), "player_a_id"),
        (lambda e, f, l: (e, f.with_columns(pl.col("prediction_batch") + pl.duration(days=7)), l), "prediction_batch"),
        (lambda e, f, l: (e, f.with_columns(pl.lit("Clay").alias("surface")), l), "surface"),
        (lambda e, f, l: (e.with_columns(~pl.col("target_a_wins")), f, l), "target_a_wins"),
        (lambda e, f, l: (e, f, l.head(5)), "missing from the validated"),
    ],
)
def test_inconsistent_joins_are_rejected(upstream, mutate, message):
    elo, feats, labels = mutate(*_frames(upstream))
    with pytest.raises(DatasetError, match=message):
        join_tables(elo, feats, labels, upstream)


def _swap_frames(elo, feats, labels):
    """The same matches with player A and B exchanged (probabilities, labels and every a_/b_ column)."""
    ren = {}
    for c in feats.columns:
        if c.startswith("a_"):
            ren[c] = "b_" + c[2:]
        elif c.startswith("b_"):
            ren[c] = "a_" + c[2:]
    ren.update({"a_spw_vs_b": "b_spw_vs_a", "b_spw_vs_a": "a_spw_vs_b"})
    diffs = [c for c in feats.columns if c.endswith("_diff")]
    f2 = feats.rename(ren).with_columns(
        pl.col("player_a_id").alias("player_b_id"), pl.col("player_b_id").alias("player_a_id"),
        *[(-pl.col(c)).alias(c) for c in diffs],
    )  # fmt: skip
    e2 = elo.with_columns(
        pl.col("player_a_id").alias("player_b_id"), pl.col("player_b_id").alias("player_a_id"),
        (1 - pl.col("predicted_probability_a")).alias("predicted_probability_a"), ~pl.col("target_a_wins"),
    )  # fmt: skip
    l2 = labels.with_columns(
        pl.col("player_a_id").alias("_a"),
    )
    l2 = l2.join(e2.select("match_id", pl.col("player_a_id").alias("_new_a")), on="match_id", how="left")
    l2 = l2.select("match_id", pl.col("_new_a").fill_null(pl.col("_a")).alias("player_a_id"),
                   pl.when(pl.col("_new_a").is_null()).then(pl.col("player_a_won")).otherwise(~pl.col("player_a_won"))
                   .alias("player_a_won"))  # fmt: skip
    return e2, f2, l2


def test_every_derived_feature_is_antisymmetric(upstream):
    elo, feats, labels = _frames(upstream)
    a = build_from_frames(elo, feats, labels, upstream).sort("match_id")
    b = build_from_frames(*_swap_frames(elo, feats, labels), upstream).sort("match_id")
    for name in DERIVED:
        np.testing.assert_allclose(b[name].to_numpy(), -a[name].to_numpy(), atol=1e-12, err_msg=name)


def test_inputs_are_guarded(upstream):
    assert_inputs_allowed(list(DERIVED), upstream)  # every registered feature passes the Phase 3A guard
    for d in DERIVED.values():
        assert not set(d.base) & FORBIDDEN, d.name
    with pytest.raises(DatasetError):
        assert_inputs_allowed(["target_a_wins"], upstream)
    with pytest.raises(DatasetError):
        model_matrix(pl.DataFrame({"stats_eligible": [1.0]}), ["stats_eligible"], upstream)


def test_guard_rejects_a_derived_feature_built_from_post_match_columns(upstream, monkeypatch):
    from tennis_data.modeling import dataset

    bad = dataset.Derived("leak", ("stats_eligible",), lambda e: pl.col("stats_eligible"), "x")
    monkeypatch.setitem(dataset.DERIVED, "leak", bad)
    with pytest.raises(ModelInputError, match="forbidden"):
        assert_inputs_allowed(["elo_logit", "leak"], upstream)


def test_post_match_columns_do_not_change_inputs(upstream):
    elo, feats, labels = _frames(upstream)
    a = build_from_frames(elo, feats, labels, upstream)
    feats2 = feats.with_columns(~pl.col("stats_eligible"))
    b = build_from_frames(elo, feats2, labels, upstream)
    for name in DERIVED:
        assert a[name].equals(b[name]), name


def test_dataset_cache_is_reused_only_when_inputs_match(upstream):
    ds1, m1 = build_dataset(upstream, use_cache=False)
    ds2, m2 = build_dataset(upstream)
    assert m2.get("_reused") and m1["input_key"] == m2["input_key"] and ds1.equals(ds2)


# --------------------------------------------------------------------------------------------- end to end


def test_selection_never_reads_exploratory_rows(upstream, run, monkeypatch):
    """Scrambling every exploratory label changes nothing in development, selection or the frozen model."""
    from tennis_data.modeling import experiments as X

    ds, _ = build_dataset(upstream)
    e0 = upstream.exploratory_years[0]
    scrambled = ds.with_columns(
        pl.when(pl.col("prediction_batch").dt.year() >= e0).then(~pl.col("target_a_wins"))
        .otherwise(pl.col("target_a_wins")).alias("target_a_wins"),
        pl.when(pl.col("prediction_batch").dt.year() >= e0).then(1 - pl.col("elo_logit"))
        .otherwise(pl.col("elo_logit")).alias("elo_logit"),
    )  # fmt: skip
    data = Data.of(scrambled)
    dev = X.development(data, upstream)
    sel = X.select(dev, data, upstream)
    assert sel == run["sel"]
    assert dev["grid"].equals(run["dev"]["grid"])
    frozen = X.freeze(dev, sel, data, upstream, run["provenance"])
    assert frozen["frozen_hash"] == run["frozen"]["frozen_hash"]


def test_calibrators_are_fitted_on_earlier_years_only(upstream, run):
    from tennis_data.modeling import experiments as X

    ds, _ = build_dataset(upstream)
    d0 = upstream.development_years[0]
    flipped = ds.with_columns(
        pl.when(pl.col("prediction_batch").dt.year() == d0).then(~pl.col("target_a_wins"))
        .otherwise(pl.col("target_a_wins")).alias("target_a_wins")
    )  # fmt: skip
    data = Data.of(flipped)
    sel = run["sel"]["selected"]
    wf = X.walk_forward(data, upstream, sel, upstream.experiment(sel).inputs, run["sel"]["C"],
                        run["dev"]["years"])  # fmt: skip
    _, cals = X.calibrate(wf, data, upstream, [d0], "temperature")
    assert cals[d0].slope == run["dev"]["calibrators"][(sel, "temperature")][d0].slope


def test_every_model_is_symmetric(run):
    assert max(run["dev"]["grid"]["max_symmetry_dev"]) < 1e-12
    for v in run["exres"]["metrics"].values():
        assert 0 < v["log_loss"] < 1


def test_m0_metrics_equal_phase2_scoring(run):
    from tennis_data.elo.evaluation import metrics

    ds = run["dataset"]
    rows = run["ex_rows"]
    m = metrics(ds[rows], "baseline_elo_probability_a").row(0, named=True)
    assert m["log_loss"] == run["exres"]["metrics"]["M0"]["log_loss"]


def test_artifacts_and_manifest(upstream, run):
    mdir = upstream.modeling_dir
    sel = pl.read_parquet(mdir / "predictions" / "selected.parquet")
    assert list(sel.columns[: len(PREDICTION_COLUMNS)]) == list(PREDICTION_COLUMNS)
    assert (sel["training_cutoff"] <= sel["prediction_batch"]).all()
    assert sel.filter(pl.col("model_id") == run["final_id"])["match_id"].n_unique() == len(run["dev_rows"]) + len(
        run["ex_rows"]
    )
    man = json.loads((mdir / "manifest.json").read_text())
    for rel, info in man["artifacts"].items():
        assert tree_checksums(mdir)[rel] == info["sha256"]
    for s in man["splits"]:
        assert s["last_train_release_batch"] is None or s["last_train_release_batch"] < s["training_cutoff_batch"]
    frozen = json.loads((mdir / "models" / "frozen_model.json").read_text())
    verify_frozen(frozen)
    frozen["specification"]["C"] = 123.0
    with pytest.raises(AssertionError):
        verify_frozen(frozen)
    con = connect(upstream.features.elo.data, modeling_dir=mdir)
    assert con.execute("SELECT count(*) FROM model_selected_predictions").fetchone()[0] == sel.height


def test_reports_are_written(upstream, run):
    out = upstream.reports_dir
    for name in ("development_report.md", "exploratory_report.md", "calibration_report.md", "ablation_report.md",
                 "model_card.md", "modeling_summary.json"):  # fmt: skip
        assert (out / name).exists(), name
    text = (out / "exploratory_report.md").read_text()
    assert "not* an untouched holdout" in text and "Prospective evaluation protocol" in text
    assert "inconclusive" in (out / "development_report.md").read_text() or "candidate" in text


def test_pipeline_is_reproducible(upstream, run, tmp_path):
    first = {**tree_checksums(upstream.reports_dir)}
    first_data = {k: v for k, v in tree_checksums(upstream.modeling_dir).items() if k != "performance.json"}
    run_modeling(upstream, use_cache=False)
    assert tree_checksums(upstream.reports_dir) == first
    second = {k: v for k, v in tree_checksums(upstream.modeling_dir).items() if k != "performance.json"}
    assert second == first_data
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_modeling_leaves_phase1_to_3a_outputs_untouched(upstream, run):
    f, e = upstream.features, upstream.features.elo
    roots = [e.data.paths.validated_dir, e.elo_dir, e.reports_dir, f.features_dir, f.reports_dir]
    before = [tree_checksums(r) for r in roots]
    run_modeling(upstream)
    assert [tree_checksums(r) for r in roots] == before

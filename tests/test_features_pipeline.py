"""End-to-end Phase 3A build on a synthetic Phase 1 dataset."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from conftest import match_row
from elo_helpers import build_phase1, synthetic_season_rows, write_elo_config
from tennis_data.features.config import load_feature_config
from tennis_data.features.dictionary import DICTIONARY_PATH, render_dictionary, schema
from tennis_data.features.pipeline import build_features, data_checksums
from tennis_data.features.report import write_report
from tennis_data.storage.query import connect


def _feature_config(tmp: Path, elo_config: Path, include_retirements=False) -> Path:
    default = (Path(__file__).resolve().parents[1] / "config" / "features.toml").read_text()
    body = default[default.index("[features]") :]
    body = body.replace("include_retirements = false", f"include_retirements = {str(include_retirements).lower()}")
    body = body.replace("min_prior_points = 20000", "min_prior_points = 500").replace(
        "min_fit_points = 20000", "min_fit_points = 500"
    )
    text = f'elo_config = "{elo_config}"\n\n[outputs]\nfeatures_dir = "{tmp / "features"}"\nreports_dir = "{tmp / "reports_features"}"\n\n{body}'
    path = tmp / "features.toml"
    path.write_text(text)
    return path


def _specials() -> list[dict]:
    common = dict(tourney_id="2020-9001", tourney_name="Specials", tourney_date="20200302", round="R16")
    return [
        match_row(1, 100, 200, score="6-3 2-1 RET", **common),  # retirement with real stats
        match_row(2, 300, 400, w_SvGms="10", l_SvGms="16", **common),  # flagged imbalance: Elo yes, stats no
        match_row(3, 500, 600, w_ace="-1", **common),  # impossible stat -> quarantined in Phase 1
        match_row(4, 700, 800, stats=False, **common),  # legitimately missing stats
    ]


def _build(tmp_path, archive, name="run", include_retirements=False):
    rows = synthetic_season_rows(stats=True)
    rows[2020] = rows[2020] + _specials()
    data_cfg = build_phase1(tmp_path / name, archive, rows)
    elo_cfg = write_elo_config(tmp_path / name, data_cfg)
    cfg = load_feature_config(_feature_config(tmp_path / name, elo_cfg, include_retirements))
    built = build_features(cfg)
    write_report(cfg, built)
    return cfg, built


@pytest.fixture
def built(tmp_path, archive):
    return _build(tmp_path, archive)


def test_output_schema_matches_registry(built):
    cfg, b = built
    files = sorted((cfg.features_dir / "serve_return").rglob("*.parquet"))
    assert files
    for f in files:
        assert dict(pl.read_parquet_schema(f)) == schema(cfg)
    t = b["table"]
    assert t["match_id"].n_unique() == t.height
    assert t["feature_version"].unique().to_list() == [cfg.full_version]
    assert t["configuration_hash"].unique().to_list() == [cfg.configuration_hash()]


def test_rows_are_exactly_the_phase2_targets(built):
    cfg, b = built
    targets = b["frame"].filter("is_target")["match_id"].sort()
    assert b["table"]["match_id"].sort().equals(targets)
    assert (b["table"].join(b["frame"].select("match_id", "player_a_id", "player_b_id"), on="match_id")
            .filter((pl.col("player_a_id") != pl.col("player_a_id_right")) | (pl.col("player_b_id") != pl.col("player_b_id_right")))
            .height == 0)  # fmt: skip


def test_stats_eligibility_is_independent_of_elo(built):
    _, b = built
    f = b["frame"].with_columns(pl.col("match_id").str.split(":").list.last().alias("num"))
    special = {r["num"]: r for r in f.filter(pl.col("tournament_id") == "2020-9001").iter_rows(named=True)}
    assert special["2"]["is_update"] is True and special["2"]["stats_eligible"] is False
    assert special["2"]["stats_exclusion_reason"] == "stats_flag_service_games_imbalance"
    assert special["1"]["stats_exclusion_reason"] == "status_retired"
    assert special["4"]["stats_exclusion_reason"] == "stats_missing" and special["4"]["is_update"] is True
    assert "3" not in special  # quarantined in Phase 1: never reaches the pipeline
    # the Phase 2 match population is unchanged by stats eligibility
    assert f.filter("is_target").height == f.filter(pl.col("match_status") == "completed").height


def test_retirement_option_adds_retired_statistics(tmp_path, archive):
    _, base = _build(tmp_path, archive, "base")
    _, ret = _build(tmp_path, archive, "ret", include_retirements=True)
    pick = lambda b: b["frame"].filter(pl.col("match_id") == "tour:2020-9001:1")["stats_eligible"].item()  # noqa: E731
    assert pick(base) is False and pick(ret) is True
    assert base["table"]["match_id"].sort().equals(ret["table"]["match_id"].sort())  # same feature rows


def test_reproducible_build(tmp_path, archive):
    a_cfg, a = _build(tmp_path, archive, "a")
    b_cfg, b = _build(tmp_path, archive, "b")
    assert a["summary"]["outputs"] == b["summary"]["outputs"]
    assert data_checksums(a_cfg) == data_checksums(b_cfg)
    assert (a_cfg.reports_dir / "tables" / "missingness.csv").read_text() == (
        b_cfg.reports_dir / "tables" / "missingness.csv"
    ).read_text()


def test_duckdb_view_and_report(built):
    cfg, b = built
    con = connect(cfg.elo.data, features_dir=cfg.features_dir)
    n = con.execute("SELECT count(*) FROM serve_return_features").fetchone()[0]
    assert n == b["table"].height
    joined = con.execute("SELECT count(*) FROM serve_return_features f JOIN matches m USING (match_id)").fetchone()[0]
    assert joined == n
    text = (cfg.reports_dir / "coverage_report.md").read_text()
    for heading in ("Two different coverage concepts", "Missingness of every exported column", "Selection bias"):
        assert heading in text


def test_every_exported_column_is_documented_and_dictionary_is_current():
    cfg = load_feature_config()
    rendered = render_dictionary(cfg)
    for col in schema(cfg):
        bare = col[2:] if col.startswith(("a_", "b_")) and col not in ("a_spw_vs_b", "b_spw_vs_a") else col
        assert f"`{bare}`" in rendered, col
    assert DICTIONARY_PATH.read_text() == rendered, "run `tennis-data features-docs`"


def test_required_columns_exist():
    required = [
        "match_id", "prediction_batch", "player_a_id", "player_b_id", "surface", "a_spw_raw", "b_spw_raw",
        "a_rpw_raw", "b_rpw_raw", "a_spw_shrunk", "b_spw_shrunk", "a_rpw_shrunk", "b_rpw_shrunk", "a_surface_spw",
        "b_surface_spw", "a_surface_rpw", "b_surface_rpw", "a_serve_adjusted", "b_serve_adjusted",
        "a_return_adjusted", "b_return_adjusted", "a_service_points_observed", "b_service_points_observed",
        "a_return_points_observed", "b_return_points_observed", "feature_version", "configuration_hash",
        "spw_diff", "rpw_diff", "serve_adjusted_diff", "return_adjusted_diff",
    ]  # fmt: skip
    cols = schema(load_feature_config())
    assert [c for c in required if c not in cols] == []

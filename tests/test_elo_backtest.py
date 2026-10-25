"""End-to-end Elo backtest on a synthetic Phase 1 build."""

from __future__ import annotations

import polars as pl
import pytest

from conftest import match_row
from elo_helpers import build_phase1, synthetic_season_rows, write_elo_config
from tennis_data.elo.backtest import PREDICTION_COLUMNS, backtest
from tennis_data.elo.config import load_elo_config
from tennis_data.elo.report import write_report
from tennis_data.storage.parquet import tree_checksums
from tennis_data.storage.query import connect

REQUIRED = (
    "match_id", "tournament_date", "prediction_batch", "player_a_id", "player_b_id", "player_a_elo_before",
    "player_b_elo_before", "surface_elo_a_before", "surface_elo_b_before", "surface_match_count_a",
    "surface_match_count_b", "predicted_probability_a", "target_a_wins", "model_version", "configuration_hash",
)  # fmt: skip


def _special_rows() -> list[dict]:
    """Rows in 2020 that must be excluded or quarantined."""
    common = dict(stats=False, tourney_id="2020-9001", tourney_name="Specials", tourney_date="20200302")
    return [
        match_row(1, 100, 200, score="W/O", round="R16", **common),
        match_row(2, 300, 400, score="6-3 2-1 RET", round="R16", **common),
        match_row(3, 500, 600, score="6-4 6-4", round="R16", **common),
        match_row(9, 500, 600, score="6-4 6-4", round="R16", **common),  # duplicate copy -> quarantine
        match_row(4, 700, 199999, score="6-4 6-4", round="R16", **common),  # placeholder -> quarantine
    ]


def _run(tmp_path, archive, name="run", mutate=None, **elo_overrides):
    rows = synthetic_season_rows()
    rows[2020] = rows[2020] + _special_rows()
    if mutate:
        rows = mutate(rows)
    data_cfg = build_phase1(tmp_path / name, archive, rows)
    cfg = load_elo_config(write_elo_config(tmp_path / name, data_cfg, **elo_overrides))
    result = backtest(cfg)
    write_report(cfg, result)
    return cfg, result


@pytest.fixture
def run(tmp_path, archive):
    return _run(tmp_path, archive)


def _preds(cfg, model="surface_elo"):
    return pl.read_parquet(cfg.elo_dir / "predictions" / f"model_name={model}" / "part-0.parquet")


def test_prediction_dataset_contract(run):
    cfg, _ = run
    for model in ("global_elo", "surface_elo"):
        p = _preds(cfg, model)
        assert tuple(p.columns) == PREDICTION_COLUMNS
        assert set(REQUIRED) <= set(p.columns)
        assert p["match_id"].n_unique() == p.height
        assert (p["match_status"] == "completed").all()
        assert p["predicted_probability_a"].is_between(0, 1, closed="none").all()
        assert p["configuration_hash"].n_unique() == 1 and p["model_version"][0] == "elo-test+engine1"


def test_excluded_and_quarantined_rows_never_appear(run):
    cfg, _ = run
    ids = set(_preds(cfg)["match_id"])
    assert "tour:2020-9001:1" not in ids  # walkover
    assert "tour:2020-9001:2" not in ids  # retirement (excluded by default)
    assert "tour:2020-9001:3" in ids
    assert "tour:2020-9001:9" not in ids  # quarantined duplicate copy
    assert "tour:2020-9001:4" not in ids  # quarantined placeholder player
    sens = pl.read_parquet(cfg.elo_dir / "sensitivity_predictions" / "**" / "*.parquet")
    assert "tour:2020-9001:2" not in set(sens["match_id"])  # sensitivity changes updates, not targets


def test_targets_and_ordering_follow_phase1(run):
    cfg, _ = run
    p = _preds(cfg)
    con = connect(cfg.data, elo_dir=cfg.elo_dir)
    joined = con.execute(
        "SELECT count(*) FROM elo_predictions e JOIN matches m USING (match_id) "
        "WHERE e.player_a_id <> m.player_a_id OR e.target_a_wins <> m.player_a_won"
    ).fetchone()[0]
    assert joined == 0
    assert 0.3 < p["target_a_wins"].mean() < 0.7  # A/B is not the winner
    assert con.execute("SELECT count(*) FROM elo_rating_history").fetchone()[0] > 0


def test_retirement_sensitivity_changes_only_later_ratings(run):
    cfg, _ = run
    base = _preds(cfg, "global_elo")
    sens = pl.read_parquet(
        cfg.elo_dir / "sensitivity_predictions" / "model_name=global_elo_with_retirements" / "*.parquet"
    )
    assert set(base["match_id"]) == set(sens["match_id"])
    early = pl.col("prediction_batch") <= pl.date(2020, 2, 29)
    cols = ["match_id", "player_a_elo_before", "player_b_elo_before"]
    assert base.filter(early).select(cols).equals(sens.filter(early).select(cols))


def test_reproducible_output_hashes(tmp_path, archive):
    a_cfg, a = _run(tmp_path, archive, "a")
    b_cfg, b = _run(tmp_path, archive, "b")
    assert a["summary"]["outputs"] == b["summary"]["outputs"]
    assert tree_checksums(a_cfg.reports_dir / "tables") == tree_checksums(b_cfg.reports_dir / "tables")
    norm = lambda c: (c.reports_dir / "evaluation_report.md").read_text()  # noqa: E731
    assert norm(a_cfg) == norm(b_cfg)


def test_configuration_hash(run):
    cfg, _ = run
    from dataclasses import replace

    h = cfg.configuration_hash()
    assert h == cfg.configuration_hash()
    assert cfg.with_params(replace(cfg.params, k_constant=31.0)).configuration_hash() != h
    assert cfg.with_params(replace(cfg.params, shrinkage_matches=7.0)).configuration_hash() != h


def test_tuning_never_sees_test_outcomes(tmp_path, archive):
    """Flipping every test-period result leaves model selection and all pre-test predictions unchanged."""

    def flip_2020(rows):
        flipped = []
        for r in rows[2020]:
            r = dict(r)
            if r["tourney_id"] != "2020-9001":
                for wcol in [c for c in r if c.startswith("winner_")] + [c for c in r if c.startswith("w_")]:
                    lcol = ("loser_" + wcol[7:]) if wcol.startswith("winner_") else ("l_" + wcol[2:])
                    r[wcol], r[lcol] = r[lcol], r[wcol]
            flipped.append(r)
        return {**rows, 2020: flipped}

    a_cfg, a = _run(tmp_path, archive, "orig")
    b_cfg, b = _run(tmp_path, archive, "flipped", mutate=flip_2020)
    assert a["summary"]["selected"] == b["summary"]["selected"]
    assert a["tables"]["validation_overall"].equals(b["tables"]["validation_overall"])
    pre = pl.col("prediction_batch") < pl.date(2020, 1, 1)
    for model in ("global_elo", "surface_elo"):
        pa, pb = _preds(a_cfg, model).filter(pre), _preds(b_cfg, model).filter(pre)
        assert pa.equals(pb)
    # ...and the test period really did change.
    assert not a["tables"]["test_overall"].equals(b["tables"]["test_overall"])


def test_report_does_not_claim_unsupported_improvement(run):
    cfg, result = run
    text = (cfg.reports_dir / "evaluation_report.md").read_text()
    for name, c in result["summary"]["test_comparisons"].items():
        if c["log_loss_diff_ci_low"] <= 0 <= c["log_loss_diff_ci_high"]:
            assert f"`{name}`: no log-loss difference" in text

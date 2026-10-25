"""Explicit eligibility predicates, retirement handling and quarantine isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from tennis_data.config import ConfigError
from tennis_data.elo.config import Eligibility, load_elo_config
from tennis_data.elo.data import engine_frame
from tennis_data.elo.eligibility import (
    assert_no_quarantined,
    serve_return_eligible,
    target_eligible,
    update_eligible,
)

STATUSES = ["completed", "retired", "walkover", "default", "abandoned", "unfinished", "unknown"]


def _matches(**overrides) -> pl.DataFrame:
    n = len(STATUSES)
    base = {
        "match_id": [f"tour:x:{i}" for i in range(n)],
        "match_status": STATUSES,
        "player_a_id": [1] * n,
        "player_b_id": [2] * n,
        "winner_id": [1] * n,
        "player_a_won": [True] * n,
        "tournament_date": [date(2020, 1, 6)] * n,
        "surface": ["Hard"] * n,
        "tournament_level": ["A"] * n,
        "draw_size": [32] * n,
        "source_family": ["tour"] * n,
        "stats_status": ["complete"] * n,
        "quality_flags": [[] for _ in range(n)],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def _flags(df, expr):
    return dict(zip(df["match_status"], df.select(expr.alias("x"))["x"], strict=True))


def test_only_completed_matches_by_default():
    e = Eligibility()
    df = _matches()
    assert _flags(df, target_eligible(e)) == {s: s == "completed" for s in STATUSES}
    assert _flags(df, update_eligible(e)) == {s: s == "completed" for s in STATUSES}


def test_retirement_sensitivity_adds_recorded_winner_to_updates_only():
    e = Eligibility(update_statuses=("completed", "retired"))
    df = _matches()
    assert _flags(df, update_eligible(e))["retired"] is True
    assert _flags(df, target_eligible(e))["retired"] is False
    for s in ("walkover", "default", "abandoned", "unfinished", "unknown"):
        assert _flags(df, update_eligible(e))[s] is False


def test_config_rejects_untrustworthy_statuses(tmp_path):
    from elo_helpers import write_elo_config
    from tennis_data.config import DEFAULT_CONFIG

    load_elo_config(write_elo_config(tmp_path / "ok", DEFAULT_CONFIG))  # the valid baseline loads
    path = write_elo_config(tmp_path / "bad", DEFAULT_CONFIG, update_statuses=["completed", "walkover"])
    with pytest.raises(ConfigError, match="walkover"):
        load_elo_config(path)


def test_identity_problems_are_never_eligible():
    df = _matches(winner_id=[3] * len(STATUSES), player_b_id=[1] * len(STATUSES))
    assert not any(_flags(df, target_eligible(Eligibility())).values())
    assert not any(_flags(df, update_eligible(Eligibility())).values())


def test_serve_return_eligibility():
    df = _matches(
        stats_status=["complete", "complete", "none", "complete", "complete", "complete", "complete"],
        quality_flags=[[], [], [], [], [], [], []],
    )
    flags = _flags(df, serve_return_eligible())
    assert flags["completed"] is True and flags["retired"] is False
    assert _flags(df, serve_return_eligible(include_retirements=True))["retired"] is True
    bad = _matches(quality_flags=[["service_games_imbalance"]] * len(STATUSES))
    assert _flags(bad, serve_return_eligible())["completed"] is False


def test_quarantined_rows_are_rejected():
    df = _matches()
    with pytest.raises(ValueError, match="quarantined"):
        assert_no_quarantined(df, pl.Series(["tour:x:3"]))
    assert_no_quarantined(df, pl.Series(["tour:other:1"]))


def test_pools_and_exclusion_reasons():
    cfg = load_elo_config()
    df = _matches(
        source_family=["tour", "tour", "tour", "tour", "tour", "futures", "tour"],
        tournament_level=["A", "A", "A", "A", "A", "S", "A"],
    ).with_columns(pl.lit(None, dtype=pl.Int32).alias("round_order"))
    out = engine_frame(df, cfg)
    reasons = dict(zip(out["match_status"], out["exclusion_reason"], strict=True))
    assert reasons["completed"] is None
    assert reasons["retired"] == "status_retired"
    assert reasons["walkover"] == "status_walkover"
    assert reasons["unfinished"] == "outside_rating_pools"  # futures row
    assert out.filter(pl.col("pool").is_null())["is_update"].to_list() == [False]


def test_eligibility_replace_keeps_targets():
    e = replace(Eligibility(), update_statuses=("completed", "retired"))
    assert e.target_statuses == ("completed",)

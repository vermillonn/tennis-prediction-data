"""Synthetic inputs for the Phase 3A feature engine."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import polars as pl

from tennis_data.features.config import load_feature_config

SAT0 = date(2020, 1, 4)  # a Saturday (Phase 2 batches start on Saturdays)


def cfg(min_prior_points: int = 0, opponent: bool = False, **shrink):
    base = load_feature_config()
    return base.with_(
        shrinkage=replace(base.shrinkage, min_prior_points=min_prior_points, **shrink),
        opponent=replace(base.opponent, enabled=opponent, min_fit_points=1, refit_interval_weeks=1),
    )


def match(
    mid,
    week,
    a,
    b,
    a_sp=(60, 40),
    b_sp=(50, 30),
    *,
    surface="Hard",
    release_weeks=0,
    target=True,
    eligible=True,
    round_order=12,
):
    pb = SAT0 + timedelta(weeks=week)
    return {
        "match_id": mid, "pool": "atp_tour", "season": pb.year, "tournament_level": "A",
        "prediction_batch": pb, "release_batch": pb + timedelta(weeks=release_weeks), "tournament_date": pb + timedelta(days=2),
        "round_order": round_order, "surface": surface, "player_a_id": a, "player_b_id": b,
        "a_sp_played": a_sp[0], "a_sp_won": a_sp[1], "b_sp_played": b_sp[0], "b_sp_won": b_sp[1],
        "is_target": target, "stats_eligible": eligible, "player_a_won": True,
    }  # fmt: skip


def frame(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"round_order": pl.Int32, "season": pl.Int32})


def run(rows, c=None) -> pl.DataFrame:
    from tennis_data.features.observations import observations
    from tennis_data.features.pipeline import assemble
    from tennis_data.features.snapshots import build_snapshots

    c = c or cfg()
    f = frame(rows)
    return assemble(build_snapshots(f, observations(f), c).features, c)


def row(table: pl.DataFrame, mid: str) -> dict:
    return table.filter(pl.col("match_id") == mid).row(0, named=True)

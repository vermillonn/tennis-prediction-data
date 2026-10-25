"""Explicit eligibility predicates over the validated match table.

Three separate questions, answered by three predicates:

- ``target_eligible``: may this match be a match-winner prediction target (scored in evaluation)?
- ``update_eligible``: may this match's result update Elo ratings?
- ``serve_return_eligible``: may this match's serve/return statistics feed future features (Phase 3)?

All three require a row from the *validated* table. Quarantined rows (untrustworthy identity,
outcome or statistics) are never loaded by the Elo engine; ``assert_no_quarantined`` enforces it.

Statuses without a trustworthy played result (walkover, default, abandoned, unfinished, unknown)
are never eligible. Retirements are excluded unless configured: the recorded winner is real, but
the match was not played to completion, so it is weaker evidence of relative strength.
"""

from __future__ import annotations

import polars as pl

from tennis_data.elo.config import Eligibility, EloConfig

NEVER_ELIGIBLE_STATUSES = ("walkover", "default", "abandoned", "unfinished", "unknown")

# Warning flags that make a row's serve statistics unusable for serve/return features.
STATS_UNUSABLE_FLAGS = (
    "partial_stats",
    "stats_on_walkover",
    "service_games_imbalance",
    "serve_points_without_service_games",
    "empty_stats_on_played_match",
)


def _identity_ok() -> pl.Expr:
    return (
        pl.col("player_a_id").is_not_null()
        & pl.col("player_b_id").is_not_null()
        & (pl.col("player_a_id") != pl.col("player_b_id"))
        & ((pl.col("winner_id") == pl.col("player_a_id")) | (pl.col("winner_id") == pl.col("player_b_id"))).fill_null(
            False
        )
    )


def target_eligible(e: Eligibility) -> pl.Expr:
    return _identity_ok() & pl.col("match_status").is_in(list(e.target_statuses))


def update_eligible(e: Eligibility) -> pl.Expr:
    return _identity_ok() & pl.col("match_status").is_in(list(e.update_statuses))


def serve_return_eligible(include_retirements: bool = False) -> pl.Expr:
    """Defined now for Phase 3; not used by the Elo engine."""
    statuses = ["completed", "retired"] if include_retirements else ["completed"]
    return (
        _identity_ok()
        & pl.col("match_status").is_in(statuses)
        & (pl.col("stats_status") == "complete")
        & ~pl.col("quality_flags").list.eval(pl.element().is_in(list(STATS_UNUSABLE_FLAGS))).list.any()
    )


def in_pool(cfg: EloConfig) -> pl.Expr:
    """Name of the first rating pool the row belongs to (null if none)."""
    expr = pl.lit(None, dtype=pl.String)
    for pool in reversed(cfg.pools):
        member = pl.col("source_family").is_in(list(pool.families)) & pl.col("tournament_level").is_in(
            list(pool.levels)
        )
        expr = pl.when(member).then(pl.lit(pool.name)).otherwise(expr)
    return expr


def exclusion_reason(cfg: EloConfig) -> pl.Expr:
    """Why a validated row is not a prediction target (null when it is one)."""
    e = cfg.eligibility
    return (
        pl.when(pl.col("pool").is_null())
        .then(pl.lit("outside_rating_pools"))
        .when(~_identity_ok())
        .then(pl.lit("invalid_identity"))
        .when(pl.col("match_status").is_in(list(e.target_statuses)))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.concat_str(pl.lit("status_"), pl.col("match_status")))
    )


def assert_no_quarantined(matches: pl.DataFrame, quarantine_ids: pl.Series) -> None:
    leaked = matches.filter(pl.col("match_id").is_in(quarantine_ids.drop_nulls().implode()))
    if leaked.height:
        raise ValueError(f"{leaked.height} quarantined match_id(s) reached the Elo input, e.g. {leaked['match_id'][0]}")

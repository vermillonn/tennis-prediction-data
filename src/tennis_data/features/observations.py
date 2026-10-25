"""Match-level point statistics and the ``stats_eligible`` predicate.

Per match, from validated Phase 1 columns (Int64, null when missing):

    service points played  (A)  = player_a_svpt
    service points won     (A)  = player_a_first_won + player_a_second_won
    return points played   (A)  = player_b_svpt
    return points won      (A)  = player_b_svpt - (player_b_first_won + player_b_second_won)

and symmetrically for B. Missing statistics stay null; nothing is replaced by zero or an average.

``stats_eligible`` is independent of Elo eligibility. A match may update Elo but not contribute
statistics (e.g. a flagged service-game imbalance), or vice versa when retirements are included here.
"""

from __future__ import annotations

import polars as pl

from tennis_data.elo.data import engine_frame
from tennis_data.elo.eligibility import STATS_UNUSABLE_FLAGS, assert_no_quarantined
from tennis_data.features.config import FeatureConfig

STAT_INPUT_COLUMNS = (
    "match_id", "tournament_id", "tournament_date", "season", "surface", "tournament_level", "draw_size",
    "round", "round_order", "match_status", "player_a_id", "player_b_id", "winner_id", "player_a_won",
    "source_family", "quality_flags", "stats_status",
    "player_a_svpt", "player_a_first_in", "player_a_first_won", "player_a_second_won",
    "player_b_svpt", "player_b_first_in", "player_b_first_won", "player_b_second_won",
)  # fmt: skip


def load_matches(cfg: FeatureConfig) -> tuple[pl.DataFrame, pl.Series]:
    paths = cfg.elo.data.paths
    files = sorted((paths.validated_dir / "matches").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No validated data under {paths.validated_dir}. Run `tennis-data run` first.")
    matches = pl.concat([pl.read_parquet(f, columns=list(STAT_INPUT_COLUMNS)) for f in files], how="vertical")
    qpath = paths.quarantine_dir / "matches.parquet"
    q = pl.read_parquet(qpath, columns=["match_id"])["match_id"] if qpath.exists() else pl.Series([], dtype=pl.String)
    return matches, q


def _won(side: str) -> pl.Expr:
    return pl.col(f"player_{side}_first_won") + pl.col(f"player_{side}_second_won")


def point_columns() -> list[pl.Expr]:
    return [
        pl.col("player_a_svpt").alias("a_sp_played"),
        _won("a").alias("a_sp_won"),
        pl.col("player_b_svpt").alias("b_sp_played"),
        _won("b").alias("b_sp_won"),
    ]


def stats_eligibility(cfg: FeatureConfig) -> tuple[pl.Expr, pl.Expr]:
    """(stats_eligible, stats_exclusion_reason). Requires point columns and Phase 2 pool/status columns."""
    statuses = ["completed", "retired"] if cfg.include_retirements else ["completed"]
    flagged = pl.col("quality_flags").list.eval(pl.element().is_in(list(STATS_UNUSABLE_FLAGS))).list.any()
    first_flag = (
        pl.col("quality_flags")
        .list.eval(pl.element().filter(pl.element().is_in(list(STATS_UNUSABLE_FLAGS))))
        .list.first()
    )
    identity = (
        pl.col("player_a_id").is_not_null() & pl.col("player_b_id").is_not_null()
        & (pl.col("player_a_id") != pl.col("player_b_id"))
    )  # fmt: skip
    positive = (pl.col("a_sp_played") > 0) & (pl.col("b_sp_played") > 0)
    bounded = (
        (pl.col("a_sp_won") >= 0) & (pl.col("b_sp_won") >= 0)
        & (pl.col("a_sp_won") <= pl.col("a_sp_played")) & (pl.col("b_sp_won") <= pl.col("b_sp_played"))
    )  # fmt: skip
    reason = (
        pl.when(pl.col("pool").is_null()).then(pl.lit("outside_rating_pools"))
        .when(~identity).then(pl.lit("invalid_identity"))
        .when(~pl.col("match_status").is_in(statuses)).then(pl.concat_str(pl.lit("status_"), pl.col("match_status")))
        .when(pl.col("stats_status") == "none").then(pl.lit("stats_missing"))
        .when(pl.col("stats_status") != "complete").then(pl.lit("stats_partial"))
        .when(flagged).then(pl.concat_str(pl.lit("stats_flag_"), first_flag))
        .when(~positive.fill_null(False)).then(pl.lit("zero_service_points"))
        .when(~bounded.fill_null(False)).then(pl.lit("won_exceeds_played"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )  # fmt: skip
    return reason.is_null().alias("stats_eligible"), reason.alias("stats_exclusion_reason")


def feature_frame(matches: pl.DataFrame, cfg: FeatureConfig, quarantine_ids: pl.Series | None = None) -> pl.DataFrame:
    """Phase 2 engine frame (pool, batches, targets) plus point counts and stats eligibility."""
    if quarantine_ids is not None:
        assert_no_quarantined(matches, quarantine_ids)
    df = engine_frame(matches, cfg.elo).with_columns(point_columns())
    eligible, reason = stats_eligibility(cfg)
    return df.with_columns(eligible, reason)


def observations(df: pl.DataFrame) -> pl.DataFrame:
    """Two server-returner observations per stats-eligible match, in canonical order."""
    base = df.filter("stats_eligible")
    keep = ["match_id", "pool", "prediction_batch", "release_batch", "tournament_date", "round_order", "surface"]
    a = base.select(
        *keep, pl.col("player_a_id").alias("server_id"), pl.col("player_b_id").alias("returner_id"),
        pl.col("a_sp_played").alias("points"), pl.col("a_sp_won").alias("won"),
    )  # fmt: skip
    b = base.select(
        *keep, pl.col("player_b_id").alias("server_id"), pl.col("player_a_id").alias("returner_id"),
        pl.col("b_sp_played").alias("points"), pl.col("b_sp_won").alias("won"),
    )  # fmt: skip
    return pl.concat([a, b]).sort(
        "release_batch", "prediction_batch", "tournament_date", "round_order", "match_id", "server_id", nulls_last=True
    )

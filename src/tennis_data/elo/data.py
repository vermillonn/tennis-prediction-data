"""Load the validated Phase 1 table and derive the Elo engine input."""

from __future__ import annotations

import polars as pl

from tennis_data.elo.chronology import add_batches
from tennis_data.elo.config import EloConfig
from tennis_data.elo.eligibility import (
    assert_no_quarantined,
    exclusion_reason,
    in_pool,
    target_eligible,
    update_eligible,
)
from tennis_data.elo.engine import surface_key

_COLUMNS = (
    "match_id", "tournament_id", "tournament_date", "season", "surface", "tournament_level", "draw_size",
    "round", "round_order", "match_status", "player_a_id", "player_b_id", "winner_id", "player_a_won",
    "source_family", "quality_flags", "stats_status",
)  # fmt: skip


def load_validated(cfg: EloConfig) -> tuple[pl.DataFrame, pl.Series]:
    paths = cfg.data.paths
    validated = paths.validated_dir / "matches"
    files = sorted(validated.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No validated data under {validated}. Run `tennis-data run` first.")
    matches = pl.concat([pl.read_parquet(f, columns=list(_COLUMNS)) for f in files], how="vertical")
    qpath = paths.quarantine_dir / "matches.parquet"
    quarantine_ids = (
        pl.read_parquet(qpath, columns=["match_id"])["match_id"] if qpath.exists() else pl.Series([], dtype=pl.String)
    )
    return matches, quarantine_ids


def engine_frame(matches: pl.DataFrame, cfg: EloConfig, quarantine_ids: pl.Series | None = None) -> pl.DataFrame:
    """All validated rows with pool, batches, eligibility and the engine's input columns."""
    if quarantine_ids is not None:
        assert_no_quarantined(matches, quarantine_ids)
    e = cfg.eligibility
    keys = {s: surface_key(s, cfg.params.carpet_policy) for s in ("Hard", "Clay", "Grass", "Carpet")}
    df = add_batches(matches.with_columns(in_pool(cfg).alias("pool")), cfg.chronology)
    return df.with_columns(
        pl.col("player_a_won").alias("target_a_wins"),
        pl.col("surface").replace_strict(keys, default=None, return_dtype=pl.String).alias("surface_key"),
        (target_eligible(e) & pl.col("pool").is_not_null()).alias("is_target"),
        (update_eligible(e) & pl.col("pool").is_not_null()).alias("is_update"),
        exclusion_reason(cfg).alias("exclusion_reason"),
    )

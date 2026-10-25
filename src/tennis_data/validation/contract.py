"""Table-level contract for the validated match table, enforced before anything is written."""

from __future__ import annotations

import polars as pl

from tennis_data.normalization.schema import CANONICAL_SCHEMA, NON_NULLABLE


class SchemaViolation(ValueError):
    """The validated table does not satisfy the canonical contract (a pipeline bug, not a data issue)."""


def enforce_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Select and cast to the canonical schema, then assert its invariants."""
    missing = [c for c in CANONICAL_SCHEMA if c not in df.columns]
    if missing:
        raise SchemaViolation(f"Validated table is missing columns: {missing}")
    out = df.select([pl.col(c).cast(t, strict=True) for c, t in CANONICAL_SCHEMA.items()])

    problems = []
    nulls = out.select(pl.col(list(NON_NULLABLE)).null_count()).row(0, named=True)
    problems += [f"{c}: {n} nulls in a non-nullable column" for c, n in nulls.items() if n]
    if out["match_id"].n_unique() != out.height:
        problems.append("match_id is not unique")
    bad_winner = out.filter(
        (pl.col("winner_id") != pl.col("player_a_id")) & (pl.col("winner_id") != pl.col("player_b_id"))
    ).height
    if bad_winner:
        problems.append(f"{bad_winner} rows whose winner_id is neither player")
    if out.filter(pl.col("player_a_won") != (pl.col("winner_id") == pl.col("player_a_id"))).height:
        problems.append("player_a_won disagrees with winner_id")
    if out.filter(pl.col("player_a_id") == pl.col("player_b_id")).height:
        problems.append("player_a_id equals player_b_id")
    if out.filter(pl.col("match_date").is_not_null() | pl.col("match_start_time").is_not_null()).height:
        problems.append("match_date/match_start_time must stay null: the source has no match timestamps")
    if problems:
        raise SchemaViolation("; ".join(problems))
    return out

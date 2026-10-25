"""Duplicate and conflicting-record detection across all selected files.

Order of evaluation:
1. duplicate_record: rows whose source content is identical except for match_num.
   The first occurrence (by source file, then row number) is kept; later copies are
   quarantined with duplicate_of pointing at the kept match_id.
2. Among the remaining rows:
   - conflicting_match_id: one match_id, several different rows.
   - conflicting_record: one (tournament, round, unordered player pair) recorded more
     than once in a round where a pair can only meet once. Every version is
     quarantined, because there is no basis to prefer one.
   - player_repeated_in_round: a player in several matches of one knockout round.
"""

from __future__ import annotations

import polars as pl

from tennis_data.config import Config
from tennis_data.normalization.normalize import CONTENT_KEY, ERRORS, LOSER_ID

DUPLICATE_OF = "duplicate_of"
_ORDER = "_order"


def _append(flag: pl.Expr, code: str) -> pl.Expr:
    return pl.when(flag).then(pl.col(ERRORS).list.concat(pl.lit([code]))).otherwise(pl.col(ERRORS)).alias(ERRORS)


def detect_duplicates(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    df = df.with_columns(pl.int_range(pl.len()).alias(_ORDER)).sort("source_file", "source_row_number")

    # The frame is sorted by source position, so "first" within a group is the earliest row.
    first = pl.col("match_id").first().over(CONTENT_KEY)
    is_copy = pl.int_range(pl.len()).over(CONTENT_KEY) > 0
    df = df.with_columns(
        pl.when(is_copy).then(first).otherwise(None).cast(pl.String).alias(DUPLICATE_OF),
    ).with_columns(_append(pl.col(DUPLICATE_OF).is_not_null(), "duplicate_record"))

    df = df.with_columns(
        pl.col(DUPLICATE_OF).is_null().alias("_live"),
        pl.min_horizontal("winner_id", LOSER_ID).alias("_lo"),
        pl.max_horizontal("winner_id", LOSER_ID).alias("_hi"),
    )
    live, lo = pl.col("_live"), pl.col("_lo")
    once_only_round = ~pl.col("round").is_in(list(config.validation.multi_meeting_rounds))
    placeholders = list(config.validation.placeholder_player_ids)

    id_conflict = live & pl.col("match_id").is_not_null() & (pl.len().over("match_id", "_live") > 1)
    pair_conflict = (
        live & once_only_round & lo.is_not_null()
        & (pl.len().over("tournament_id", "round", "_lo", "_hi", "_live") > 1)
    )  # fmt: skip
    df = df.with_columns(_append(id_conflict.fill_null(False), "conflicting_match_id")).with_columns(
        _append(pair_conflict.fill_null(False), "conflicting_record")
    )

    # A player appearing twice in one knockout round (ignoring copies and placeholder IDs).
    long = (
        df.filter(live & once_only_round)
        .select(_ORDER, "tournament_id", "round", pl.concat_list("winner_id", LOSER_ID).alias("pid"))
        .explode("pid")
        .filter(pl.col("pid").is_not_null() & ~pl.col("pid").is_in(placeholders))
    )
    repeated = long.filter(pl.len().over("tournament_id", "round", "pid") > 1)[_ORDER].unique()
    df = df.with_columns(_append(pl.col(_ORDER).is_in(repeated.implode()), "player_repeated_in_round"))
    return df.sort(_ORDER).drop(_ORDER, "_live", "_lo", "_hi")

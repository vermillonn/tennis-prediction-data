"""Checks against the archive's reference files: the player master and the rankings.

These checks never quarantine a row on their own. Identity gaps become warnings;
rank disagreements are written to the discrepancy report only.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tennis_data.config import Config
from tennis_data.ingestion.sources import PLAYERS_FILE, RANKINGS_FILES
from tennis_data.normalization.normalize import WARNINGS


def _norm_name(expr: pl.Expr) -> pl.Expr:
    return expr.str.strip_chars().str.replace_all(r"\s+", " ").str.to_lowercase()


def load_players(config: Config) -> pl.DataFrame | None:
    path = config.raw_source_dir / PLAYERS_FILE
    if not config.selection.include_players or not path.exists():
        return None
    df = pl.read_csv(path, infer_schema=False)
    return df.select(
        pl.col("player_id").cast(pl.Int64, strict=False).alias("player_id"),
        _norm_name(
            pl.concat_str([pl.col("name_first").fill_null(""), pl.col("name_last").fill_null("")], separator=" ")
        ).alias("master_name"),
        (pl.col("name_last").str.strip_chars().str.to_lowercase() == "unknown")
        .fill_null(False)
        .alias("master_unnamed"),
    ).unique("player_id", keep="first", maintain_order=True)


def load_rankings(config: Config) -> pl.DataFrame | None:
    paths: list[Path] = [config.raw_source_dir / p for p in RANKINGS_FILES]
    if not config.selection.include_rankings or not all(p.exists() for p in paths):
        return None
    frames = [pl.read_csv(p, infer_schema=False) for p in paths]
    df = pl.concat(frames, how="vertical")
    return df.select(
        pl.col("ranking_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("ranking_date"),
        pl.col("player").cast(pl.Int64, strict=False).alias("player_id"),
        pl.col("rank").cast(pl.Int64, strict=False).alias("rank"),
    ).drop_nulls()


def apply_identity_checks(df: pl.DataFrame, players: pl.DataFrame | None) -> pl.DataFrame:
    if players is None:
        return df
    known = players["player_id"].implode()
    unnamed = players.filter("master_unnamed")["player_id"].implode()
    names = dict(zip(players["player_id"].to_list(), players["master_name"].to_list(), strict=True))

    def mismatch(id_col: str, name_col: str) -> pl.Expr:
        master = pl.col(id_col).replace_strict(names, default=None, return_dtype=pl.String)
        return (master.is_not_null() & (master != _norm_name(pl.col(name_col)))).fill_null(False)

    checks = [
        ("player_not_in_master", ~pl.col("player_a_id").is_in(known) | ~pl.col("player_b_id").is_in(known)),
        ("player_unnamed_in_master", pl.col("player_a_id").is_in(unnamed) | pl.col("player_b_id").is_in(unnamed)),
        (
            "player_name_mismatch_master",
            mismatch("player_a_id", "player_a_name") | mismatch("player_b_id", "player_b_name"),
        ),
    ]
    exprs = [pl.when(e.fill_null(False)).then(pl.lit(code)).otherwise(None) for code, e in checks]
    return df.with_columns(pl.concat_list(pl.col(WARNINGS), *exprs).list.drop_nulls().alias(WARNINGS))


def rankings_duplicates(rankings: pl.DataFrame) -> int:
    """Number of (ranking_date, player) keys that appear more than once in the rankings files."""
    return rankings.group_by("ranking_date", "player_id").len().filter(pl.col("len") > 1).height


# The rankings are weekly; an entry older than this is not "as of" the tournament date
# (e.g. a player who dropped out of the rankings months earlier) and is not compared.
RANK_TOLERANCE_DAYS = 14


def rank_discrepancies(matches: pl.DataFrame, rankings: pl.DataFrame) -> pl.DataFrame:
    """One row per (match, side): match-file rank vs the rankings-file rank published on or up to
    RANK_TOLERANCE_DAYS before tournament_date.

    When a (date, player) key is duplicated in the rankings files, the best (lowest) rank is used.
    """
    ranks = (
        rankings.group_by("ranking_date", "player_id")
        .agg(pl.col("rank").min().alias("rankings_file_rank"))
        .sort("ranking_date")
    )
    long = pl.concat(
        [
            matches.select(
                "match_id", "season", "tournament_date",
                pl.lit(side.upper()).alias("side"),
                pl.col(f"player_{side}_id").alias("player_id"),
                pl.col(f"player_{side}_rank").alias("match_file_rank"),
            )
            for side in "ab"
        ]
    ).sort("tournament_date")  # fmt: skip
    joined = long.join_asof(
        ranks, left_on="tournament_date", right_on="ranking_date", by="player_id", strategy="backward",
        tolerance=f"{RANK_TOLERANCE_DAYS}d",
        check_sortedness=False,  # both sides are sorted on the date key above
    ).join_asof(
        ranks.rename({"ranking_date": "next_ranking_date", "rankings_file_rank": "next_rankings_file_rank"}),
        left_on="tournament_date", right_on="next_ranking_date", by="player_id", strategy="forward",
        tolerance=f"{RANK_TOLERANCE_DAYS}d", check_sortedness=False, allow_exact_matches=False,
    )  # fmt: skip
    # A match-file rank that differs from the list in force but equals the NEXT published list means
    # the source used a list published after tournament_date (typically events starting on a weekend).
    later = (pl.col("match_file_rank") == pl.col("next_rankings_file_rank")).fill_null(False) & (
        pl.col("match_file_rank") != pl.col("rankings_file_rank")
    ).fill_null(True)
    return joined.with_columns(
        pl.when(pl.col("match_file_rank").is_null() & pl.col("rankings_file_rank").is_null())
        .then(pl.lit("both_missing"))
        .when(pl.col("match_file_rank") == pl.col("rankings_file_rank"))
        .then(pl.lit("equal"))
        .when(later)
        .then(pl.lit("equals_later_list"))
        .when(pl.col("rankings_file_rank").is_null())
        .then(pl.lit("missing_in_rankings_file"))
        .when(pl.col("match_file_rank").is_null())
        .then(pl.lit("missing_in_match_file"))
        .otherwise(pl.lit("different"))
        .alias("comparison"),
        (pl.col("tournament_date") - pl.col("ranking_date")).dt.total_days().alias("ranking_age_days"),
        pl.when(later)
        .then((pl.col("next_ranking_date") - pl.col("tournament_date")).dt.total_days())
        .alias("later_list_days_after"),
    ).sort("match_id", "side")

"""Source and canonical schemas.

``CANONICAL_COLUMNS`` is the single source of truth for the validated match table:
it drives casting, schema enforcement, the stable-schema test and the generated
data dictionary (``tennis-data docs``).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# The 49 columns of every atp_matches_*.csv singles file at the pinned commit, in file order.
SOURCE_MATCH_COLUMNS = (
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand", "winner_ht", "winner_ioc",
    "winner_age", "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht",
    "loser_ioc", "loser_age", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
)  # fmt: skip

# Source columns without which a row cannot be identified or interpreted at all.
REQUIRED_SOURCE_VALUES = ("tourney_id", "tourney_date", "match_num", "winner_id", "loser_id", "round", "best_of")

# Serve statistics: canonical suffix -> source suffix (prefixed w_/l_ in the source).
STAT_FIELDS = {
    "ace": "ace",
    "df": "df",
    "svpt": "svpt",
    "first_in": "1stIn",
    "first_won": "1stWon",
    "second_won": "2ndWon",
    "sv_gms": "SvGms",
    "bp_saved": "bpSaved",
    "bp_faced": "bpFaced",
}
STAT_DESCRIPTIONS = {
    "ace": "aces served",
    "df": "double faults",
    "svpt": "serve points played",
    "first_in": "first serves in",
    "first_won": "first-serve points won",
    "second_won": "second-serve points won",
    "sv_gms": "service games played",
    "bp_saved": "break points saved",
    "bp_faced": "break points faced",
}

# Per-player attributes: canonical suffix -> (source suffix after winner_/loser_, dtype, description).
PLAYER_FIELDS = {
    "name": ("name", pl.String, "player name as printed in the match file"),
    "seed": ("seed", pl.Int64, "tournament seed (pre-tournament)"),
    "entry": ("entry", pl.String, "entry type: WC, Q, LL, PR, ALT, SE, ITF, ... (pre-tournament)"),
    "hand": ("hand", pl.String, "R, L, U(nknown), A(mbidextrous); biographical, not time-stamped"),
    "height_cm": ("ht", pl.Int64, "height in cm; biographical value from the source, not time-stamped"),
    "ioc": ("ioc", pl.String, "three-letter country code"),
    "age": ("age", pl.Float64, "age in years as of tournament_date"),
    "rank": ("rank", pl.Int64, "ATP rank as of tournament_date (source: match file)"),
    "rank_points": ("rank_points", pl.Int64, "ATP ranking points as of tournament_date (source: match file)"),
}


@dataclass(frozen=True)
class Column:
    name: str
    dtype: pl.DataType
    nullable: bool
    description: str
    group: str


def _player_columns(side: str) -> list[Column]:
    cols = [Column(f"player_{side}_id", pl.Int64, False, f"player {side.upper()} ID (source player_id)", "players")]
    for suffix, (_, dtype, desc) in PLAYER_FIELDS.items():
        cols.append(Column(f"player_{side}_{suffix}", dtype, True, f"player {side.upper()}: {desc}", "players"))
    return cols


def _stat_columns(side: str) -> list[Column]:
    return [
        Column(f"player_{side}_{s}", pl.Int64, True, f"player {side.upper()}: {d}", "serve_stats")
        for s, d in STAT_DESCRIPTIONS.items()
    ]


CANONICAL_COLUMNS: tuple[Column, ...] = tuple(
    [
        Column("match_id", pl.String, False,
               "Deterministic key '<family>:<tournament_id>:<match_num>'; unique in the validated table", "keys"),
        Column("tournament_id", pl.String, False, "Source tourney_id (first four characters are the year)", "tournament"),
        Column("tournament_name", pl.String, True, "Source tourney_name", "tournament"),
        Column("tournament_date", pl.Date, False,
               "Tournament start date (usually the Monday of the event week). NOT the match date.", "tournament"),
        Column("season", pl.Int32, False, "Season, taken from the source file name (partition key)", "tournament"),
        Column("surface", pl.String, True, "Hard | Clay | Grass | Carpet; null when the source omits it", "tournament"),
        Column("tournament_level", pl.String, False,
               "Source tourney_level: G, M, A, D, F, O, C, S, 15, 25, ... (see docs)", "tournament"),
        Column("draw_size", pl.Int64, True, "Draw size (often rounded up to a power of two)", "tournament"),
        Column("round", pl.String, False, "Q1-Q4, ER, R128, R64, R32, R16, RR, QF, SF, BR, F", "match"),
        Column("round_order", pl.Int32, False,
               "Ordinal of the round within an event, for sorting only; not a timestamp, ties are possible", "match"),
        Column("match_num", pl.Int64, False, "Source match_num (event-specific identifier, not chronological)", "match"),
        Column("best_of", pl.Int32, False, "Maximum number of sets: 1, 3 or 5", "match"),
        Column("match_date", pl.Date, True,
               "Date the match was played. Always null: the source does not record it.", "time"),
        Column("match_start_time", pl.Datetime("us", "UTC"), True,
               "Match start timestamp. Always null: the source does not record it and it is never inferred.", "time"),
        Column("time_precision", pl.String, False,
               "Finest known temporal resolution of this row; 'tournament_week' for all source rows", "time"),
        *_player_columns("a"),
        *_player_columns("b"),
        Column("winner_id", pl.Int64, False, "ID of the winner; always equals player_a_id or player_b_id", "outcome"),
        Column("player_a_won", pl.Boolean, False, "True when winner_id == player_a_id", "outcome"),
        Column("score", pl.String, True, "Source score string, games listed from the WINNER's perspective", "outcome"),
        Column("match_status", pl.String, False,
               "completed | retired | walkover | default | abandoned | unfinished | unknown", "outcome"),
        Column("minutes", pl.Int64, True, "Match duration in minutes, where available", "outcome"),
        *_stat_columns("a"),
        *_stat_columns("b"),
        Column("stats_status", pl.String, False,
               "complete (all 18 serve stats present) | partial | none", "serve_stats"),
        Column("quality_flags", pl.List(pl.String), False,
               "Warning codes for this row (kept in the validated table); empty list when clean", "quality"),
        Column("source_family", pl.String, False, "tour | qual_chall | futures", "provenance"),
        Column("source_repo", pl.String, False, "GitHub repository the raw file came from", "provenance"),
        Column("source_commit", pl.String, False, "Pinned commit SHA of the raw file", "provenance"),
        Column("source_file", pl.String, False, "Repository-relative path of the raw file", "provenance"),
        Column("source_row_number", pl.Int64, False, "1-based data row in the raw file (header excluded)", "provenance"),
        Column("source_sha256", pl.String, False, "SHA-256 of the raw file", "provenance"),
    ]
)  # fmt: skip

CANONICAL_SCHEMA: dict[str, pl.DataType] = {c.name: c.dtype for c in CANONICAL_COLUMNS}
NON_NULLABLE = tuple(c.name for c in CANONICAL_COLUMNS if not c.nullable)

STAT_COLUMNS = tuple(f"player_{side}_{s}" for side in "ab" for s in STAT_FIELDS)

# Extra columns carried only by the quarantine table.
QUARANTINE_EXTRA: dict[str, pl.DataType] = {
    "quarantine_reasons": pl.List(pl.String),
    "quarantine_categories": pl.List(pl.String),
    "duplicate_of": pl.String,
    "raw_record": pl.String,
}

ROUND_ORDER = {
    "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "ER": 5, "R128": 10, "R64": 11, "R32": 12, "R16": 13,
    "RR": 14, "QF": 15, "SF": 16, "BR": 17, "F": 18,
}  # fmt: skip

"""Row-level validation rules.

Severity semantics:
  error   -> the row is quarantined (moved to the quarantine table, never dropped).
  warning -> the row stays in the validated table; the code is listed in quality_flags.

Legitimately missing statistics are NOT a warning: they are represented by
null values and stats_status = 'none'. Only internally inconsistent or
impossible values are flagged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from tennis_data.config import Config
from tennis_data.normalization.normalize import ERRORS, LOSER_ID, WARNINGS
from tennis_data.normalization.schema import REQUIRED_SOURCE_VALUES, STAT_FIELDS
from tennis_data.normalization.score import consistent_with_best_of

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Rule:
    code: str
    severity: str
    category: str
    description: str
    expr: Callable[[Config], pl.Expr] | None = None  # None: raised elsewhere (normalize/duplicates/identity)


def _c(name: str) -> pl.Expr:
    return pl.col(name)


def _either(fn: Callable[[str], pl.Expr]) -> pl.Expr:
    """Rule violated by player A or player B; null comparisons count as not violated."""
    return (fn("a").fill_null(False)) | (fn("b").fill_null(False))


def _stat(side: str, name: str) -> pl.Expr:
    return pl.col(f"player_{side}_{name}")


def _score_inconsistent(cfg: Config) -> pl.Expr:
    return (pl.col("match_status") == "completed") & pl.struct("score", "best_of").map_elements(
        lambda r: consistent_with_best_of(r["score"], r["best_of"]) is False, return_dtype=pl.Boolean
    ).fill_null(False)


RULES: tuple[Rule, ...] = (
    # --- schema / parse failures (raised in normalize.py) ---
    *[Rule(f"missing_{c}", ERROR, "identity" if c.endswith("_id") else "schema",
           f"Required source value `{c}` is empty") for c in REQUIRED_SOURCE_VALUES],
    *[Rule(f"unparseable_{c}", ERROR, "schema", f"`{c}` is present but not a valid integer")
      for c in ("winner_id", "loser_id", "match_num", "best_of")],
    *[Rule(f"unparseable_{p}_{s}", ERROR, "stats", f"`{p}_{s}` is present but not a valid integer")
      for p in "wl" for s in STAT_FIELDS.values()],
    Rule("unparseable_tourney_date", ERROR, "schema", "`tourney_date` is not a valid YYYYMMDD date"),
    *[Rule(f"unparseable_{c}", WARNING, "schema", f"`{c}` is present but not numeric; left null")
      for c in ("draw_size", "minutes", "winner_seed", "loser_seed", "winner_ht", "loser_ht", "winner_rank",
                "loser_rank", "winner_rank_points", "loser_rank_points", "winner_age", "loser_age")],
    Rule("invalid_surface", ERROR, "categorical", "Surface is not one of the allowed surfaces"),
    Rule("missing_surface", WARNING, "categorical", "Surface is empty in the source (kept, surface = null)"),
    Rule("surface_case_normalized", WARNING, "categorical", "Surface differed only in letter case, e.g. 'clay'"),
    # --- categorical ---
    Rule("invalid_best_of", ERROR, "categorical", "best_of is not one of the allowed values",
         lambda cfg: _c("best_of").is_not_null() & ~_c("best_of").is_in(list(cfg.validation.allowed_best_of))),
    Rule("invalid_tournament_level", ERROR, "categorical", "tourney_level is empty or not a known code",
         lambda cfg: _c("tournament_level").is_null() | ~_c("tournament_level").is_in(list(cfg.validation.allowed_levels))),
    Rule("invalid_round", ERROR, "categorical", "round is not a known round code",
         lambda cfg: _c("round").is_not_null() & ~_c("round").is_in(list(cfg.validation.allowed_rounds))),
    Rule("date_season_mismatch", ERROR, "schema", "tournament_date is more than one year from the file's season",
         lambda cfg: (_c("tournament_date").dt.year().cast(pl.Int32) - _c("season")).abs() > 1),
    # --- identity / outcome ---
    Rule("same_player_both_sides", ERROR, "identity", "winner_id equals loser_id",
         lambda cfg: _c("winner_id") == _c(LOSER_ID)),
    Rule("placeholder_player_id", ERROR, "identity",
         "A player is a shared placeholder ID (e.g. 199999 'Unknown') standing for unidentified players",
         lambda cfg: _c("winner_id").is_in(list(cfg.validation.placeholder_player_ids))
         | _c(LOSER_ID).is_in(list(cfg.validation.placeholder_player_ids))),
    Rule("invalid_winner_reference", ERROR, "outcome", "winner_id is not one of player_a_id / player_b_id",
         lambda cfg: _c("winner_id").is_not_null() & _c("player_a_id").is_not_null()
         & (_c("winner_id") != _c("player_a_id")) & (_c("winner_id") != _c("player_b_id"))),
    Rule("unknown_match_status", WARNING, "outcome",
         "Score is empty, garbled ('?'), 'NA' or 'UNK'; the result (winner) is still recorded",
         lambda cfg: _c("match_status") == "unknown"),
    Rule("score_inconsistent_with_best_of", WARNING, "outcome",
         "Completed score does not give the winner exactly the sets needed for best_of", _score_inconsistent),
    # --- serve statistics: impossible values (error) ---
    Rule("stat_negative", ERROR, "stats", "A serve statistic is negative",
         lambda cfg: _either(lambda s: pl.min_horizontal([_stat(s, f) for f in STAT_FIELDS]) < 0)),
    Rule("first_in_exceeds_serve_points", ERROR, "stats", "first serves in > serve points",
         lambda cfg: _either(lambda s: _stat(s, "first_in") > _stat(s, "svpt"))),
    Rule("first_won_exceeds_first_in", ERROR, "stats", "first-serve points won > first serves in",
         lambda cfg: _either(lambda s: _stat(s, "first_won") > _stat(s, "first_in"))),
    Rule("second_serve_points_exceeded", ERROR, "stats",
         "second-serve points won + double faults > second-serve points (svpt - first_in)",
         lambda cfg: _either(lambda s: _stat(s, "second_won") + _stat(s, "df") > _stat(s, "svpt") - _stat(s, "first_in"))),
    Rule("aces_exceed_serve_points_won", ERROR, "stats", "aces > serve points won",
         lambda cfg: _either(lambda s: _stat(s, "ace") > _stat(s, "first_won") + _stat(s, "second_won"))),
    Rule("bp_saved_exceeds_faced", ERROR, "stats", "break points saved > break points faced",
         lambda cfg: _either(lambda s: _stat(s, "bp_saved") > _stat(s, "bp_faced"))),
    Rule("bp_faced_exceeds_serve_points", ERROR, "stats", "break points faced > serve points",
         lambda cfg: _either(lambda s: _stat(s, "bp_faced") > _stat(s, "svpt"))),
    Rule("breaks_exceed_service_games", ERROR, "stats", "break points lost (faced - saved) > service games",
         lambda cfg: _either(lambda s: _stat(s, "bp_faced") - _stat(s, "bp_saved") > _stat(s, "sv_gms"))),
    # A finished service game needs >= 4 points. In completed matches every counted game finished;
    # after a retirement/default the last counted game may never have started (observed: '1-0 RET', 0 points).
    Rule("too_few_points_for_service_games", ERROR, "stats",
         "serve points < 4 x service games (completed match) or < 4 x (service games - 1) (otherwise)",
         lambda cfg: _either(lambda s: _stat(s, "svpt") < 4 * (
             _stat(s, "sv_gms") - pl.when(_c("match_status") == "completed").then(0).otherwise(1)))),
    # --- serve statistics: implausible or inconsistent (warning) ---
    Rule("partial_stats", WARNING, "stats", "Some but not all of the 18 serve statistics are present",
         lambda cfg: _c("stats_status") == "partial"),
    Rule("stats_on_walkover", WARNING, "stats", "A walkover carries serve statistics",
         lambda cfg: (_c("match_status") == "walkover") & (_c("stats_status") != "none")),
    Rule("service_games_imbalance", WARNING, "stats",
         "|service games A - service games B| exceeds max_service_game_diff",
         lambda cfg: ((_c("player_a_sv_gms") - _c("player_b_sv_gms")).abs() > cfg.validation.max_service_game_diff).fill_null(False)),
    Rule("serve_points_without_service_games", WARNING, "stats", "serve points > 0 but service games = 0",
         lambda cfg: _either(lambda s: (_stat(s, "svpt") > 0) & (_stat(s, "sv_gms") == 0))),
    Rule("empty_stats_on_played_match", WARNING, "stats", "Played match whose stats record zero serve points for both",
         lambda cfg: (_c("stats_status") == "complete") & ~_c("match_status").is_in(["walkover"])
         & (_c("player_a_svpt") == 0) & (_c("player_b_svpt") == 0)),
    # --- other plausibility (warning) ---
    Rule("implausible_minutes", WARNING, "plausibility", "minutes <= 0 or > 1000",
         lambda cfg: ((_c("minutes") <= 0) | (_c("minutes") > 1000)).fill_null(False)),
    Rule("implausible_age", WARNING, "plausibility", "player age < 13 or > 60",
         lambda cfg: _either(lambda s: (_stat(s, "age") < 13) | (_stat(s, "age") > 60))),
    Rule("implausible_height", WARNING, "plausibility", "player height < 140 cm or > 230 cm",
         lambda cfg: _either(lambda s: (_stat(s, "height_cm") < 140) | (_stat(s, "height_cm") > 230))),
    # --- raised in duplicates.py ---
    Rule("duplicate_record", ERROR, "duplicate",
         "Same content as an earlier row (all source fields except match_num); first occurrence kept"),
    Rule("conflicting_match_id", ERROR, "conflict", "Two different rows share the same match_id"),
    Rule("conflicting_record", ERROR, "conflict",
         "Same tournament, round and player pair recorded more than once with different content "
         "(in a round where a pair can meet only once); all versions quarantined"),
    Rule("player_repeated_in_round", ERROR, "conflict",
         "A player appears in more than one match of the same knockout round of an event"),
    # --- raised in identity checks (players file) ---
    Rule("player_not_in_master", WARNING, "identity", "Player ID is absent from atp_players.csv"),
    Rule("player_name_mismatch_master", WARNING, "discrepancy", "Name in the match file differs from atp_players.csv"),
    Rule("player_unnamed_in_master", WARNING, "identity", "atp_players.csv lists the player's last name as 'Unknown'"),
)  # fmt: skip

RULES_BY_CODE = {r.code: r for r in RULES}


def category_of(code: str) -> str:
    rule = RULES_BY_CODE.get(code)
    return rule.category if rule else "other"


def apply_row_rules(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Append the codes of every violated expression rule to _errors / _warnings."""
    exprs = {ERROR: [], WARNING: []}
    for rule in RULES:
        if rule.expr is not None:
            exprs[rule.severity].append(
                pl.when(rule.expr(config).fill_null(False)).then(pl.lit(rule.code)).otherwise(None)
            )
    return df.with_columns(
        pl.concat_list(pl.col(ERRORS), *exprs[ERROR]).list.drop_nulls().alias(ERRORS),
        pl.concat_list(pl.col(WARNINGS), *exprs[WARNING]).list.drop_nulls().alias(WARNINGS),
    )

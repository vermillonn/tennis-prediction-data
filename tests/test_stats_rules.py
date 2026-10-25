"""Serve-statistic validation: impossible values are rejected, missing values are not."""

from __future__ import annotations

import pytest

from conftest import match_row
from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.validation.rules import RULES_BY_CODE


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"w_ace": "-1"}, "stat_negative"),
        ({"l_bpSaved": "-4"}, "stat_negative"),
        ({"w_1stIn": "61"}, "first_in_exceeds_serve_points"),
        ({"w_1stWon": "39"}, "first_won_exceeds_first_in"),
        ({"w_2ndWon": "21"}, "second_serve_points_exceeded"),  # 21 + 2 df > 60 - 38
        ({"l_ace": "36"}, "aces_exceed_serve_points_won"),
        ({"w_bpSaved": "2"}, "bp_saved_exceeds_faced"),
        ({"l_bpFaced": "70", "l_bpSaved": "60"}, "bp_faced_exceeds_serve_points"),
        ({"l_bpFaced": "15", "l_bpSaved": "3"}, "breaks_exceed_service_games"),  # 12 breaks > 10 games
        ({"w_svpt": "39", "w_1stIn": "30", "w_1stWon": "25", "w_2ndWon": "5"}, "too_few_points_for_service_games"),
        ({"w_svpt": "60.5"}, "unparseable_w_svpt"),
        ({"w_svpt": "sixty"}, "unparseable_w_svpt"),
    ],
)
def test_impossible_stats_are_quarantined(validate_rows, overrides, code):
    _, q = validate_rows([match_row(1, 100, 200, **overrides), match_row(2, 300, 400)])
    assert q.height == 1, f"expected {code} to quarantine the row"
    assert code in q["quarantine_reasons"][0].to_list()
    assert RULES_BY_CODE[code].severity == "error"
    assert "stats" in q["quarantine_categories"][0].to_list()


def test_valid_fixture_row_passes_every_rule(validate_rows):
    v, q = validate_rows([match_row(1, 100, 200)])
    assert q.height == 0
    assert v["quality_flags"][0].to_list() == []
    assert v["stats_status"][0] == "complete"


def test_missing_stats_are_legitimate_not_invalid(validate_rows):
    v, q = validate_rows([match_row(1, 100, 200, stats=False)])
    assert q.height == 0
    row = v.row(0, named=True)
    assert row["stats_status"] == "none"
    assert row["quality_flags"] == []
    assert all(row[f"player_{s}_svpt"] is None for s in "ab")  # nothing imputed


def test_partial_stats_are_kept_and_flagged(validate_rows):
    v, q = validate_rows([match_row(1, 100, 200, w_ace=None, l_ace=None)])
    assert q.height == 0
    assert v["stats_status"][0] == "partial"
    assert "partial_stats" in v["quality_flags"][0].to_list()
    assert v["player_a_ace"][0] is None and v["player_b_ace"][0] is None


def test_retirement_allows_an_unstarted_final_service_game(validate_rows):
    # '1-0 RET': the loser is credited with one service game but served no points.
    overrides = {"score": "1-0 RET", "w_svpt": "4", "w_1stIn": "3", "w_1stWon": "3", "w_2ndWon": "1",
                 "w_SvGms": "1", "w_ace": "1", "w_df": "0", "w_bpSaved": "0", "w_bpFaced": "0",
                 "l_svpt": "0", "l_1stIn": "0", "l_1stWon": "0", "l_2ndWon": "0", "l_SvGms": "1",
                 "l_ace": "0", "l_df": "0", "l_bpSaved": "0", "l_bpFaced": "0"}  # fmt: skip
    v, q = validate_rows([match_row(1, 100, 200, **overrides)])
    assert q.height == 0 and v["match_status"][0] == "retired"
    # The same numbers in a completed match are impossible.
    v, q = validate_rows([match_row(1, 100, 200, **{**overrides, "score": "6-0 6-0"})])
    assert "too_few_points_for_service_games" in q["quarantine_reasons"][0].to_list()


@pytest.mark.parametrize(
    "score, status",
    [("6-4 6-4", "completed"), ("6-4 2-1 RET", "retired"), ("W/O", "walkover"), ("Walkover", "walkover"),
     ("6-3 DEF", "default"), ("6-2 Def.", "default"), ("6-4 3-1 ABD", "abandoned"),
     ("6-6 Played and unfinished", "unfinished"), ("6-2 ?-?", "unknown"), ("NA", "unknown"), (None, "unknown")],
)  # fmt: skip
def test_match_status(validate_rows, score, status):
    v, _ = validate_rows([match_row(1, 100, 200, stats=False, score=score)])
    assert v["match_status"][0] == status
    assert ("unknown_match_status" in v["quality_flags"][0].to_list()) == (status == "unknown")


def test_walkover_with_stats_is_flagged(validate_rows):
    v, _ = validate_rows([match_row(1, 100, 200, score="W/O")])
    assert "stats_on_walkover" in v["quality_flags"][0].to_list()


def test_categorical_checks(validate_rows):
    rows = [match_row(1, 100, 200, surface="Sand"), match_row(2, 300, 400, surface=None),
            match_row(3, 500, 600, surface="clay"), match_row(4, 700, 800, best_of="4"),
            match_row(5, 900, 100, tourney_level="Z", tourney_id="2020-0002")]  # fmt: skip
    v, q = validate_rows(rows)
    reasons = {r["match_num"]: r["quarantine_reasons"] for r in q.iter_rows(named=True)}
    assert reasons == {1: ["invalid_surface"], 4: ["invalid_best_of"], 5: ["invalid_tournament_level"]}
    kept = {r["match_num"]: r for r in v.iter_rows(named=True)}
    assert kept[2]["surface"] is None and "missing_surface" in kept[2]["quality_flags"]
    assert kept[3]["surface"] == "Clay" and "surface_case_normalized" in kept[3]["quality_flags"]


def test_score_inconsistent_with_best_of_is_a_warning(validate_rows):
    v, q = validate_rows([match_row(1, 100, 200, best_of="5")])  # 2 sets won in a best-of-5
    assert q.height == 0
    assert "score_inconsistent_with_best_of" in v["quality_flags"][0].to_list()


def test_fixture_covers_full_source_schema():
    assert set(match_row(1, 100, 200)) == set(SOURCE_MATCH_COLUMNS)

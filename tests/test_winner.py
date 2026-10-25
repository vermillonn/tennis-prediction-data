"""Winner identification and identity checks."""

from __future__ import annotations

import polars as pl
import pytest

from conftest import match_row
from tennis_data.validation.contract import SchemaViolation, enforce_contract


def test_winner_is_source_winner_and_one_of_the_players(validate_rows):
    rows = [match_row(1, 100, 200), match_row(2, 300, 400), match_row(3, 500, 600)]
    v, q = validate_rows(rows)
    assert q.height == 0
    winners = dict(zip(v["match_id"], v["winner_id"], strict=True))
    assert winners == {"tour:2020-0001:1": 100, "tour:2020-0001:2": 300, "tour:2020-0001:3": 500}
    assert v.filter(
        (pl.col("winner_id") != pl.col("player_a_id")) & (pl.col("winner_id") != pl.col("player_b_id"))
    ).is_empty()
    assert (v["player_a_won"] == (v["winner_id"] == v["player_a_id"])).all()


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"winner_id": None}, "missing_winner_id"),
        ({"loser_id": None}, "missing_loser_id"),
        ({"winner_id": "abc"}, "unparseable_winner_id"),
        ({"loser_id": "100"}, "same_player_both_sides"),
        ({"winner_id": "199999"}, "placeholder_player_id"),
        ({"tourney_date": "2020-01-06"}, "unparseable_tourney_date"),
        ({"tourney_id": None}, "missing_tourney_id"),
    ],
)
def test_bad_identity_is_quarantined_not_dropped(validate_rows, overrides, reason):
    v, q = validate_rows([match_row(1, 100, 200, **overrides), match_row(2, 300, 400)])
    assert v.height == 1 and q.height == 1
    assert reason in q["quarantine_reasons"][0].to_list()
    assert q["raw_record"][0]  # the original source values are preserved


def test_contract_rejects_winner_outside_the_pair(validate_rows):
    v, _ = validate_rows([match_row(1, 100, 200)])
    broken = v.with_columns(pl.lit(999, dtype=pl.Int64).alias("winner_id"))
    with pytest.raises(SchemaViolation, match="winner_id is neither player"):
        enforce_contract(broken)


def test_contract_rejects_fabricated_timestamps(validate_rows):
    v, _ = validate_rows([match_row(1, 100, 200)])
    broken = v.with_columns(pl.col("tournament_date").alias("match_date"))
    with pytest.raises(SchemaViolation, match="match timestamps"):
        enforce_contract(broken)


def test_timestamps_are_explicitly_missing(validate_rows):
    v, _ = validate_rows([match_row(1, 100, 200)])
    assert v["match_date"].is_null().all()
    assert v["match_start_time"].is_null().all()
    assert (v["time_precision"] == "tournament_week").all()


def test_player_master_checks(validate_rows):
    v, _ = validate_rows([match_row(1, 100, 950), match_row(2, 300, 400, winner_name="Charles Cole")])
    flags = dict(zip(v["match_num"], v["quality_flags"].to_list(), strict=True))
    assert "player_not_in_master" in flags[1]
    assert "player_name_mismatch_master" in flags[2]


def test_missing_match_num_is_quarantined(validate_rows):
    row = match_row(1, 100, 200)
    row["match_num"] = ""
    _, q = validate_rows([row, match_row(2, 300, 400)])
    assert q["quarantine_reasons"][0].to_list() == ["missing_match_num"]
    assert q["match_id"][0] is None

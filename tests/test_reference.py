"""Cross-checks against the rankings reference files."""

from __future__ import annotations

from datetime import date

import polars as pl

from tennis_data.validation.reference import rank_discrepancies


def test_rank_comparison_classes():
    matches = pl.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "season": [2020] * 4,
            "tournament_date": [date(2020, 1, 6), date(2020, 1, 5), date(2020, 1, 6), date(2020, 3, 2)],
            "player_a_id": [1, 1, 2, 3],
            "player_a_rank": [10, 9, 50, None],
            "player_b_id": [2, 2, 3, 4],
            "player_b_rank": [20, 22, None, 7],
        }
    )
    rankings = pl.DataFrame(
        {
            "ranking_date": [
                date(2019, 12, 30),
                date(2020, 1, 6),
                date(2019, 12, 30),
                date(2020, 1, 6),
                date(2020, 1, 6),
            ],
            "player_id": [1, 1, 2, 2, 3],
            "rank": [11, 10, 21, 20, 30],
        }
    )
    out = {(r["match_id"], r["side"]): r for r in rank_discrepancies(matches, rankings).iter_rows(named=True)}
    assert out[("m1", "A")]["comparison"] == "equal"
    assert out[("m1", "B")]["comparison"] == "equal"
    # Sunday start: the list in force is 2019-12-30; 9 and 22 match neither that nor the next list.
    assert out[("m2", "A")]["comparison"] == "different"
    assert out[("m2", "B")]["comparison"] == "different"
    assert out[("m3", "A")]["comparison"] == "different"  # 50 vs 21
    assert out[("m3", "B")]["comparison"] == "missing_in_match_file"
    # Last list for player 3 is > 14 days old: not comparable.
    assert out[("m4", "A")]["comparison"] == "both_missing"
    assert out[("m4", "B")]["comparison"] == "missing_in_rankings_file"


def test_rank_from_a_later_list_is_identified():
    matches = pl.DataFrame(
        {"match_id": ["m"], "season": [2020], "tournament_date": [date(2020, 1, 5)],  # a Sunday
         "player_a_id": [1], "player_a_rank": [10], "player_b_id": [2], "player_b_rank": [21]}
    )  # fmt: skip
    rankings = pl.DataFrame(
        {
            "ranking_date": [date(2019, 12, 30), date(2020, 1, 6)] * 2,
            "player_id": [1, 1, 2, 2],
            "rank": [11, 10, 21, 20],
        }
    )
    out = {r["side"]: r for r in rank_discrepancies(matches, rankings).iter_rows(named=True)}
    assert out["A"]["comparison"] == "equals_later_list" and out["A"]["later_list_days_after"] == 1
    assert out["B"]["comparison"] == "equal" and out["B"]["later_list_days_after"] is None

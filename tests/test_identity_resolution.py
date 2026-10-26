"""Overlap assignment: the archive ID used in the same match, found through the opponent, never the disputed mapping."""

from __future__ import annotations

from datetime import date

import polars as pl

from tennis_data.operations.identity_resolution import overlap_assignment


def test_overlap_assignment_pairs_through_opponent_round_and_week():
    hist = pl.DataFrame({"player_a_id": [10, 10, 30, 10], "player_b_id": [20, 21, 20, 22],
                         "winner_id": [10, 21, 20, 10], "round": ["R32", "R16", "R32", "R32"],
                         "tournament_date": [date(2026, 3, 2), date(2026, 3, 2), date(2026, 3, 2), date(2026, 3, 30)]})  # fmt: skip
    starts = {"t1": date(2026, 3, 1), "t2": date(2026, 3, 29)}
    canon = {"Q": 20, "R": 21, "S": 22}  # opponents' mappings; "P" itself is never looked up
    matches = [{"tournament_id": "t1", "round": "R32", "player1_id": "P", "player2_id": "Q", "winner_id": "P"},
               {"tournament_id": "t1", "round": "R16", "player1_id": "P", "player2_id": "R", "winner_id": "P"},
               {"tournament_id": "t2", "round": "R32", "player1_id": "U", "player2_id": "P", "winner_id": "U"},
               {"tournament_id": "t2", "round": "F", "player1_id": "P", "player2_id": "S", "winner_id": "P"}]  # fmt: skip
    out = overlap_assignment(matches, starts, canon, hist, {"P"})
    # R32 vs Q: archive has 10-20 and 30-20 in that round and week -> ambiguous; R16 vs R: 10 lost there -> winner
    # differs; U unmapped; final: no archive match in that round
    assert out["P"] == {"several_archive_matches": 1, "10:winner_differs": 1, "opponent_unmapped": 1,
                        "no_archive_match": 1}  # fmt: skip
    hist2 = hist.filter(pl.col("player_a_id") != 30)
    assert overlap_assignment(matches[:1], starts, canon, hist2, {"P"})["P"] == {"10": 1}

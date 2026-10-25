"""Player A/B assignment is deterministic and independent of the winner."""

from __future__ import annotations

import random

import polars as pl

from conftest import match_row
from tennis_data.normalization.ordering import assign_players


def test_assignment_is_symmetric_in_its_arguments():
    for i in range(500):
        mid = f"tour:2020-0001:{i}"
        assert assign_players(mid, 100, 200) == assign_players(mid, 200, 100)


def test_assignment_is_stable_across_calls():
    # Pinned values: changing the hashing scheme would silently reshuffle every historical row.
    assert assign_players("tour:2024-0339:300", 105777, 208029) == (208029, 105777)
    assert assign_players("tour:2024-0339:299", 208029, 126128) == (126128, 208029)


def test_assignment_is_balanced_and_not_id_sorted():
    rng = random.Random(0)
    lower_is_a = 0
    n = 20_000
    for i in range(n):
        x, y = rng.sample(range(100_000, 300_000), 2)
        a, _ = assign_players(f"tour:x:{i}", x, y)
        lower_is_a += a == min(x, y)
    assert abs(lower_is_a / n - 0.5) < 0.02


def test_reversing_the_result_keeps_players_and_flips_the_label(validate_rows):
    """Same match, opposite winner: A/B identities must not move; only the outcome changes."""
    original = match_row(1, 100, 200)
    reversed_ = match_row(1, 200, 100)
    # Swap the per-player source fields too, so it is the same two players with the opposite result.
    for src_w, src_l in (("winner_", "loser_"), ("w_", "l_")):
        for col in [c for c in original if c.startswith(src_w)]:
            reversed_[col], reversed_[src_l + col[len(src_w) :]] = original[src_l + col[len(src_w) :]], original[col]

    v1, _ = validate_rows([original])
    v2, _ = validate_rows([reversed_])
    r1, r2 = v1.row(0, named=True), v2.row(0, named=True)
    assert (r1["player_a_id"], r1["player_b_id"]) == (r2["player_a_id"], r2["player_b_id"])
    assert r1["player_a_won"] != r2["player_a_won"]
    assert r1["winner_id"] == 100 and r2["winner_id"] == 200
    # Per-player attributes follow the player, not the result.
    for col in ("player_a_name", "player_a_rank", "player_a_svpt", "player_b_ace", "player_b_height_cm"):
        assert r1[col] == r2[col]


def test_player_attributes_follow_the_player(validate_rows):
    # One match per event, so the same pair never meets twice in a round.
    v, _ = validate_rows([match_row(i, 100, 200, tourney_id=f"2020-{i:04d}") for i in range(1, 30)])
    for r in v.iter_rows(named=True):
        a_is_winner = r["player_a_id"] == 100
        assert r["player_a_won"] == a_is_winner
        assert r["player_a_name"] == ("Alice Archer" if a_is_winner else "Bob Baker")
        assert r["player_a_svpt"] == (60 if a_is_winner else 65)
        assert r["player_a_rank"] == (10 if a_is_winner else 20)
    assert 0 < v["player_a_won"].sum() < v.height  # both orientations occur
    assert v.filter(pl.col("player_a_id") == pl.col("player_b_id")).height == 0

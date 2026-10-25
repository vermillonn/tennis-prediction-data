"""Elo formula, zero-sum updates, frozen batches, order invariance and leakage."""

from __future__ import annotations

import math
from dataclasses import replace

import polars as pl
import pytest

from elo_helpers import frame, random_frame
from tennis_data.elo.config import EloParams
from tennis_data.elo.engine import expected_score, k_factor, match_k, run_elo, surface_key, surface_probability

P = EloParams()  # 1500 start, constant K = 32


def _pred(run, mid):
    return run.predictions.filter(pl.col("match_id") == mid).row(0, named=True)


def _final(run, rating_type="global"):
    h = run.history.filter(pl.col("rating_type") == rating_type)
    return dict(h.group_by("player_id").agg(pl.col("rating_after").last()).iter_rows())


# --- formula -------------------------------------------------------------------------------


def test_expected_score_values():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(1 / (1 + 10**-1))
    assert expected_score(1500, 1900) == pytest.approx(1 / (1 + 10**1))
    for a, b in ((1500, 1600), (2100, 1350), (1720.5, 1719.25)):
        assert expected_score(a, b) + expected_score(b, a) == pytest.approx(1.0)


def test_single_update_matches_hand_calculation():
    run = run_elo(frame([("m1", 0, 1, 2, True), ("m2", 1, 1, 2, True)]), P, record_history=True)
    assert _pred(run, "m1")["player_a_elo_before"] == 1500.0
    r = _pred(run, "m2")
    assert r["player_a_elo_before"] == pytest.approx(1516.0)
    assert r["player_b_elo_before"] == pytest.approx(1484.0)
    assert r["global_probability_a"] == pytest.approx(expected_score(1516, 1484))
    assert r["global_match_count_a"] == 1


def test_decay_k_is_shared_by_both_players():
    p = EloParams(k_mode="decay", k_numerator=250, k_offset=5, k_shape=0.4)
    assert k_factor(p, 0) == pytest.approx(250 / 5**0.4)
    assert match_k(p, 0, 20) == pytest.approx((k_factor(p, 0) + k_factor(p, 20)) / 2)


@pytest.mark.parametrize("params", [P, EloParams(k_mode="decay"), EloParams(k_constant=17.0, surface_init="constant")])
def test_updates_are_zero_sum(params):
    df = random_frame(seed=3)
    run = run_elo(df, params, record_history=True)
    final = _final(run)
    assert math.fsum(final.values()) == pytest.approx(1500.0 * len(final), abs=1e-6)
    for s in ("Hard", "Clay", "Grass"):
        fs = _final(run, s)
        assert math.fsum(fs.values()) == pytest.approx(1500.0 * len(fs), abs=1e-6)


def test_every_batch_close_is_zero_sum():
    run = run_elo(random_frame(seed=4), EloParams(k_mode="decay"), record_history=True)
    state: dict = {}
    for batch, part in run.history.filter(pl.col("rating_type") == "global").group_by("batch", maintain_order=True):
        before = math.fsum(state.get(pid, 1500.0) for pid in part["player_id"])
        after = math.fsum(part["rating_after"])
        assert after == pytest.approx(before, abs=1e-9), batch
        state.update(zip(part["player_id"], part["rating_after"], strict=True))


# --- chronology ----------------------------------------------------------------------------


def test_predictions_within_a_batch_use_the_frozen_state():
    # Week 0: player 1 beats 2, then (same batch) plays 3. Week 1: 1 vs 3 again.
    run = run_elo(frame([("a", 0, 1, 2, True), ("b", 0, 1, 3, True), ("c", 1, 1, 3, True)]), P)
    assert _pred(run, "a")["player_a_elo_before"] == 1500.0
    assert _pred(run, "b")["player_a_elo_before"] == 1500.0  # not updated by match "a"
    assert _pred(run, "b")["global_match_count_a"] == 0
    # Both week-0 results are applied simultaneously from 1500: 1 gains 16 + 16.
    assert _pred(run, "c")["player_a_elo_before"] == pytest.approx(1532.0)
    assert _pred(run, "c")["player_b_elo_before"] == pytest.approx(1484.0)


def test_simultaneous_update_is_not_sequential():
    # Sequential Elo would give player 1 a smaller second gain (expected score above 0.5 after the first win).
    run = run_elo(frame([("a", 0, 1, 2, True), ("b", 0, 1, 3, True), ("z", 1, 1, 4, True)]), P)
    assert _pred(run, "z")["player_a_elo_before"] == pytest.approx(1500 + 2 * 32 * 0.5)


def test_multi_week_results_wait_for_the_last_batch():
    # Match "long" belongs to a 2-week event: predicted in week 0, released at the close of week 1.
    run = run_elo(frame([("long", 0, 1, 2, True, "Hard", 1), ("w1", 1, 1, 5, True), ("w2", 2, 1, 5, True)]), P)
    assert _pred(run, "w1")["player_a_elo_before"] == 1500.0
    assert _pred(run, "w2")["player_a_elo_before"] == pytest.approx(1500 + 16 + 16)


def test_release_before_prediction_is_rejected():
    df = frame([("x", 1, 1, 2, True)]).with_columns(pl.col("prediction_batch").dt.offset_by("1w"))
    with pytest.raises(ValueError, match="release_batch earlier"):
        run_elo(df, P)


def test_batch_order_invariance_is_bitwise():
    df = random_frame(n_weeks=25, per_week=8, seed=5)
    base = run_elo(df, EloParams(k_mode="decay"), record_history=True)
    for seed in range(5):
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        other = run_elo(shuffled, EloParams(k_mode="decay"), record_history=True, canonical_order=False)
        assert other.predictions.equals(base.predictions)
        assert other.history.equals(base.history)


def test_deterministic_replay():
    df = random_frame(seed=6)
    a, b = run_elo(df, P, record_history=True), run_elo(df, P, record_history=True)
    assert a.predictions.equals(b.predictions) and a.history.equals(b.history)


def test_no_information_from_later_batches():
    """Predictions for batch b are identical when every later match is removed from the input."""
    df = random_frame(n_weeks=15, seed=8)
    full = run_elo(df, P).predictions
    for b in sorted(df["prediction_batch"].unique()):
        truncated = run_elo(df.filter(pl.col("prediction_batch") <= b), P).predictions
        mine = df.filter(pl.col("prediction_batch") == b)["match_id"]
        assert full.filter(pl.col("match_id").is_in(mine.implode())).equals(
            truncated.filter(pl.col("match_id").is_in(mine.implode()))
        )


def test_future_outcome_perturbation():
    """Flipping every result from a cutoff batch on cannot change any prediction up to that batch."""
    df = random_frame(n_weeks=30, seed=9)
    cutoff = sorted(df["prediction_batch"].unique())[15]
    perturbed = df.with_columns(
        pl.when(pl.col("prediction_batch") >= cutoff).then(~pl.col("target_a_wins")).otherwise(pl.col("target_a_wins"))
    )
    before = pl.col("match_id").is_in(df.filter(pl.col("prediction_batch") <= cutoff)["match_id"].implode())
    a, b = run_elo(df, P).predictions, run_elo(perturbed, P).predictions
    assert a.filter(before).equals(b.filter(before))
    assert not a.filter(~before).equals(b.filter(~before))  # the perturbation is real


# --- surface ratings ---------------------------------------------------------------------------


def test_surface_initialization_constant_and_global():
    rows = [("h1", 0, 1, 2, True, "Hard"), ("h2", 1, 1, 3, True, "Hard"), ("c1", 2, 1, 4, True, "Clay")]
    const = run_elo(frame(rows), P)
    glob = run_elo(frame(rows), replace(P, surface_init="global"))
    assert _pred(const, "c1")["surface_elo_a_before"] == 1500.0
    g = _pred(glob, "c1")
    assert g["surface_elo_a_before"] == pytest.approx(g["player_a_elo_before"])  # first clay match starts at global
    assert g["surface_match_count_a"] == 0
    assert _pred(const, "c1")["surface_match_count_a"] == 0
    assert _pred(const, "h2")["surface_match_count_a"] == 1


def test_surfaces_are_independent_pools():
    run = run_elo(frame([("h", 0, 1, 2, True, "Hard"), ("c", 1, 1, 2, True, "Clay"), ("g", 2, 1, 2, True, "Grass")]), P)
    assert _pred(run, "c")["surface_elo_a_before"] == 1500.0
    # Global keeps accumulating across surfaces: +16, then +32 * (1 - E(1516, 1484)).
    assert _pred(run, "g")["player_a_elo_before"] == pytest.approx(1516 + 32 * (1 - expected_score(1516, 1484)))
    assert _pred(run, "g")["surface_elo_a_before"] == 1500.0


def test_shrinkage_weights():
    df = pl.DataFrame(
        {
            "player_a_elo_before": [1600.0, 1600.0, 1600.0],
            "player_b_elo_before": [1500.0, 1500.0, 1500.0],
            "surface_elo_a_before": [1400.0, 1400.0, None],
            "surface_elo_b_before": [1500.0, 1500.0, None],
            "surface_match_count_a": [10, 0, 0],
            "surface_match_count_b": [30, 30, 0],
        }
    )
    pure = surface_probability(df, 0.0)
    assert pure["surface_effective_elo_a"][0] == 1400.0  # m = 0: pure surface rating
    assert pure["surface_effective_elo_a"][1] == 1600.0  # no surface history: weight 0
    shrunk = surface_probability(df, 10.0)
    assert shrunk["surface_effective_elo_a"][0] == pytest.approx(0.5 * 1400 + 0.5 * 1600)
    assert shrunk["surface_effective_elo_b"][0] == pytest.approx(0.75 * 1500 + 0.25 * 1500)
    huge = surface_probability(df, 1e12)
    assert huge["surface_probability_a"][0] == pytest.approx(expected_score(1600, 1500))
    # Missing surface: the surface model is the global model.
    assert pure["surface_probability_a"][2] == pytest.approx(expected_score(1600, 1500))


def test_missing_surface_updates_no_surface_rating():
    run = run_elo(frame([("x", 0, 1, 2, True, None), ("y", 1, 1, 2, True, "Hard")]), P)
    y = _pred(run, "y")
    assert y["surface_match_count_a"] == 0 and y["surface_elo_a_before"] == 1500.0
    assert y["global_match_count_a"] == 1


def test_carpet_policies():
    assert surface_key("Carpet", "separate") == "Carpet"
    assert surface_key("Carpet", "hard") == "Hard"
    assert surface_key("Carpet", "none") is None
    assert surface_key(None, "separate") is None
    assert surface_key("Clay", "hard") == "Clay"


def test_symmetry_under_player_swap():
    df = random_frame(seed=10)
    swapped = df.with_columns(
        pl.col("player_b_id").alias("player_a_id"), pl.col("player_a_id").alias("player_b_id"), ~pl.col("target_a_wins")
    )
    a, b = run_elo(df, P).predictions, run_elo(swapped, P).predictions
    assert (a["global_probability_a"] + b["global_probability_a"] - 1).abs().max() < 1e-12
    assert (a["player_a_elo_before"] - b["player_b_elo_before"]).abs().max() < 1e-9

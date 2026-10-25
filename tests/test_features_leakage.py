"""Chronology and leakage: frozen batches, release delays, perturbation, truncation, order invariance."""

from __future__ import annotations

import random

import polars as pl
import pytest

from feature_helpers import SAT0, cfg, frame, match, row, run
from tennis_data.features.observations import observations
from tennis_data.features.pipeline import assemble
from tennis_data.features.snapshots import build_snapshots

OPP = cfg(opponent=True)


def _random_rows(n_weeks=40, per_week=6, seed=0, multi_week_every=7):
    rng = random.Random(seed)
    rows = []
    for w in range(n_weeks):
        players = rng.sample(range(100, 130), 2 * per_week)
        for j in range(per_week):
            a, b = players[2 * j], players[2 * j + 1]
            sp_a, sp_b = rng.randint(40, 110), rng.randint(40, 110)
            rows.append(match(
                f"m{w:03d}-{j}", w, a, b, a_sp=(sp_a, rng.randint(0, sp_a)), b_sp=(sp_b, rng.randint(0, sp_b)),
                surface=rng.choice(["Hard", "Clay", "Grass", "Carpet", None]),
                release_weeks=1 if w % multi_week_every == 0 else 0,
                eligible=rng.random() > 0.15, round_order=rng.choice([10, 11, 12, 13, 15, 16, 18]),
            ))  # fmt: skip
    return rows


def _table(rows, c=OPP):
    f = frame(rows)
    return assemble(build_snapshots(f, observations(f), c).features, c)


def _before(t, cutoff):
    return t.filter(pl.col("prediction_batch") <= cutoff)


def test_all_matches_in_a_batch_share_the_frozen_state():
    rows = [
        match("w0", 0, 100, 200),
        match("x1", 1, 100, 300, a_sp=(90, 80)),
        match("x2", 1, 100, 400),
        match("x3", 1, 400, 100),
    ]
    t = run(rows, OPP)
    a1, a2, b3 = row(t, "x1"), row(t, "x2"), row(t, "x3")
    for k in (
        "service_points_observed",
        "spw_shrunk",
        "spw_decay52w",
        "spw_last5",
        "serve_adjusted",
        "return_adjusted",
    ):
        assert a1[f"a_{k}"] == a2[f"a_{k}"] == b3[f"b_{k}"]
    assert a1["a_service_points_observed"] == 60  # only week 0, not the 90 points of x1 in the same batch


def test_own_match_statistics_never_enter_own_snapshot():
    base = [match("w0", 0, 100, 200), match("own", 1, 100, 300, a_sp=(70, 50))]
    changed = [match("w0", 0, 100, 200), match("own", 1, 100, 300, a_sp=(70, 10), b_sp=(99, 1))]
    a, b = row(run(base, OPP), "own"), row(run(changed, OPP), "own")
    assert {k: v for k, v in a.items()} == {k: v for k, v in b.items()}


def test_multi_week_statistics_are_not_available_early():
    rows = [
        match("slam", 0, 100, 200, a_sp=(100, 90), release_weeks=1),
        match("w1", 1, 100, 300),
        match("w2", 2, 100, 400),
    ]
    t = run(rows)
    assert row(t, "w1")["a_service_points_observed"] == 0  # released only at the close of week 1
    assert row(t, "w2")["a_service_points_observed"] == 100 + 60


@pytest.mark.parametrize("seed", [1, 2])
def test_future_statistics_perturbation(seed):
    from datetime import timedelta

    rows = _random_rows(seed=seed)
    cutoff = SAT0 + timedelta(weeks=20)
    rng = random.Random(99)
    perturbed = []
    for r in rows:
        r = dict(r)
        if r["prediction_batch"] >= cutoff:
            r["a_sp_played"] = rng.randint(40, 120)
            r["a_sp_won"] = rng.randint(0, r["a_sp_played"])
            r["stats_eligible"] = rng.random() > 0.5
        perturbed.append(r)
    a, b = _table(rows), _table(perturbed)
    # Rows in the cutoff batch itself are perturbed too (the strict same-batch case). Their own
    # `stats_eligible` metadata flag legitimately changes; every pre-match feature must not.
    own = ["stats_eligible"]
    assert _before(a, cutoff).drop(own).equals(_before(b, cutoff).drop(own))
    assert not a.drop(own).equals(b.drop(own))  # the perturbation is real


def test_future_winner_perturbation():
    rows = _random_rows(seed=3)
    flipped = [{**r, "player_a_won": not r["player_a_won"]} for r in rows]
    assert _table(rows).equals(_table(flipped))  # features never read outcomes


def test_input_order_invariance():
    rows = _random_rows(seed=4)
    base = _table(rows)
    for s in range(3):
        shuffled = rows[:]
        random.Random(s).shuffle(shuffled)
        assert _table(shuffled).equals(base)


def test_truncation_at_batch_boundary_preserves_earlier_features():
    from datetime import timedelta

    rows = _random_rows(seed=5)
    full = _table(rows)
    for weeks in (5, 17, 30):
        cutoff = SAT0 + timedelta(weeks=weeks)
        trunc = _table([r for r in rows if r["prediction_batch"] <= cutoff])
        assert _before(full, cutoff).equals(_before(trunc, cutoff))


def test_opponent_model_uses_only_released_data():
    t = _table(_random_rows(seed=6))
    fitted = t.filter(pl.col("adjustment_fit_batch").is_not_null())
    assert fitted.height > 0
    assert (fitted["adjustment_fit_batch"] <= fitted["prediction_batch"]).all()
    adj = fitted.select("a_serve_adjusted", "b_return_adjusted", "a_spw_vs_b").to_numpy()
    assert ((adj > 0) & (adj < 1)).all()


def test_reproducible_output_hash():
    import hashlib
    import io

    def digest(table):
        buf = io.BytesIO()
        table.write_parquet(buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    rows = _random_rows(seed=7)
    assert digest(_table(rows)) == digest(_table(rows))

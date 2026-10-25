"""Regression tests: recency, missingness and activity/reliability inputs use only released history.

The Phase 3B derived inputs are applied to the Phase 3A engine output for small synthetic histories. Each
input must be a function of observations whose label was released before the prediction batch: not the match
itself, not other matches of the same batch, not later batches, and not week 2 of a two-week event early.
"""

from __future__ import annotations

import math
import random
from datetime import timedelta

import polars as pl
import pytest

from feature_helpers import SAT0, cfg, frame, match, run
from tennis_data.features.observations import observations
from tennis_data.features.pipeline import assemble
from tennis_data.features.snapshots import build_snapshots
from tennis_data.modeling.dataset import DERIVED

C = cfg()  # population prior available from the first released point, so prior dominance is defined
HISTORY_INPUTS = (
    "decay_missing_diff", "spw_decay_diff", "rpw_decay_diff", "log_service_ess_diff", "log_stat_matches_diff",
    "has_history_diff", "service_prior_dominated_diff",
)  # fmt: skip


def derived(table: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for n in HISTORY_INPUTS:
        d = DERIVED[n]
        e = d.build(1e-6)
        exprs.append((e.fill_null(0.0) if d.fill_zero else e).cast(pl.Float64).alias(n))
    return table.select("match_id", "prediction_batch", *exprs).sort("match_id")


def inputs(rows) -> pl.DataFrame:
    f = frame(rows)
    return derived(assemble(build_snapshots(f, observations(f), C).features, C))


def get(t: pl.DataFrame, mid: str) -> dict:
    return t.filter(pl.col("match_id") == mid).row(0, named=True)


def test_registry_covers_every_activity_and_reliability_input():
    from tennis_data.modeling.config import load_modeling_config

    cfg_ = load_modeling_config()
    used = set(cfg_.experiment("M4A").inputs) | set(cfg_.experiment("M4B").inputs)
    assert used - {"elo_logit", "spw_shrunk_diff", "rpw_shrunk_diff"} == set(HISTORY_INPUTS)


def test_first_match_has_no_history_and_missingness_is_antisymmetric():
    t = inputs([match("first", 0, 100, 200), match("next", 1, 100, 300)])
    first = get(t, "first")
    assert all(first[n] == 0.0 for n in HISTORY_INPUTS)  # neither player has history: every difference is 0
    nxt = get(t, "next")
    assert nxt["decay_missing_diff"] == -1.0  # A (100) now has released history, B (300) has none
    assert nxt["has_history_diff"] == 1.0
    assert nxt["log_stat_matches_diff"] > 0 and nxt["log_service_ess_diff"] > 0


def test_same_batch_matches_never_count():
    """Two matches of player 100 in one batch: neither sees the other, and both equal a no-history snapshot."""
    rows = [match("s1", 0, 100, 200, a_sp=(90, 80)), match("s2", 0, 100, 300, a_sp=(70, 20))]
    t = inputs(rows)
    for mid in ("s1", "s2"):
        assert get(t, mid)["decay_missing_diff"] == 0.0  # both players missing: 1 - 1
        assert get(t, mid)["has_history_diff"] == 0.0
        assert get(t, mid)["log_stat_matches_diff"] == 0.0
    # Removing the same-batch sibling or changing its statistics changes nothing.
    alone = inputs([rows[0]])
    changed = inputs([rows[0], match("s2", 0, 100, 300, a_sp=(120, 119))])
    assert get(alone, "s1") == get(t, "s1") == get(changed, "s1")


def test_two_week_event_is_not_available_one_batch_later():
    rows = [
        match("slam", 0, 100, 200, a_sp=(100, 90), release_weeks=1),
        match("w1", 1, 100, 300),
        match("w2", 2, 100, 400),
    ]
    t = inputs(rows)
    w1, w2 = get(t, "w1"), get(t, "w2")
    assert w1["decay_missing_diff"] == 0.0 and w1["has_history_diff"] == 0.0  # slam not released yet
    assert w2["decay_missing_diff"] == -1.0 and w2["has_history_diff"] == 1.0  # released at close of week 1
    # w2 sees slam (week 0) and w1 (week 1), never itself
    assert w2["log_stat_matches_diff"] == pytest.approx(math.log1p(2))


def test_own_match_statistics_do_not_change_inputs():
    base = [match("h", 0, 100, 200), match("own", 1, 100, 300, a_sp=(70, 50))]
    changed = [match("h", 0, 100, 200), match("own", 1, 100, 300, a_sp=(140, 1), b_sp=(99, 98), eligible=False)]
    assert get(inputs(base), "own") == get(inputs(changed), "own")


def _random_rows(seed, n_weeks=40, per_week=5):
    rng = random.Random(seed)
    rows = []
    for w in range(n_weeks):
        players = rng.sample(range(100, 125), 2 * per_week)
        for j in range(per_week):
            a, b = players[2 * j], players[2 * j + 1]
            sa, sb = rng.randint(30, 110), rng.randint(30, 110)
            rows.append(match(f"m{w:03d}-{j}", w, a, b, a_sp=(sa, rng.randint(0, sa)), b_sp=(sb, rng.randint(0, sb)),
                              release_weeks=1 if w % 6 == 0 else 0, eligible=rng.random() > 0.2))  # fmt: skip
    return rows


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_future_and_same_batch_perturbation(seed):
    """Perturb statistics, eligibility and participation of every match from the cutoff batch on."""
    rows = _random_rows(seed)
    cutoff = SAT0 + timedelta(weeks=20)
    rng = random.Random(seed + 100)
    pert = []
    for r in rows:
        if r["prediction_batch"] >= cutoff:
            r = dict(r)
            r["a_sp_played"] = rng.randint(10, 150)
            r["a_sp_won"] = rng.randint(0, r["a_sp_played"])
            r["stats_eligible"] = rng.random() > 0.5
        pert.append(r)
    # extra future matches for players that also appear before the cutoff
    pert += [match(f"extra{k}", 20 + k, 100 + k, 110 + k, a_sp=(200, 190)) for k in range(5)]
    a, b = inputs(rows), inputs(pert)
    early_a = a.filter(pl.col("prediction_batch") <= cutoff)
    early_b = b.filter(pl.col("prediction_batch") <= cutoff).filter(~pl.col("match_id").str.starts_with("extra"))
    assert early_a.equals(early_b)  # includes the cutoff batch itself (the same-batch case)
    later_a = a.filter(pl.col("prediction_batch") > cutoff)
    later_b = b.filter((pl.col("prediction_batch") > cutoff) & ~pl.col("match_id").str.starts_with("extra"))
    assert not later_a.equals(later_b)  # the perturbation is real and reaches later inputs


def test_outcomes_never_enter_history_inputs():
    rows = _random_rows(21)
    flipped = [{**r, "player_a_won": not r["player_a_won"]} for r in rows]
    assert inputs(rows).equals(inputs(flipped))


def test_truncating_the_future_preserves_inputs():
    rows = _random_rows(22)
    full = inputs(rows)
    for weeks in (7, 19, 33):
        cutoff = SAT0 + timedelta(weeks=weeks)
        trunc = inputs([r for r in rows if r["prediction_batch"] <= cutoff])
        assert full.filter(pl.col("prediction_batch") <= cutoff).equals(trunc)


def test_feature_helper_run_matches_engine():
    rows = _random_rows(23, n_weeks=6)
    assert derived(run(rows, C)).equals(inputs(rows))

"""Metric formulas and bit-reproducible aggregation."""

from __future__ import annotations

import math
import random

import polars as pl
import pytest

from tennis_data.elo.evaluation import calibration, expected_calibration_error, metrics, paired_bootstrap


def _df(n=5000, seed=0):
    rng = random.Random(seed)
    p = [rng.random() for _ in range(n)]
    return pl.DataFrame(
        {"p": p, "q": [min(max(x + rng.uniform(-0.1, 0.1), 0.01), 0.99) for x in p],
         "target_a_wins": [rng.random() < x for x in p], "batch": [i // 50 for i in range(n)],
         "surface": [rng.choice(["Hard", "Clay"]) for _ in range(n)]}
    )  # fmt: skip


def test_metric_values_by_hand():
    df = pl.DataFrame({"p": [0.8, 0.3, 0.5], "target_a_wins": [True, True, False]})
    m = metrics(df, "p").row(0, named=True)
    assert m["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.3) + math.log(0.5)) / 3)
    assert m["brier"] == pytest.approx((0.04 + 0.49 + 0.25) / 3)
    assert m["accuracy"] == pytest.approx((1 + 0 + 0.5) / 3)  # p = 0.5 earns half credit


def test_metrics_are_bitwise_order_independent():
    df = _df()
    base_m, base_s = metrics(df, "p"), metrics(df, "p", ["surface"])
    base_c = calibration(df, "p", 10)
    base_b = paired_bootstrap(df, "q", "p", "batch", 200, 3)
    for seed in range(3):
        sh = df.sample(fraction=1.0, shuffle=True, seed=seed)
        assert metrics(sh, "p").equals(base_m)
        assert metrics(sh, "p", ["surface"]).equals(base_s)
        assert calibration(sh, "p", 10).equals(base_c)
        assert expected_calibration_error(calibration(sh, "p", 10)) == expected_calibration_error(base_c)
        assert paired_bootstrap(sh, "q", "p", "batch", 200, 3) == base_b


def test_paired_bootstrap_identical_models_is_zero():
    df = _df(1000)
    r = paired_bootstrap(df, "p", "p", "batch", 100, 1)
    assert r["log_loss_diff"] == 0 and r["log_loss_diff_ci_low"] == 0 and r["log_loss_diff_ci_high"] == 0


def test_calibration_bins_cover_all_rows():
    cal = calibration(_df(), "p", 10)
    assert cal["n"].sum() == 5000 and cal["bin"].to_list() == list(range(10))

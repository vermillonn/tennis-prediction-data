"""Point-statistic formulas, shrinkage and surface blending (pure functions, known answers)."""

from __future__ import annotations

import math

import pytest

from tennis_data.features.estimates import (
    logit,
    method_of_moments_kappa,
    prior_weight,
    rate,
    shrunk,
    sigmoid,
    surface_estimate,
)


def test_rate_is_none_without_evidence():
    assert rate(0, 0) is None  # never a misleading 0%
    assert rate(30, 50) == 0.6


def test_prior_only_estimate_with_zero_observations():
    assert shrunk(0, 0, 0.62, 200) == pytest.approx(0.62)
    assert prior_weight(0, 200) == 1.0


def test_small_sample_is_pulled_toward_prior():
    # 9 of 10 points won: raw 0.9, shrunk barely moves from 0.62
    est = shrunk(9, 10, 0.62, 200)
    assert est == pytest.approx((9 + 200 * 0.62) / 210)
    assert 0.62 < est < 0.64


def test_large_sample_converges_to_observed_rate():
    est = shrunk(70_000, 100_000, 0.62, 200)
    assert est == pytest.approx(0.7, abs=0.001)
    assert prior_weight(100_000, 200) < 0.002


def test_shrinkage_needs_a_prior():
    assert shrunk(10, 20, None, 200) is None


def test_logit_sigmoid_inverse_and_bounds():
    for p in (1e-9, 0.2, 0.5, 0.62, 0.999999):
        assert sigmoid(logit(p)) == pytest.approx(p, rel=1e-9)
    assert 0 < sigmoid(-800) < 1e-300 or sigmoid(-800) == 0.0
    assert sigmoid(800) == 1.0


def test_surface_estimate_uses_off_surface_evidence_and_population_shift():
    mu_all, mu_grass = 0.62, 0.66
    # No grass points: the estimate is the off-surface rate shifted by the population grass effect.
    off = shrunk(620, 1000, mu_all, 200)
    expected = sigmoid(logit(off) + logit(mu_grass) - logit(mu_all))
    assert surface_estimate(0, 0, 620, 1000, mu_all, mu_grass, 200, 600) == pytest.approx(expected)
    assert expected > off  # grass helps servers in the population, so the base moves up


def test_surface_estimate_does_not_double_count():
    # All of the player's points are on this surface: the off-surface part carries no evidence,
    # so the base is the (shifted) population prior, not the player's own rate.
    est = surface_estimate(700, 1000, 700, 1000, 0.62, 0.62, 200, 600)
    assert est == pytest.approx((700 + 600 * 0.62) / 1600)


def test_surface_estimate_converges_to_surface_rate():
    assert surface_estimate(70_000, 100_000, 80_000, 120_000, 0.62, 0.6, 200, 600) == pytest.approx(0.7, abs=0.003)


def test_surface_estimate_without_priors_is_null():
    assert surface_estimate(10, 20, 30, 50, None, 0.6, 200, 600) is None
    assert surface_estimate(10, 20, 30, 50, 0.6, None, 200, 600) is None


def test_method_of_moments_recovers_known_prior_strength():
    import random

    rng = random.Random(1)
    kappa_true, mu = 200.0, 0.62
    won, played = [], []
    for _ in range(3000):
        p = rng.betavariate(kappa_true * mu, kappa_true * (1 - mu))
        n = 2000
        won.append(sum(rng.random() < p for _ in range(n)))
        played.append(n)
    est = method_of_moments_kappa(won, played)
    assert est["mu"] == pytest.approx(mu, abs=0.005)
    assert est["kappa"] == pytest.approx(kappa_true, rel=0.15)
    assert math.isfinite(est["tau"])

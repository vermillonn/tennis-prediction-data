"""Opponent-adjustment model on synthetic data with known serve/return strengths."""

from __future__ import annotations

import numpy as np
import pytest

from tennis_data.features.opponent import fit

SURF = ("Hard", "Clay", "Grass", "Carpet")


def _simulate(alpha, beta, matches_per_pair=6, points=80, seed=0, mu=0.5):
    rng = np.random.default_rng(seed)
    P = len(alpha)
    rows = []
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            for _ in range(matches_per_pair):
                p = 1 / (1 + np.exp(-(mu + alpha[i] - beta[j])))
                rows.append((i, j, 0, points, rng.binomial(points, p)))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(float), a[:, 4].astype(float)


def _fit(server, returner, surface, points, won, P, k=50.0, ks=500.0, weight=None):
    w = np.ones(len(points)) if weight is None else weight
    return fit(server, returner, surface, points, won, w, P, SURF, k, k, ks, 200, 1e-9, 1.0)


def test_recovers_known_serve_and_return_ordering():
    alpha = np.array([0.4, 0.2, 0.0, -0.2, -0.4])
    beta = np.array([-0.3, 0.3, 0.0, 0.1, -0.1])
    f = _fit(*_simulate(alpha, beta, matches_per_pair=40), P=5)
    assert f.converged
    assert list(np.argsort(-f.alpha)) == list(np.argsort(-alpha))
    assert list(np.argsort(-f.beta)) == list(np.argsort(-beta))
    assert np.corrcoef(f.alpha, alpha)[0, 1] > 0.98


def test_stronger_returner_reduces_expected_serving_success():
    alpha = np.zeros(4)
    beta = np.array([0.6, 0.0, 0.0, -0.6])  # player 0 returns best, player 3 worst
    f = _fit(*_simulate(alpha, beta, seed=1), P=4)

    def spw(i, j):
        return 1 / (1 + np.exp(-(f.mu[0] + f.alpha[i] - f.beta[j])))

    assert spw(1, 0) < spw(1, 2) < spw(1, 3)
    # ...and a stronger server reduces the returner's return success symmetrically.
    alpha2 = np.array([0.6, 0.0, 0.0, -0.6])
    g = _fit(*_simulate(alpha2, np.zeros(4), seed=2), P=4)
    rpw = lambda j, i: 1 - 1 / (1 + np.exp(-(g.mu[0] + g.alpha[i] - g.beta[j])))  # noqa: E731
    assert rpw(1, 0) < rpw(1, 2) < rpw(1, 3)


def test_raw_rate_vs_adjusted_when_schedules_differ():
    """Two equal servers; one only faces strong returners. Raw SPW differs, adjusted strength does not."""
    rng = np.random.default_rng(3)
    beta = {2: 0.8, 3: -0.8, 4: 0.0}
    rows = []
    for _ in range(400):
        for server, opp in ((0, 2), (1, 3), (0, 4), (1, 4), (4, 2), (4, 3)):
            p = 1 / (1 + np.exp(-(0.5 + 0.0 - beta[opp])))
            rows.append((server, opp, 0, 60, rng.binomial(60, p)))
        for server, opp in ((2, 4), (3, 4), (2, 0), (3, 1)):  # give returners serve data too
            rows.append((server, opp, 0, 60, rng.binomial(60, 1 / (1 + np.exp(-0.5)))))
    a = np.array(rows)
    f = _fit(a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(float), a[:, 4].astype(float), P=5, k=20)
    raw = [a[a[:, 0] == s, 4].sum() / a[a[:, 0] == s, 3].sum() for s in (0, 1)]
    assert raw[1] - raw[0] > 0.1  # schedule makes player 1 look far better on raw SPW
    assert abs(f.alpha[0] - f.alpha[1]) < 0.05  # the model attributes it to the opponents


def test_sparse_players_are_regularised_and_unseen_players_fall_back():
    alpha = np.array([0.3, -0.3, 0.0])
    s, r, sf, n, k = _simulate(alpha, np.zeros(3), seed=4)
    # add one tiny observation for player 3 (5 of 5 points won) and include an unseen player 4
    s, r = np.append(s, 3), np.append(r, 0)
    sf, n, k = np.append(sf, 0), np.append(n, 5.0), np.append(k, 5.0)
    f = _fit(s, r, sf, n, k, P=5, k=300)
    assert 0 < f.alpha[3] < 0.1  # perfect 5/5 does not produce an extreme strength
    assert f.alpha[4] == 0.0 and f.beta[4] == 0.0  # unseen player: exactly the population baseline
    assert f.serve_weight[4] == 0 and f.return_weight[4] == 0


def test_estimates_finite_and_bounded_on_extreme_data():
    # One server wins every point; ridge keeps everything finite and probabilities inside (0, 1).
    s = np.array([0, 0, 1, 2])
    r = np.array([1, 2, 0, 0])
    sf = np.zeros(4, dtype=int)
    n = np.array([100.0, 100.0, 100.0, 100.0])
    k = np.array([100.0, 100.0, 0.0, 30.0])
    f = _fit(s, r, sf, n, k, P=3, k=100)
    eta = f.mu[0] + f.alpha[:, None] - f.beta[None, :]
    p = 1 / (1 + np.exp(-eta))
    assert np.isfinite(f.alpha).all() and np.isfinite(f.beta).all()
    assert ((p > 0) & (p < 1)).all()


def test_surface_deviation_is_shrunk_more():
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(300):
        for i, j in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
            for s, bonus in ((0, 0.0), (2, 0.5 if i == 0 else 0.0)):  # player 0 much better on grass
                p = 1 / (1 + np.exp(-(0.5 + bonus)))
                rows.append((i, j, s, 50, rng.binomial(50, p)))
    a = np.array(rows)
    f = _fit(a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(float), a[:, 4].astype(float), P=3, k=50, ks=500)
    assert f.a_surf[0, 2] > 0.1  # grass bonus detected
    assert abs(f.a_surf[1, 2]) < 0.1 and abs(f.a_surf[2, 2]) < 0.1


def test_fit_is_deterministic():
    data = _simulate(np.array([0.2, 0.0, -0.2]), np.array([0.1, 0.0, -0.1]), seed=6)
    a, b = _fit(*data, P=3), _fit(*data, P=3)
    assert np.array_equal(a.alpha, b.alpha) and np.array_equal(a.beta, b.beta) and np.array_equal(a.mu, b.mu)


def test_warm_start_reaches_same_solution():
    data = _simulate(np.array([0.2, 0.0, -0.2]), np.array([0.1, 0.0, -0.1]), seed=7)
    cold = _fit(*data, P=3)
    w = np.ones(len(data[3]))
    warm = fit(*data, w, 3, SURF, 50.0, 50.0, 500.0, 200, 1e-9, 1.0, init=cold)
    assert warm.iterations <= 2
    assert np.allclose(warm.alpha, cold.alpha, atol=1e-8)


@pytest.mark.parametrize("weight", [0.5, 2.0])
def test_uniform_weights_with_rescaled_kappa_give_same_fit(weight):
    """Scaling all weights by c is equivalent to dividing the ridge strength by c."""
    data = _simulate(np.array([0.3, 0.0, -0.3]), np.zeros(3), seed=8)
    base = _fit(*data, P=3, k=60.0, ks=600.0)
    scaled = _fit(*data, P=3, k=60.0 * weight, ks=600.0 * weight, weight=np.full(len(data[3]), weight))
    assert np.allclose(base.alpha, scaled.alpha, atol=1e-6)

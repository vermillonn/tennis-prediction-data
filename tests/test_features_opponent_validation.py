"""Independent validation of the ridge-binomial opponent estimator against a generic optimiser.

The objective and gradient below are written from the model specification, not imported from the
module, and minimised with scipy's L-BFGS-B. The custom block-Newton fit must reach the same
penalised optimum, the same fitted probabilities, decrease its objective monotonically and stay
stable on extreme or badly initialised problems.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.optimize import minimize  # noqa: E402

from tennis_data.features.opponent import fit, penalized_objective  # noqa: E402

SURF = ("Hard", "Clay", "Grass", "Carpet")
K_SERVE, K_RETURN, K_SURF = 40.0, 60.0, 150.0


def _softplus(x):
    return np.logaddexp(0.0, x)


def _dataset(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    P = {"random": 6, "sparse": 8, "decay": 7}[kind]
    S_used = 2
    alpha, beta = rng.normal(0, 0.3, P), rng.normal(0, 0.3, P)
    n_obs = {"random": 300, "sparse": 25, "decay": 400}[kind]
    rows = []
    for _ in range(n_obs):
        i, j = rng.choice(P, 2, replace=False)
        s = int(rng.integers(S_used))
        n = int(rng.integers(4, 15) if kind == "sparse" else rng.integers(20, 120))
        p = 1 / (1 + np.exp(-(0.5 + 0.2 * s + alpha[i] - beta[j])))
        rows.append((i, j, s, n, rng.binomial(n, p)))
    a = np.array(rows)
    if kind == "sparse":  # perfect and zero records for some observations
        a[0, 4] = a[0, 3]
        a[1, 4] = 0
    w = np.ones(len(a)) if kind != "decay" else np.exp2(-rng.integers(0, 300, len(a)) / 104.0)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(float), a[:, 4].astype(float), w, P, S_used


def _reference(server, returner, surface, n, k, w, P, S):
    """Independent penalised objective + gradient over [mu(S), alpha(P), beta(P), a(P,S), b(P,S)]."""
    pbar = (w * k).sum() / (w * n).sum()
    la, lb, ls = (K_SERVE * pbar * (1 - pbar), K_RETURN * pbar * (1 - pbar), K_SURF * pbar * (1 - pbar))

    def unpack(x):
        mu, al, be = x[:S], x[S : S + P], x[S + P : S + 2 * P]
        a = x[S + 2 * P : S + 2 * P + P * S].reshape(P, S)
        b = x[S + 2 * P + P * S :].reshape(P, S)
        return mu, al, be, a, b

    def f(x):
        mu, al, be, a, b = unpack(x)
        eta = mu[surface] + al[server] + a[server, surface] - be[returner] - b[returner, surface]
        F = (w * (k * _softplus(-eta) + (n - k) * _softplus(eta))).sum()
        F += 0.5 * (la * al @ al + lb * be @ be + ls * ((a * a).sum() + (b * b).sum()))
        g_eta = w * (n / (1 + np.exp(-eta)) - k)
        g_mu = np.bincount(surface, g_eta, S)
        g_al = np.bincount(server, g_eta, P) + la * al
        g_be = -np.bincount(returner, g_eta, P) + lb * be
        g_a = np.zeros((P, S))
        np.add.at(g_a, (server, surface), g_eta)
        g_b = np.zeros((P, S))
        np.add.at(g_b, (returner, surface), -g_eta)
        g_a += ls * a
        g_b += ls * b
        return F, np.concatenate([g_mu, g_al, g_be, g_a.ravel(), g_b.ravel()])

    return f, unpack, (la, lb, ls)


def _custom(server, returner, surface, n, k, w, P, init=None, tol=1e-10):
    return fit(server, returner, surface, n, k, w, P, SURF, K_SERVE, K_RETURN, K_SURF, 5000, tol, 1.0,
               init=init, record_trace=True)  # fmt: skip


def _probs_custom(f, S, P):
    return 1 / (1 + np.exp(-(f.mu[None, None, :S] + f.alpha[:, None, None] + f.a_surf[:, None, :S]
                             - f.beta[None, :, None] - f.b_surf[None, :, :S])))  # fmt: skip


def _probs_ref(params, P):
    mu, al, be, a, b = params
    return 1 / (
        1 + np.exp(-(mu[None, None, :] + al[:, None, None] + a[:, None, :] - be[None, :, None] - b[None, :, :]))
    )


@pytest.mark.parametrize("kind,seed", [("random", 0), ("random", 1), ("sparse", 2), ("decay", 3)])
def test_matches_generic_optimizer(kind, seed):
    server, returner, surface, n, k, w, P, S = _dataset(kind, seed)
    custom = _custom(server, returner, surface, n, k, w, P)
    assert custom.converged

    f_ref, unpack, lams = _reference(server, returner, surface, n, k, w, P, S)
    x0 = np.zeros(S + 2 * P + 2 * P * S)
    res = minimize(f_ref, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 50_000, "maxcor": 50, "gtol": 1e-12, "ftol": 1e-16})  # fmt: skip
    F_ref = res.fun

    # The custom solution evaluated with the independent objective (only the used surfaces carry data;
    # the unused surface deviations are exactly zero and add nothing).
    assert np.all(custom.a_surf[:, S:] == 0) and np.all(custom.b_surf[:, S:] == 0)
    x_custom = np.concatenate([custom.mu[:S], custom.alpha, custom.beta, custom.a_surf[:, :S].ravel(),
                               custom.b_surf[:, :S].ravel()])  # fmt: skip
    F_custom = f_ref(x_custom)[0]
    scale = max(1.0, abs(F_ref))
    assert F_custom <= F_ref + 1e-9 * scale, (F_custom, F_ref)
    # Observed agreement is ~1e-12 relative; a fit stopped after 2 sweeps is off by ~3e-7 relative or more.
    assert abs(F_custom - F_ref) <= 1e-10 * scale
    # The module's own objective agrees with the independent definition.
    assert custom.objective == pytest.approx(F_custom, rel=1e-12, abs=1e-9)

    p_custom = _probs_custom(custom, S, P)
    p_ref = _probs_ref(unpack(res.x), P)
    assert np.max(np.abs(p_custom - p_ref)) < 1e-7  # observed ~1e-8
    # Gradient of the independent objective vanishes at the custom solution.
    assert np.max(np.abs(f_ref(x_custom)[1])) < 1e-5 * scale


def test_validation_detects_an_unconverged_fit():
    """The tolerances above are tight enough to reject a solution stopped early."""
    server, returner, surface, n, k, w, P, S = _dataset("random", 0)
    early = fit(server, returner, surface, n, k, w, P, SURF, K_SERVE, K_RETURN, K_SURF, 2, 1e-10, 1.0)
    f_ref, _, _ = _reference(server, returner, surface, n, k, w, P, S)
    res = minimize(f_ref, np.zeros(S + 2 * P + 2 * P * S), jac=True, method="L-BFGS-B",
                   options={"maxiter": 50_000, "maxcor": 50, "gtol": 1e-12, "ftol": 1e-16})  # fmt: skip
    x = np.concatenate(
        [early.mu[:S], early.alpha, early.beta, early.a_surf[:, :S].ravel(), early.b_surf[:, :S].ravel()]
    )
    assert not early.converged
    assert f_ref(x)[0] - res.fun > 1e-10 * abs(res.fun)


@pytest.mark.parametrize("kind,seed", [("random", 4), ("sparse", 5), ("decay", 6)])
def test_objective_is_monotone(kind, seed):
    server, returner, surface, n, k, w, P, _ = _dataset(kind, seed)
    f = _custom(server, returner, surface, n, k, w, P)
    t = np.array(f.trace)
    assert len(t) >= 2
    assert np.all(np.diff(t) <= 1e-12 * np.maximum(1.0, np.abs(t[:-1])))


def test_stable_from_bad_initialisation():
    server, returner, surface, n, k, w, P, _ = _dataset("random", 7)
    cold = _custom(server, returner, surface, n, k, w, P)
    rng = np.random.default_rng(0)
    bad = _custom(server, returner, surface, n, k, w, P)
    bad.mu, bad.alpha, bad.beta = rng.normal(0, 3, bad.mu.shape), rng.normal(0, 3, P), rng.normal(0, 3, P)
    bad.a_surf, bad.b_surf = rng.normal(0, 3, bad.a_surf.shape), rng.normal(0, 3, bad.b_surf.shape)
    warm = _custom(server, returner, surface, n, k, w, P, init=bad)
    assert warm.converged
    assert np.all(np.diff(warm.trace) <= 1e-12 * np.maximum(1.0, np.abs(np.array(warm.trace[:-1]))))
    assert warm.objective == pytest.approx(cold.objective, rel=1e-9)
    assert np.allclose(warm.alpha, cold.alpha, atol=1e-6) and np.allclose(warm.beta, cold.beta, atol=1e-6)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e4])
def test_stable_across_data_scales(scale):
    server, returner, surface, n, k, w, P, _ = _dataset("random", 8)
    f = fit(server, returner, surface, n * scale, k * scale, w, P, SURF, K_SERVE, K_RETURN, K_SURF,
            5000, 1e-10, 1.0, record_trace=True)  # fmt: skip
    assert f.converged
    for arr in (f.mu, f.alpha, f.beta, f.a_surf, f.b_surf):
        assert np.isfinite(arr).all()
    eta = f.mu[surface] + f.alpha[server] + f.a_surf[server, surface] - f.beta[returner] - f.b_surf[returner, surface]
    p = 1 / (1 + np.exp(-eta))
    assert ((p > 0) & (p < 1)).all()


def test_extreme_records_stay_finite_and_monotone():
    # Player 0 wins every point it serves; player 1 never wins one.
    server = np.array([0, 0, 1, 1, 2, 2])
    returner = np.array([1, 2, 0, 2, 0, 1])
    surface = np.zeros(6, dtype=np.int64)
    n = np.full(6, 80.0)
    k = np.array([80.0, 80.0, 0.0, 0.0, 40.0, 45.0])
    f = fit(server, returner, surface, n, k, np.ones(6), 3, SURF, K_SERVE, K_RETURN, K_SURF, 5000, 1e-10, 1.0,
            record_trace=True)  # fmt: skip
    assert f.converged and np.isfinite(f.objective)
    assert np.all(np.diff(f.trace) <= 1e-12 * np.maximum(1.0, np.abs(np.array(f.trace[:-1]))))
    assert abs(f.alpha).max() < 5 and abs(f.beta).max() < 5  # ridge keeps strengths bounded


def test_penalized_objective_matches_hand_calculation():
    eta = np.array([0.0, np.log(3.0)])
    wk, wn = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    params = (np.array([0.5]), np.array([-1.0]), np.zeros(1), np.array([2.0]))
    F = penalized_objective(eta, wk, wn, params, (2.0, 4.0, 1.0))
    nll = 1 * np.log(2) + 1 * np.log(2) + 2 * np.log(1 + 1 / 3) + 2 * np.log(1 + 3)
    pen = 0.5 * (2.0 * 0.25 + 4.0 * 1.0 + 1.0 * 4.0)
    assert F == pytest.approx(nll + pen)

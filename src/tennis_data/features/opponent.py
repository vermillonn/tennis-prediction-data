"""Opponent-adjusted serve/return strengths: a ridge-regularised weighted binomial model.

For a server i facing returner j on surface s, each observation is ``won`` of ``points`` with

    logit P(server wins a point) = mu_s + alpha_i + a_is - beta_j - b_js

- mu_s: surface baseline (unpenalised; identified by the ridge terms being centred at 0).
- alpha_i / beta_j: overall serving / returning strength, ridge penalties lambda_serve / lambda_return.
- a_is / b_js: surface deviations, stronger ridge penalty lambda_s (sparse data shrink to 0).
- lambda = kappa * pbar * (1 - pbar): kappa is the penalty in equivalent points, since one point
  carries Fisher information of about pbar * (1 - pbar) on the logit scale.
- Observations are weighted by 2 ** (-age / half_life) relative to the fit's batch.

Identifiability convention: the ridge (zero-mean Gaussian) prior fixes the location of alpha and beta;
an unseen player has alpha = beta = 0, i.e. exactly the population baseline for that surface.

Flat directions: the likelihood is exactly invariant to (alpha_i + c for every active server,
mu_s - c), to (beta_j + c for every active returner, mu_s + c), to the per-surface analogues for
a_is / b_js, and, per player, to (alpha_i + c, a_is - c on every surface the player served on) and
the same for beta_j / b_js. Only the ridge pins these down, so plain coordinate descent creeps along them. After every
sweep the optimal shift along each direction is applied in closed form (c = -mean of the active
parameters), which only lowers the penalised objective. A consequence: at the optimum the active
alpha, beta and per-surface deviations each average exactly zero.

Fitting: block coordinate Newton. Within each block (mu, alpha, beta, a, b) every observation touches
exactly one parameter, so the block Hessian is diagonal and one exact Newton step per block is cheap
(numpy.bincount). Steps are clipped to +-max_step. Iterate to max|step| < tolerance or max_iterations.
Arithmetic is deterministic for a fixed observation order; callers pass canonically sorted arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OpponentFit:
    fit_batch: object
    surfaces: tuple[str, ...]
    mu: np.ndarray  # (S,)
    alpha: np.ndarray  # (P,)
    beta: np.ndarray  # (P,)
    a_surf: np.ndarray  # (P, S)
    b_surf: np.ndarray  # (P, S)
    serve_weight: np.ndarray  # (P,) weighted points as server
    return_weight: np.ndarray  # (P,) weighted points as returner
    surface_weight: np.ndarray  # (S,) weighted points per surface
    iterations: int
    converged: bool
    max_step: float
    seconds: float
    n_obs: int
    stats: dict = field(default_factory=dict)

    @property
    def mu_overall(self) -> float:
        w = self.surface_weight
        return float((self.mu * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _expit(x: np.ndarray) -> np.ndarray:
    """Logistic function; the tanh form is overflow-free for any finite input."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def fit(
    server: np.ndarray,
    returner: np.ndarray,
    surface: np.ndarray,
    points: np.ndarray,
    won: np.ndarray,
    weight: np.ndarray,
    n_players: int,
    surfaces: tuple[str, ...],
    kappa_serve: float,
    kappa_return: float,
    kappa_surface: float,
    max_iterations: int,
    tolerance: float,
    max_step: float,
    fit_batch=None,
    init: OpponentFit | None = None,
) -> OpponentFit:
    t0 = time.perf_counter()
    S, P = len(surfaces), n_players
    w = weight.astype(np.float64)
    n = points.astype(np.float64)
    k = won.astype(np.float64)
    wn, wk = w * n, w * k
    pbar = wk.sum() / wn.sum()
    lam_a = kappa_serve * pbar * (1 - pbar)
    lam_b = kappa_return * pbar * (1 - pbar)
    lam_s = kappa_surface * pbar * (1 - pbar)
    i_s = server * S + surface
    j_s = returner * S + surface

    surf_w = np.bincount(surface, wn, S)
    if init is not None:
        mu, alpha, beta = init.mu.copy(), init.alpha.copy(), init.beta.copy()
        a_surf, b_surf = init.a_surf.ravel().copy(), init.b_surf.ravel().copy()
    else:
        rates = np.bincount(surface, wk, S) / np.where(surf_w > 0, surf_w, 1.0)
        rates = np.where(surf_w > 0, np.clip(rates, 1e-6, 1 - 1e-6), pbar)
        mu = np.log(rates / (1 - rates))
        alpha, beta = np.zeros(P), np.zeros(P)
        a_surf, b_surf = np.zeros(P * S), np.zeros(P * S)

    act_a = np.bincount(server, minlength=P) > 0
    act_b = np.bincount(returner, minlength=P) > 0
    act_as = np.bincount(i_s, minlength=P * S) > 0
    act_bs = np.bincount(j_s, minlength=P * S) > 0

    def recentre() -> float:
        """Exact minimisation along the likelihood-invariant directions; returns the largest shift."""
        shifts = [0.0]
        if act_a.any():
            c = -alpha[act_a].mean()
            alpha[act_a] += c
            mu[:] -= c
            shifts.append(abs(c))
        if act_b.any():
            c = -beta[act_b].mean()
            beta[act_b] += c
            mu[:] += c
            shifts.append(abs(c))
        a2, b2 = a_surf.reshape(P, S), b_surf.reshape(P, S)
        ma, mb = act_as.reshape(P, S), act_bs.reshape(P, S)
        # Per player: split between the overall strength and its surface deviations,
        # argmin_c lam_p (theta + c)^2 + lam_s sum_{active s} (dev_s - c)^2.
        for theta, dev, mask, lam in ((alpha, a2, ma, lam_a), (beta, b2, mb, lam_b)):
            m = mask.sum(axis=1)
            has = m > 0
            if has.any():
                c = (lam_s * (dev * mask).sum(axis=1) - lam * theta) / (lam + lam_s * m)
                c = np.where(has, c, 0.0)
                theta += c
                dev -= c[:, None] * mask
                shifts.append(float(np.abs(c).max()))
        for s_ in range(S):
            if ma[:, s_].any():
                c = -a2[ma[:, s_], s_].mean()
                a2[ma[:, s_], s_] += c
                mu[s_] -= c
                shifts.append(abs(c))
            if mb[:, s_].any():
                c = -b2[mb[:, s_], s_].mean()
                b2[mb[:, s_], s_] += c
                mu[s_] += c
                shifts.append(abs(c))
        return max(shifts)

    recentre()
    eta = mu[surface] + alpha[server] + a_surf[i_s] - beta[returner] - b_surf[j_s]
    converged, it, biggest = False, 0, float("inf")
    for it in range(1, max_iterations + 1):  # noqa: B007
        biggest = 0.0
        for block in ("mu", "alpha", "beta", "a", "b"):
            p = _expit(eta)
            r = wk - wn * p
            h = wn * p * (1 - p)
            if block == "mu":
                g, H, idx, sign, theta, lam, size = (
                    np.bincount(surface, r, S),
                    np.bincount(surface, h, S),
                    surface,
                    1.0,
                    mu,
                    0.0,
                    S,
                )
            elif block == "alpha":
                g, H, idx, sign, theta, lam, size = (
                    np.bincount(server, r, P),
                    np.bincount(server, h, P),
                    server,
                    1.0,
                    alpha,
                    lam_a,
                    P,
                )
            elif block == "beta":
                g, H, idx, sign, theta, lam, size = (
                    -np.bincount(returner, r, P),
                    np.bincount(returner, h, P),
                    returner,
                    -1.0,
                    beta,
                    lam_b,
                    P,
                )
            elif block == "a":
                g, H, idx, sign, theta, lam, size = (
                    np.bincount(i_s, r, P * S),
                    np.bincount(i_s, h, P * S),
                    i_s,
                    1.0,
                    a_surf,
                    lam_s,
                    P * S,
                )
            else:
                g, H, idx, sign, theta, lam, size = (
                    -np.bincount(j_s, r, P * S),
                    np.bincount(j_s, h, P * S),
                    j_s,
                    -1.0,
                    b_surf,
                    lam_s,
                    P * S,
                )
            grad = g - lam * theta
            hess = H + lam
            step = np.divide(grad, hess, out=np.zeros(size), where=hess > 0)
            step = np.clip(step, -max_step, max_step)
            theta += step
            eta += sign * step[idx]
            biggest = max(biggest, float(np.abs(step).max()) if size else 0.0)
        shift = recentre()
        if shift > 0:
            eta = mu[surface] + alpha[server] + a_surf[i_s] - beta[returner] - b_surf[j_s]
        biggest = max(biggest, shift)
        if biggest < tolerance:
            converged = True
            break
    return OpponentFit(
        fit_batch=fit_batch,
        surfaces=surfaces,
        mu=mu,
        alpha=alpha,
        beta=beta,
        a_surf=a_surf.reshape(P, S),
        b_surf=b_surf.reshape(P, S),
        serve_weight=np.bincount(server, wn, P),
        return_weight=np.bincount(returner, wn, P),
        surface_weight=surf_w,
        iterations=it,
        converged=converged,
        max_step=biggest,
        seconds=time.perf_counter() - t0,
        n_obs=len(points),
        stats={
            "pbar": float(pbar),
            "lambda_serve": float(lam_a),
            "lambda_return": float(lam_b),
            "lambda_surface": float(lam_s),
        },
    )

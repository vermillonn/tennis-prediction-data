"""L2-penalised logistic regression without intercept, by damped Newton's method.

Objective (sklearn convention): ``sum_i logloss_i(beta) + |beta|^2 / (2 C)``. With antisymmetric inputs and no
intercept, ``P(A beats B) = sigmoid(x beta)`` and ``P(B beats A) = sigmoid(-x beta) = 1 - P(A beats B)`` exactly,
so player-order symmetry holds by construction. The objective is strictly convex, so the minimiser is unique
and the result does not depend on the row order beyond floating-point rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticFit:
    coef: np.ndarray  # on the scaled inputs
    c: float
    iterations: int
    converged: bool
    objective: float
    grad_norm: float


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def objective(beta: np.ndarray, x: np.ndarray, y: np.ndarray, c: float) -> float:
    z = x @ beta
    # logloss = softplus(z) - y z
    return float(np.sum(_softplus(z) - y * z) + beta @ beta / (2 * c))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def fit_logistic(x: np.ndarray, y: np.ndarray, c: float, max_iter: int = 100, tol: float = 1e-10) -> LogisticFit:
    n, k = x.shape
    if c <= 0:
        raise ValueError("C must be positive")
    y = y.astype(np.float64)
    beta = np.zeros(k)
    f = objective(beta, x, y, c)
    it, converged, gnorm = 0, False, float("inf")
    for it in range(1, max_iter + 1):
        p = sigmoid(x @ beta)
        g = x.T @ (p - y) + beta / c
        gnorm = float(np.max(np.abs(g)))
        if gnorm <= tol * max(1.0, n):
            converged = True
            it -= 1
            break
        h = (x * (p * (1 - p))[:, None]).T @ x + np.eye(k) / c
        step = np.linalg.solve(h, g)
        t = 1.0
        while True:  # backtracking (Armijo); Newton steps are almost always accepted at t = 1
            cand = beta - t * step
            fc = objective(cand, x, y, c)
            if fc <= f - 1e-4 * t * float(g @ step) or t < 1e-10:
                break
            t *= 0.5
        if f - fc <= 1e-15 * max(1.0, abs(f)):
            beta, f = cand, fc
            converged = True
            break
        beta, f = cand, fc
    return LogisticFit(beta, c, it, converged, f, gnorm)


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ beta)

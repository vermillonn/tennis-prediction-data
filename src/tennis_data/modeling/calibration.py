"""Symmetric post-hoc calibration, fitted only on chronologically earlier out-of-fold predictions.

Both methods satisfy ``g(1 - p) = 1 - g(p)``, so calibration cannot break player-order symmetry:

- ``temperature``: ``sigmoid(b * logit(p))`` (Platt scaling without an intercept; an intercept would favour
  whichever player happens to be labelled A).
- ``isotonic``: pool-adjacent-violators on the symmetrised sample ``{(p, y)} U {(1 - p, 1 - y)}``, then linear
  interpolation between the fitted knots. The isotonic fit of a symmetric sample is itself symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tennis_data.modeling.logistic import sigmoid


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    q = np.clip(p, eps, 1 - eps)
    return np.log(q / (1 - q))


@dataclass(frozen=True)
class Calibrator:
    method: str
    eps: float
    rows: int = 0
    slope: float = 1.0
    knots_x: tuple[float, ...] = field(default=())
    knots_y: tuple[float, ...] = field(default=())

    def apply(self, p: np.ndarray) -> np.ndarray:
        if self.method == "raw":
            return p
        if self.method == "temperature":
            return sigmoid(self.slope * logit(p, self.eps))
        if self.method == "isotonic":
            out = np.interp(p, np.asarray(self.knots_x), np.asarray(self.knots_y))
            return np.clip(out, self.eps, 1 - self.eps)
        raise ValueError(self.method)

    def summary(self) -> dict:
        d = {"method": self.method, "rows": self.rows}
        if self.method == "temperature":
            d["slope"] = self.slope
        if self.method == "isotonic":
            d["knots"] = len(self.knots_x)
        return d


def fit_temperature(p: np.ndarray, y: np.ndarray, eps: float, max_iter: int = 100) -> Calibrator:
    z, y = logit(p, eps), y.astype(np.float64)
    b = 1.0
    for _ in range(max_iter):
        q = sigmoid(b * z)
        g = float(np.sum((q - y) * z))
        h = float(np.sum(q * (1 - q) * z * z))
        if h <= 0:
            break
        step = g / h
        b -= step
        if abs(step) < 1e-12:
            break
    return Calibrator("temperature", eps, len(p), slope=b)


def pava(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-decreasing least-squares fit; ties in x are pooled first. Returns (unique x, fitted values)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(np.float64)
    ux, start = np.unique(xs, return_index=True)
    sums = np.add.reduceat(ys, start)
    cnts = np.diff(np.append(start, len(xs))).astype(np.float64)
    vals, wts, sizes = [], [], []
    for s, w in zip(sums / cnts, cnts, strict=True):
        vals.append(s)
        wts.append(w)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            n2 = sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [v2], [w2], [n2]
    fitted = np.repeat(np.asarray(vals), np.asarray(sizes))
    return ux, fitted


def _compress(x: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first and last knot of every constant block: the interpolant is unchanged."""
    if len(x) <= 2:
        return x, f
    keep = np.ones(len(x), dtype=bool)
    keep[1:-1] = (f[1:-1] != f[:-2]) | (f[1:-1] != f[2:])
    return x[keep], f[keep]


def fit_isotonic(p: np.ndarray, y: np.ndarray, eps: float) -> Calibrator:
    y = y.astype(np.float64)
    xs = np.concatenate([p, 1 - p])
    ys = np.concatenate([y, 1 - y])
    ux, fy = _compress(*pava(xs, ys))
    return Calibrator("isotonic", eps, len(p), knots_x=tuple(ux.tolist()), knots_y=tuple(fy.tolist()))


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray, eps: float) -> Calibrator:
    if method == "raw":
        return Calibrator("raw", eps, len(p))
    if method == "temperature":
        return fit_temperature(p, y, eps)
    if method == "isotonic":
        return fit_isotonic(p, y, eps)
    raise ValueError(f"unknown calibration method {method}")

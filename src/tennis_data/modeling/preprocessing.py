"""Training-fold-only preprocessing that preserves antisymmetry.

Inputs are A-minus-B differences, so ``x(B, A) = -x(A, B)``. Centring would break this (the mean of an
antisymmetric feature over one orientation is not zero), so each column is only *scaled* by its root mean
square on the training rows. Missing differences were already filled with 0 (the antisymmetric value) and
are accompanied by missingness-indicator differences where they carry information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scaler:
    columns: tuple[str, ...]
    scale: tuple[float, ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x / np.asarray(self.scale)

    def to_dict(self) -> dict:
        return dict(zip(self.columns, self.scale, strict=True))


def fit_scaler(x_train: np.ndarray, columns) -> Scaler:
    if not np.isfinite(x_train).all():
        raise ValueError("non-finite model inputs in the training fold")
    rms = np.sqrt(np.mean(x_train**2, axis=0)) if len(x_train) else np.ones(x_train.shape[1])
    rms = np.where(rms > 0, rms, 1.0)
    return Scaler(tuple(columns), tuple(float(v) for v in rms))

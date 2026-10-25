"""Chronological walk-forward splits.

The model for calendar year ``y`` (rows whose ``prediction_batch`` falls in ``y``) is trained on rows with
``prediction_batch >= train_start`` whose label was *released* before the first prediction batch of ``y``:
``release_batch < cutoff(y)``. A label is known only after its batch closes; two-week events release one week
later (Phase 2 chronology). The tournament start date is never used as evidence that a result was known.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from tennis_data.modeling.config import ModelingConfig


@dataclass(frozen=True)
class YearSplit:
    year: int
    cutoff: date  # first prediction batch of `year`; training labels released strictly before it
    train_rows: int
    predict_rows: int
    last_train_release: date | None  # latest release batch among training rows
    first_train_batch: date | None
    last_train_batch: date | None


def year_col() -> pl.Expr:
    return pl.col("prediction_batch").dt.year()


def cutoffs(ds: pl.DataFrame) -> dict[int, date]:
    g = ds.group_by(year_col().alias("y")).agg(pl.col("prediction_batch").min().alias("c")).sort("y")
    return dict(zip(g["y"].to_list(), g["c"].to_list(), strict=True))


def train_mask(ds: pl.DataFrame, cfg: ModelingConfig, cutoff: date) -> pl.Series:
    return (ds["prediction_batch"] >= cfg.train_start) & (ds["release_batch"] < cutoff)


def predict_mask(ds: pl.DataFrame, year: int) -> pl.Series:
    return ds.select(year_col() == year).to_series()


def split(ds: pl.DataFrame, cfg: ModelingConfig, year: int, cutoff: date) -> YearSplit:
    tr = ds.filter(train_mask(ds, cfg, cutoff))
    return YearSplit(
        year=year,
        cutoff=cutoff,
        train_rows=tr.height,
        predict_rows=int(predict_mask(ds, year).sum()),
        last_train_release=tr["release_batch"].max() if tr.height else None,
        first_train_batch=tr["prediction_batch"].min() if tr.height else None,
        last_train_batch=tr["prediction_batch"].max() if tr.height else None,
    )


def calibration_mask(oof: pl.DataFrame, cfg: ModelingConfig, year: int, cutoff: date) -> pl.Series:
    """Out-of-fold rows usable to calibrate year ``year``: years y-window..y-1 AND label released before cutoff.

    The release filter matters for multi-week events straddling New Year: their labels are not known at
    the first batch of ``year`` even though they were predicted in ``year - 1``.
    """
    yrs = oof.select(year_col()).to_series()
    return (yrs >= year - cfg.calibration_window) & (yrs < year) & (oof["release_batch"] < cutoff)


def fold_of(year: int, cfg: ModelingConfig) -> str | None:
    for a, b in cfg.folds:
        if a <= year <= b:
            return f"{a}-{b}"
    return None

"""Prediction and model artifacts (Parquet + JSON) with checksums."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from tennis_data.storage.parquet import write_single

PREDICTION_COLUMNS = (
    "match_id", "prediction_batch", "player_a_id", "player_b_id", "surface", "model_id", "model_version",
    "training_cutoff", "calibration_cutoff", "predicted_probability_a", "baseline_elo_probability_a",
    "target_a_wins", "configuration_hash",
)  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default(o):
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n"


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps(obj))
    tmp.replace(path)
    return path


def prediction_frame(ds: pl.DataFrame, rows: np.ndarray, cols: dict[str, np.ndarray], meta: dict) -> pl.DataFrame:
    base = ds[rows].select(
        "match_id", "prediction_batch", "release_batch", "player_a_id", "player_b_id", "surface",
        "tournament_level", "baseline_elo_probability_a", "target_a_wins",
    )  # fmt: skip
    return base.with_columns(
        *[pl.lit(v).alias(k) for k, v in meta.items()], *[pl.Series(k, v) for k, v in cols.items()]
    )


def write_predictions(df: pl.DataFrame, path: Path) -> Path:
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prediction artifact lacks required columns {missing}")
    lead = list(PREDICTION_COLUMNS)
    df = df.select(lead + [c for c in df.columns if c not in lead]).sort("model_id", "prediction_batch", "match_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_single(df, path)

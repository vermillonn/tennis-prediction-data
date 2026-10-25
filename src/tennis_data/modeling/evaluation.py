"""Metrics, subgroups and the paired block bootstrap for Phase 3B.

Point metrics reuse the Phase 2 scorer (``elo.evaluation``: exactly rounded means), so M0 numbers here equal
the Phase 2 numbers on the same rows. The bootstrap is new: it supports any cluster column and any alpha.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from tennis_data.elo.evaluation import calibration, expected_calibration_error, metrics
from tennis_data.modeling.config import ModelingConfig

__all__ = ["calibration", "expected_calibration_error", "metrics", "row_losses", "paired_bootstrap", "subgroups"]

EPS = 1e-15


def row_losses(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(p, EPS, 1 - EPS)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    return ll, (p - y) ** 2


def summarize(p: np.ndarray, y: np.ndarray, bins: int) -> dict:
    df = pl.DataFrame({"p": p, "target_a_wins": y.astype(bool)})
    m = metrics(df, "p").row(0, named=True)
    m["ece"] = expected_calibration_error(calibration(df, "p", bins))
    return m


def paired_bootstrap(
    p_new: np.ndarray,
    p_base: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    samples: int,
    seed: int,
    alphas: tuple[float, ...] = (0.05,),
) -> dict:
    """Delta = loss(new) - loss(base) on identical rows; negative means the candidate is better.

    Whole clusters (default: weekly prediction batches) are resampled with replacement, because predictions
    within a cluster share a frozen rating state, players and conditions. Per-cluster sums use ``math.fsum``
    so the point estimate does not depend on row order.
    """
    y = y.astype(np.float64)
    ll_n, br_n = row_losses(p_new, y)
    ll_b, br_b = row_losses(p_base, y)
    uniq, inv = np.unique(clusters, return_inverse=True)
    k = len(uniq)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(k + 1))
    dll, dbr = ll_n - ll_b, br_n - br_b
    s_ll = np.array([math.fsum(dll[order[bounds[i] : bounds[i + 1]]]) for i in range(k)])
    s_br = np.array([math.fsum(dbr[order[bounds[i] : bounds[i + 1]]]) for i in range(k)])
    n = np.diff(bounds).astype(np.float64)
    total = float(n.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(samples, k))
    m = n[idx].sum(axis=1)
    draws_ll = np.sort(s_ll[idx].sum(axis=1) / m)
    draws_br = np.sort(s_br[idx].sum(axis=1) / m)
    out = {
        "rows": int(total),
        "clusters": int(k),
        "samples": samples,
        "seed": seed,
        "log_loss_diff": math.fsum(s_ll) / total,
        "brier_diff": math.fsum(s_br) / total,
        "share_ll_below_zero": float(np.mean(draws_ll < 0)),
    }
    for a in alphas:
        tag = f"{100 * (1 - a):g}"
        out[f"ll_ci{tag}"] = (float(np.quantile(draws_ll, a / 2)), float(np.quantile(draws_ll, 1 - a / 2)))
        out[f"brier_ci{tag}"] = (float(np.quantile(draws_br, a / 2)), float(np.quantile(draws_br, 1 - a / 2)))
    return out


def verdict(low: float, high: float) -> str:
    if high < 0:
        return "candidate better"
    if low > 0:
        return "candidate worse"
    return "inconclusive (interval includes 0)"


def add_groups(df: pl.DataFrame, cfg: ModelingConfig) -> pl.DataFrame:
    e = cfg.sample_size_edges
    pts = pl.col("min_prior_service_points").fill_null(0)
    size = pl.when(pts < e[0]).then(pl.lit(f"0: no prior points (<{e[0]})"))
    for i in range(len(e) - 1):
        size = size.when(pts < e[i + 1]).then(pl.lit(f"{i + 1}: {e[i]}-{e[i + 1] - 1}"))
    size = size.otherwise(pl.lit(f"{len(e)}: >= {e[-1]}"))
    conf = pl.max_horizontal(pl.col("baseline_elo_probability_a"), 1 - pl.col("baseline_elo_probability_a"))
    cb = (conf * 10).floor().clip(5, 9).cast(pl.Int32)
    return df.with_columns(
        pl.col("surface").fill_null("(missing)"),
        pl.col("tournament_level").fill_null("(missing)"),
        size.alias("sample_size"),
        pl.when(pts < 1000).then(pl.lit("sparse (<1000 prior service points, either player)"))
        .otherwise(pl.lit("substantial (>=1000 for both)")).alias("history"),
        pl.format("{}-{}", cb * 10, cb * 10 + 10).alias("elo_confidence_pct"),
        pl.when(pl.col("tournament_level") == "G").then(pl.lit("Grand Slam")).otherwise(pl.lit("other tour-level"))
        .alias("event_type"),
        pl.col("prediction_batch").dt.year().alias("year"),
    )  # fmt: skip


SUBGROUPS = ("surface", "event_type", "tournament_level", "sample_size", "history", "elo_confidence_pct", "year")


def subgroups(df: pl.DataFrame, models: dict[str, str], by: str) -> pl.DataFrame:
    """Per-group n, log loss and Brier of each model column, and the difference to M0."""
    frames = []
    for name, col in models.items():
        m = metrics(df, col, [by]).select(
            by, "n", pl.col("log_loss").alias(f"ll_{name}"), pl.col("brier").alias(f"br_{name}")
        )
        frames.append(m if not frames else m.drop("n"))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=by, how="left")
    return out.sort(by)

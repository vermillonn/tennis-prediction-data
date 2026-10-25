"""Scoring of probabilistic predictions: log loss, Brier, accuracy, calibration, paired comparison."""

from __future__ import annotations

import math
import random

import polars as pl

EPS = 1e-15


def _loss_columns(prob: str, target: str = "target_a_wins") -> list[pl.Expr]:
    p = pl.col(prob).clip(EPS, 1 - EPS)
    y = pl.col(target).cast(pl.Float64)
    return [
        (-(y * p.log() + (1 - y) * (1 - p).log())).alias("_ll"),
        ((pl.col(prob) - y) ** 2).alias("_brier"),
        # A probability of exactly 0.5 is a coin flip: half credit.
        pl.when(pl.col(prob) == 0.5)
        .then(0.5)
        .otherwise(((pl.col(prob) > 0.5) == pl.col(target)).cast(pl.Float64))
        .alias("_acc"),
    ]


def _fmean(values: list[float]) -> float:
    """Exactly rounded mean: independent of row order and thread scheduling (polars sums are not)."""
    return math.fsum(values) / len(values) if values else float("nan")


def _exact_means(df: pl.DataFrame, by: list[str], cols: dict[str, str]) -> pl.DataFrame:
    """Per-group count and fsum-based means of ``cols`` ({output name: input column})."""
    grouped = (
        df.group_by(by).agg(pl.len().alias("n"), *[pl.col(c) for c in set(cols.values())])
        if by
        else (df.select(pl.len().alias("n"), *[pl.col(c).implode() for c in set(cols.values())]))
    )
    rows = []
    for r in grouped.iter_rows(named=True):
        rows.append({**{k: r[k] for k in by}, "n": r["n"], **{out: _fmean(r[c]) for out, c in cols.items()}})
    schema = {**{k: df.schema[k] for k in by}, "n": pl.UInt32, **dict.fromkeys(cols, pl.Float64)}
    out = pl.DataFrame(rows, schema=schema, orient="row")
    return out.sort(by) if by else out


def metrics(df: pl.DataFrame, prob: str, by: list[str] | None = None) -> pl.DataFrame:
    """n, log_loss, brier, accuracy, mean prediction and observed rate (optionally per group)."""
    scored = df.with_columns(*_loss_columns(prob), pl.col("target_a_wins").cast(pl.Float64).alias("_y"))
    cols = {"log_loss": "_ll", "brier": "_brier", "accuracy": "_acc", "mean_predicted_a": prob, "observed_a": "_y"}
    return _exact_means(scored, by or [], cols)


def calibration(df: pl.DataFrame, prob: str, bins: int) -> pl.DataFrame:
    """Equal-width probability bins on P(player A wins)."""
    b = (pl.col(prob) * bins).floor().cast(pl.Int32).clip(0, bins - 1)
    binned = df.with_columns(b.alias("bin"), pl.col("target_a_wins").cast(pl.Float64).alias("_y"))
    out = _exact_means(binned, ["bin"], {"mean_predicted": prob, "observed": "_y"}).with_columns(
        (pl.col("bin") / bins).alias("bin_lower"),
        ((pl.col("bin") + 1) / bins).alias("bin_upper"),
        (pl.col("observed") - pl.col("mean_predicted")).alias("observed_minus_predicted"),
    )
    return out.select("bin", "bin_lower", "bin_upper", "n", "mean_predicted", "observed", "observed_minus_predicted")


def expected_calibration_error(cal: pl.DataFrame) -> float:
    total = int(cal["n"].sum())
    terms = [n * abs(g) for n, g in zip(cal["n"], cal["observed_minus_predicted"], strict=True)]
    return math.fsum(terms) / total if total else float("nan")


def paired_bootstrap(
    df: pl.DataFrame, prob_new: str, prob_base: str, cluster: str, samples: int, seed: int
) -> dict[str, float]:
    """Difference new - base in log loss and Brier on identical rows, with a cluster bootstrap CI.

    Rows are resampled by whole clusters (weekly prediction batches), because predictions within
    a batch share a frozen rating state and are not independent.
    """
    new = df.with_columns(_loss_columns(prob_new)).select(
        cluster, pl.col("_ll").alias("ll_n"), pl.col("_brier").alias("br_n")
    )
    base = df.with_columns(_loss_columns(prob_base)).select(pl.col("_ll").alias("ll_b"), pl.col("_brier").alias("br_b"))
    per = (
        pl.concat([new, base], how="horizontal")
        .group_by(cluster)
        .agg(
            pl.len().alias("n"),
            (pl.col("ll_n") - pl.col("ll_b")).alias("d_ll"),
            (pl.col("br_n") - pl.col("br_b")).alias("d_br"),
        )
        .sort(cluster)
    )
    n = per["n"].to_list()
    d_ll = [math.fsum(x) for x in per["d_ll"].to_list()]
    d_br = [math.fsum(x) for x in per["d_br"].to_list()]
    total = sum(n)
    point_ll, point_br = math.fsum(d_ll) / total, math.fsum(d_br) / total
    rng = random.Random(seed)
    k = len(n)
    draws_ll, draws_br = [], []
    for _ in range(samples):
        idx = [rng.randrange(k) for _ in range(k)]
        m = sum(n[i] for i in idx)
        draws_ll.append(math.fsum(d_ll[i] for i in idx) / m)
        draws_br.append(math.fsum(d_br[i] for i in idx) / m)
    draws_ll.sort()
    draws_br.sort()

    def q(xs, p):
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]

    return {
        "rows": total,
        "clusters": k,
        "log_loss_diff": point_ll,
        "log_loss_diff_ci_low": q(draws_ll, 0.025),
        "log_loss_diff_ci_high": q(draws_ll, 0.975),
        "log_loss_diff_share_below_zero": sum(x < 0 for x in draws_ll) / samples,
        "brier_diff": point_br,
        "brier_diff_ci_low": q(draws_br, 0.025),
        "brier_diff_ci_high": q(draws_br, 0.975),
    }

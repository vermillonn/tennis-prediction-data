"""Paired M4A / surface-Elo scoring on identical matches, reusing the Phase 2/3B scorer.

``elo.evaluation.metrics`` (exactly rounded means, the same 1e-15 clipping, half credit for p = 0.5) gives log loss,
Brier and accuracy; ``modeling.evaluation.row_losses`` gives the per-match losses for the paired differences
``d_i = LL_M4A,i - LL_Elo,i`` (negative favours M4A). Nothing is fitted: probabilities are scored exactly as
committed (raw M4A, committed Elo baseline).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from tennis_data.modeling.evaluation import EPS, calibration, expected_calibration_error, metrics, row_losses

MODELS = {"M4A": "p_m4a", "Elo": "p_elo"}


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {"match_id": [r["match_id"] for r in rows], "p_m4a": [float(r["probability_a"]) for r in rows],
         "p_elo": [float(r["baseline_elo_probability_a"]) for r in rows],
         "target_a_wins": [bool(r["y"]) for r in rows]},
        schema={"match_id": pl.String, "p_m4a": pl.Float64, "p_elo": pl.Float64, "target_a_wins": pl.Boolean},
    )  # fmt: skip


def check_clip(protocol: dict) -> None:
    if protocol["probability_clip"] != EPS:
        raise ValueError(f"protocol clip {protocol['probability_clip']} differs from the shared scorer's {EPS}")


def _clean(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def score(rows: list[dict], protocol: dict) -> dict:
    """Point metrics of both models on the same rows, per-match paired differences and the probability spread."""
    check_clip(protocol)
    rows = sorted(rows, key=lambda r: r["match_id"])
    n = len(rows)
    out = {"n": n, "models": {}, "paired": None, "matches": [], "probability_distribution": []}
    if n == 0:
        return out
    df = frame(rows)
    bins = protocol["calibration_bins"]
    for name, col in MODELS.items():
        m = metrics(df, col).row(0, named=True)
        cal = calibration(df, col, bins)
        out["models"][name] = {k: _clean(m[k]) for k in ("log_loss", "brier", "accuracy", "mean_predicted_a",
                                                          "observed_a")} | {
            "ece": _clean(expected_calibration_error(cal))}  # fmt: skip
    y = df["target_a_wins"].to_numpy().astype(np.float64)
    ll_m, br_m = row_losses(df["p_m4a"].to_numpy(), y)
    ll_e, br_e = row_losses(df["p_elo"].to_numpy(), y)
    d_ll, d_br = ll_m - ll_e, br_m - br_e
    out["paired"] = {"n": n, "log_loss_diff": math.fsum(d_ll.tolist()) / n, "brier_diff": math.fsum(d_br.tolist()) / n,
                     "share_matches_m4a_better": float(np.mean(d_ll < 0))}  # fmt: skip
    out["matches"] = [{"match_id": r["match_id"], "seq": r["seq"], "y": r["y"], "p_m4a": r["probability_a"],
                       "p_elo": r["baseline_elo_probability_a"], "ll_m4a": float(a), "ll_elo": float(b),
                       "d_ll": float(d)} for r, a, b, d in zip(rows, ll_m, ll_e, d_ll, strict=True)]  # fmt: skip
    edges = [i / bins for i in range(bins + 1)]
    counts = np.histogram(df["p_m4a"].to_numpy(), bins=edges)[0]
    out["probability_distribution"] = [{"bin_lower": edges[i], "bin_upper": edges[i + 1], "n_m4a": int(c)}
                                       for i, c in enumerate(counts)]  # fmt: skip
    return out

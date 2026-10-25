"""Independent leakage audit of the recency/reliability inputs.

Recomputes, from the Phase 1 validated matches and the Phase 3A eligibility table only (not from the Phase 3A
state machine), each sampled player's released service history before the prediction batch and compares it with
the feature table. A match counts only if ``release_batch < prediction_batch`` of the target match. Agreement shows
that the inputs contain no statistics from the current match or from later-released matches.
"""

from __future__ import annotations

import math

import polars as pl

from tennis_data.modeling.config import ModelingConfig


def leakage_audit(cfg: ModelingConfig, ds: pl.DataFrame, sample: int = 300, seed: int = 7) -> dict:
    fdir = cfg.features.features_dir
    elig = pl.read_parquet(fdir / "match_stats_eligibility.parquet").filter(pl.col("stats_eligible"))
    vdir = cfg.features.elo.data.paths.validated_dir / "matches"
    ids = pl.concat([pl.read_parquet(p, columns=["match_id", "player_a_id", "player_b_id"])
                     for p in sorted(vdir.rglob("*.parquet"))])  # fmt: skip
    obs = elig.join(ids, on="match_id", how="inner", validate="1:1")
    long = pl.concat([
        obs.select("match_id", "prediction_batch", "release_batch", pl.col("player_a_id").alias("pid"),
                   pl.col("a_sp_played").alias("n"), pl.col("a_sp_won").alias("w")),
        obs.select("match_id", "prediction_batch", "release_batch", pl.col("player_b_id").alias("pid"),
                   pl.col("b_sp_played").alias("n"), pl.col("b_sp_won").alias("w")),
    ])  # fmt: skip
    hl = cfg.features.half_life_weeks[0]
    tag = f"{int(hl)}w" if float(hl).is_integer() else f"{hl}w"
    pts_col, rate_col = f"a_service_points_decay{tag}", f"a_spw_decay{tag}"
    feats = pl.concat([pl.read_parquet(p, columns=["match_id", "a_service_points_observed", pts_col, rate_col])
                       for p in sorted((fdir / "serve_return").rglob("*.parquet"))])  # fmt: skip
    rows = (
        ds.filter(pl.col("prediction_batch").dt.year() >= cfg.first_prediction_year)
        .select("match_id", "prediction_batch", "player_a_id")
        .sample(n=min(sample, ds.height), seed=seed)
        .sort("match_id")
        .join(feats, on="match_id", how="left")
    )
    worst_count, worst_pts, worst_rate, current_included = 0, 0.0, 0.0, 0
    for r in rows.iter_rows(named=True):
        hist = long.filter((pl.col("pid") == r["player_a_id"]) & (pl.col("release_batch") < r["prediction_batch"]))
        current_included += hist.filter(pl.col("match_id") == r["match_id"]).height
        observed = int(hist["n"].sum()) if hist.height else 0
        worst_count = max(worst_count, abs(observed - (r["a_service_points_observed"] or 0)))
        ages = [(r["prediction_batch"] - b).days / 7 for b in hist["prediction_batch"].to_list()]
        wts = [2 ** (-a / hl) for a in ages]
        den = math.fsum(w * n for w, n in zip(wts, hist["n"].to_list(), strict=True))
        num = math.fsum(w * x for w, x in zip(wts, hist["w"].to_list(), strict=True))
        feat_den = r[pts_col] or 0.0
        worst_pts = max(worst_pts, abs(den - feat_den) / max(1.0, den))
        if den > 0 and r[rate_col] is not None:
            worst_rate = max(worst_rate, abs(num / den - r[rate_col]))
        elif (den > 0) != (r[rate_col] is not None):
            worst_rate = float("inf")
    return {
        "sampled_rows": rows.height,
        "seed": seed,
        "max_abs_diff_service_points_observed": worst_count,
        "max_rel_diff_decay_points": worst_pts,
        "max_abs_diff_decay_rate": worst_rate,
        "current_match_in_history": current_included,
        "passed": worst_count == 0 and worst_pts < 1e-9 and worst_rate < 1e-9 and current_included == 0,
    }

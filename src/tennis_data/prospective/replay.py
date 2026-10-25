"""Historical replay: does the inference path reproduce the published Phase 2/3A/3B values?

1. State reconstruction: the full engine run over the archive snapshot must give every historical target exactly the
   Phase 2 surface-Elo probability and exactly the Phase 3A feature row.
2. Inference replay: selected historical matches are re-submitted as *pending* duplicates (no outcome, no updates)
   and priced with the frozen M4A annual fit of their year. The probabilities must equal the published Phase 3B
   predictions within ``[replay].tolerance``. Nothing is refitted to force agreement.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from tennis_data.prospective.config import ProspectiveConfig
from tennis_data.prospective.inference import load_walk_forward, predict_state
from tennis_data.prospective.state import StateSnapshot, UpcomingMatch, compute_state

CATEGORIES = {
    "established": pl.col("min_prior_service_points") >= 5000,
    "new_player": pl.col("has_history_diff") != 0,
    "sparse_history": (pl.col("min_prior_service_points") > 0) & (pl.col("min_prior_service_points") < 1000),
    "hard": pl.col("surface") == "Hard",
    "clay": pl.col("surface") == "Clay",
    "grass": pl.col("surface") == "Grass",
    "carpet": pl.col("surface") == "Carpet",
    "grand_slam": pl.col("tournament_level") == "G",
    "missing_recent_form": pl.col("decay_missing_diff") != 0,
    "multi_week_boundary": pl.col("unreleased_prior_result"),
    "development_period": pl.col("phase") == "development",
    "exploratory_period": pl.col("phase") == "exploratory",
}


def _unreleased(ds: pl.DataFrame) -> pl.Series:
    """True when a player has an earlier-started match whose result is not yet released at this batch."""
    long = pl.concat([
        ds.select("match_id", "prediction_batch", "release_batch", pl.col("player_a_id").alias("pid")),
        ds.select("match_id", "prediction_batch", "release_batch", pl.col("player_b_id").alias("pid")),
    ])  # fmt: skip
    pending_until = long.filter(pl.col("release_batch") > pl.col("prediction_batch")).select(
        "pid", pl.col("prediction_batch").alias("pb0"), pl.col("release_batch").alias("rb0")
    )
    me = long.select("match_id", "pid", "prediction_batch")
    hit = (
        me.join(pending_until, on="pid")
        .filter((pl.col("pb0") < pl.col("prediction_batch")) & (pl.col("rb0") >= pl.col("prediction_batch")))
        .select("match_id")
        .unique()
    )
    return ds["match_id"].is_in(hit["match_id"].implode())


def select_cases(cfg: ProspectiveConfig) -> pl.DataFrame:
    mdir = cfg.modeling.modeling_dir
    sel = pl.read_parquet(mdir / "predictions" / "selected.parquet")
    sel = sel.filter(~pl.col("model_id").str.contains("@static"))
    ds = pl.read_parquet(mdir / "dataset.parquet")
    ds = ds.with_columns(_unreleased(ds).alias("unreleased_prior_result"))
    j = sel.select("match_id", "predicted_probability_a", "baseline_elo_probability_a", "phase").join(
        ds.select("match_id", "prediction_batch", "tournament_date", "tournament_id", "tournament_level", "surface",
                  "player_a_id", "player_b_id", "min_prior_service_points", "has_history_diff", "decay_missing_diff",
                  "unreleased_prior_result"),
        on="match_id", how="inner", validate="1:1",
    )  # fmt: skip
    picks = []
    for name, cond in CATEGORIES.items():
        sub = j.filter(cond).sort("match_id")
        n = min(cfg.replay_cases_per_category, sub.height)
        if n:
            picks.append(sub.sample(n=n, seed=cfg.replay_seed).with_columns(pl.lit(name).alias("category")))
    ids = pl.concat(picks)["match_id"].unique().implode()
    cases = j.filter(pl.col("match_id").is_in(ids))
    tags = pl.concat_list([pl.when(cond).then(pl.lit(name)).otherwise(None) for name, cond in CATEGORIES.items()])
    return cases.with_columns(tags.list.drop_nulls().alias("category")).sort("match_id")


def _fixtures(cases: pl.DataFrame, cfg: ProspectiveConfig) -> list[UpcomingMatch]:
    v = cfg.data.paths.validated_dir / "matches"
    meta = pl.concat(
        [
            pl.read_parquet(p, columns=["match_id", "round", "draw_size", "best_of"])
            for p in sorted(v.rglob("*.parquet"))
        ]
    )
    c = cases.join(meta, on="match_id", how="left", validate="1:1")
    return [
        UpcomingMatch(f"replay:{r['match_id']}", r["player_a_id"], r["player_b_id"], r["tournament_id"],
                      r["tournament_date"], r["tournament_level"], r["surface"], r["round"], r["draw_size"], r["best_of"])
        for r in c.iter_rows(named=True)
    ]  # fmt: skip


def run_replay(snapshot: StateSnapshot, cfg: ProspectiveConfig, extra: list[UpcomingMatch] = ()) -> dict:
    """``extra`` pending matches share the same engine run (e.g. a freshness probe); they are not replay cases."""
    t0 = time.perf_counter()
    cases = select_cases(cfg)
    fixtures = _fixtures(cases, cfg)
    st = compute_state(snapshot, fixtures + list(extra), cfg, keep_history=True)
    t_state = time.perf_counter() - t0
    # 1. state reconstruction against the published Phase 2 and Phase 3A outputs
    elo_pub = pl.read_parquet(cfg.elo.elo_dir / "predictions" / "model_name=surface_elo" / "part-0.parquet").select(
        "match_id", pl.col("predicted_probability_a").alias("pub")
    )
    hist = st["history"]
    # Rows added by the backfill are not in the published outputs; every published row must be reproduced exactly.
    e = elo_pub.join(hist.select("match_id", "predicted_probability_a"), on="match_id", how="left")
    elo_rows = e.height
    elo_missing = e.filter(pl.col("predicted_probability_a").is_null()).height
    elo_err = float((e["predicted_probability_a"] - e["pub"]).abs().max() or 0.0)
    extra_rows = hist.height - hist.filter(pl.col("match_id").is_in(elo_pub["match_id"].implode())).height
    feat_pub = pl.concat(
        [pl.read_parquet(p) for p in sorted((cfg.features.features_dir / "serve_return").rglob("*.parquet"))]
    )
    feat_cols = feat_pub.columns
    feat_mine = (hist.filter(pl.col("match_id").is_in(feat_pub["match_id"].implode())).select(feat_cols)
                 .sort("prediction_batch", "match_id"))  # fmt: skip
    feat_pub = feat_pub.sort("prediction_batch", "match_id")
    feat_equal = feat_mine.equals(feat_pub)
    diff_cols = [] if feat_equal else [c for c in feat_cols if feat_mine.height != feat_pub.height
                                       or not feat_mine[c].equals(feat_pub[c])]  # fmt: skip
    # 2. inference replay with the frozen annual fits
    pend = st["pending"].filter(pl.col("match_id").str.starts_with("pending:replay:"))
    pend = pend.with_columns(pl.col("match_id").str.replace("^pending:replay:", "").alias("orig"))
    preds = []
    for y in sorted(pend["prediction_batch"].dt.year().unique().to_list()):
        art = load_walk_forward(cfg, y)
        sub = pend.filter(pl.col("prediction_batch").dt.year() == y)
        p = predict_state(sub, art, cfg, prediction_mode="historical_reconstruction",
                          snapshot_id=snapshot.snapshot_id, components=snapshot.components)  # fmt: skip
        preds.append(p.with_columns(pl.Series("orig", sub["orig"].to_list())))
    pr = pl.concat(preds, how="vertical_relaxed").join(cases.select(
        pl.col("match_id").alias("orig"), "predicted_probability_a", pl.col("baseline_elo_probability_a").alias("pub_elo"),
        "category"), on="orig", how="left", validate="1:1")  # fmt: skip
    pr = pr.with_columns(
        (pl.col("probability_a") - pl.col("predicted_probability_a")).abs().alias("abs_error"),
        (pl.col("baseline_elo_probability_a") - pl.col("pub_elo")).abs().alias("elo_abs_error"),
        (pl.col("probability_a") + pl.col("probability_b") - 1).abs().alias("symmetry_error"),
    )
    by_cat = (pr.explode("category").group_by("category")
              .agg(pl.len().alias("n"), pl.col("abs_error").max().alias("max_abs_error"),
                   pl.col("elo_abs_error").max().alias("max_elo_abs_error")).sort("category"))  # fmt: skip
    tol = cfg.replay_tolerance
    result = {
        "cases": pr.height,
        "max_abs_error": float(pr["abs_error"].max()),
        "max_elo_abs_error": float(pr["elo_abs_error"].max()),
        "max_symmetry_error": float(pr["symmetry_error"].max()),
        "tolerance": tol,
        "by_category": by_cat.to_dicts(),
        "years": sorted(pend["prediction_batch"].dt.year().unique().to_list()),
        "state_reconstruction": {
            "elo_rows": elo_rows,
            "elo_unmatched_rows": elo_missing,
            "elo_max_abs_error": elo_err,
            "feature_rows": feat_pub.height,
            "feature_columns": len(feat_cols),
            "features_identical": feat_equal,
            "differing_feature_columns": diff_cols,
            "rows_added_by_backfill": extra_rows,
        },  # fmt: skip
        "seconds_state": round(t_state, 1),
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    result["passed"] = bool(
        result["max_abs_error"] <= tol
        and result["max_elo_abs_error"] <= tol
        and feat_equal
        and elo_missing == 0
        and elo_err <= tol
        and result["max_symmetry_error"] <= 1e-12
    )
    return {"summary": result, "predictions": pr, "state": st}


__all__ = ["run_replay", "select_cases", "CATEGORIES", "np"]

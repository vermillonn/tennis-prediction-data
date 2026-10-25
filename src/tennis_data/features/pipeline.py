"""Build the Phase 3A feature table and its intermediate outputs."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict

import polars as pl

from tennis_data.features.config import FeatureConfig
from tennis_data.features.dictionary import schema
from tennis_data.features.observations import feature_frame, load_matches, observations
from tennis_data.features.snapshots import build_snapshots
from tennis_data.storage.parquet import tree_checksums, write_partitioned, write_single

DIFFS = {
    "spw_diff": ("a_spw_shrunk", "b_spw_shrunk"),
    "rpw_diff": ("a_rpw_shrunk", "b_rpw_shrunk"),
    "surface_spw_diff": ("a_surface_spw", "b_surface_spw"),
    "surface_rpw_diff": ("a_surface_rpw", "b_surface_rpw"),
    "serve_adjusted_diff": ("a_serve_adjusted", "b_serve_adjusted"),
    "return_adjusted_diff": ("a_return_adjusted", "b_return_adjusted"),
    "matchup_spw_diff": ("a_spw_vs_b", "b_spw_vs_a"),
}


DATA_FILES = ("match_stats_eligibility.parquet", "opponent_fit_log.parquet", "opponent_fit_parameters.parquet")


def data_checksums(cfg: FeatureConfig) -> dict[str, str]:
    """SHA-256 of the deterministic data outputs (excludes performance.json, whose timings vary)."""
    out = cfg.features_dir
    sums = {f"serve_return/{k}": v for k, v in tree_checksums(out / "serve_return").items()}
    for name in DATA_FILES:
        if (out / name).exists():
            sums[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    return dict(sorted(sums.items()))


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux reports KiB


def assemble(snap: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    target = schema(cfg)
    df = snap
    for name, (a, b) in DIFFS.items():
        if name in target:
            df = df.with_columns((pl.col(a) - pl.col(b)).alias(name))
    df = df.with_columns(
        pl.lit(cfg.full_version).alias("feature_version"), pl.lit(cfg.configuration_hash()).alias("configuration_hash")
    )
    missing = [c for c in target if c not in df.columns]
    if missing:
        raise RuntimeError(f"feature table is missing registered columns: {missing}")
    extra = [c for c in df.columns if c not in target and not c.startswith("_")]
    if extra:
        raise RuntimeError(f"feature table has unregistered columns: {extra}")
    return df.select([pl.col(c).cast(t, strict=True) for c, t in target.items()]).sort("prediction_batch", "match_id")


def compute(cfg: FeatureConfig, matches: pl.DataFrame | None = None, quarantine_ids=None) -> dict:
    """Everything in memory: frame, observations, snapshot result and the final table."""
    t0 = time.perf_counter()
    if matches is None:
        matches, quarantine_ids = load_matches(cfg)
    frame = feature_frame(matches, cfg, quarantine_ids)
    obs = observations(frame)
    t1 = time.perf_counter()
    snap = build_snapshots(frame, obs, cfg)
    table = assemble(snap.features, cfg)
    t2 = time.perf_counter()
    timings = {
        "load_and_frame_seconds": t1 - t0,
        **snap.timings,
        "assemble_seconds": t2 - t1 - sum(snap.timings.values()),
    }
    return {"frame": frame, "observations": obs, "snap": snap, "table": table, "timings": timings}


def build_features(cfg: FeatureConfig) -> dict:
    t0 = time.perf_counter()
    res = compute(cfg)
    table, frame = res["table"], res["frame"]
    out = cfg.features_dir
    out.mkdir(parents=True, exist_ok=True)
    write_partitioned(table, out / "serve_return", partition_col="season")
    stats = frame.select(
        "match_id", "season", "pool", "prediction_batch", "release_batch", "match_status", "stats_status",
        "is_target", "is_update", "stats_eligible", "stats_exclusion_reason",
        "a_sp_played", "a_sp_won", "b_sp_played", "b_sp_won",
    ).sort("match_id")  # fmt: skip
    write_single(stats, out / "match_stats_eligibility.parquet")
    snap = res["snap"]
    if snap.fit_log.height:
        # Wall-clock seconds vary between runs; they go to performance.json, not the data file.
        write_single(snap.fit_log.drop("seconds").sort("fit_batch", "pool"), out / "opponent_fit_log.parquet")
    if cfg.opponent.store_fits and snap.fit_params.height:
        write_single(snap.fit_params.sort("fit_batch", "pool", "player_id"), out / "opponent_fit_parameters.parquet")
    total = time.perf_counter() - t0
    summary = {
        "feature_version": cfg.full_version,
        "configuration_hash": cfg.configuration_hash(),
        "configuration": cfg.hash_payload(),
        "shrinkage": asdict(cfg.shrinkage),
        "opponent": asdict(cfg.opponent),
        "rows": table.height,
        "columns": len(table.columns),
        "outputs": data_checksums(cfg),
    }
    perf = {
        **res["timings"],
        "total_seconds": total,
        "peak_rss_mb": peak_rss_mb(),
        "opponent_fits": snap.fit_log.height,
        "opponent_fit_seconds_max": float(snap.fit_log["seconds"].max()) if snap.fit_log.height else 0.0,
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    # Timings vary run to run, so they are kept out of the reproducible summary.
    (out / "performance.json").write_text(json.dumps(perf, indent=2, sort_keys=True) + "\n")
    return {**res, "summary": summary, "performance": perf}

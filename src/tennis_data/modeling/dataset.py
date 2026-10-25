"""Modeling dataset: Phase 2 surface-Elo predictions joined 1:1 with Phase 3A features.

Every model input is a *derived antisymmetric* feature: swapping player A and player B negates it.
Each derived feature is built only from upstream columns that pass the Phase 3A model-input guard
(``features.model_inputs.check_model_inputs``), so labels, current-match statistics and audit fields can
never reach a model. Identifiers and labels stay in the table for evaluation but are never inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tennis_data.features.model_inputs import check_model_inputs
from tennis_data.modeling.config import MODELING_ENGINE_VERSION, ModelingConfig
from tennis_data.storage.parquet import write_single


class DatasetError(ValueError):
    """The upstream tables cannot be joined consistently."""


@dataclass(frozen=True)
class Derived:
    name: str
    base: tuple[str, ...]
    build: Callable[[float], pl.Expr]  # epsilon -> expression
    description: str
    fill_zero: bool = True  # null -> 0 (the only value consistent with antisymmetry)


def _d(a: str, b: str) -> Callable[[float], pl.Expr]:
    return lambda eps: pl.col(a) - pl.col(b)


def _logit(eps: float) -> pl.Expr:
    p = pl.col("predicted_probability_a").clip(eps, 1 - eps)
    return (p / (1 - p)).log()


def _logdiff(a: str, b: str) -> Callable[[float], pl.Expr]:
    return lambda eps: pl.col(a).log1p() - pl.col(b).log1p()


def _flagdiff(a: str, b: str) -> Callable[[float], pl.Expr]:
    return lambda eps: pl.col(a).cast(pl.Float64) - pl.col(b).cast(pl.Float64)


def _nulldiff(a: str, b: str) -> Callable[[float], pl.Expr]:
    return lambda eps: pl.col(a).is_null().cast(pl.Float64) - pl.col(b).is_null().cast(pl.Float64)


DERIVED: dict[str, Derived] = {
    d.name: d
    for d in (
        Derived("elo_logit", ("predicted_probability_a",), _logit,
                "log(p/(1-p)) of the Phase 2 surface-Elo probability, p clipped to [eps, 1-eps]", False),
        Derived("spw_shrunk_diff", ("spw_diff",), lambda e: pl.col("spw_diff"), "a_spw_shrunk - b_spw_shrunk"),
        Derived("rpw_shrunk_diff", ("rpw_diff",), lambda e: pl.col("rpw_diff"), "a_rpw_shrunk - b_rpw_shrunk"),
        Derived("surface_spw_diff", ("surface_spw_diff",), lambda e: pl.col("surface_spw_diff"),
                "a_surface_spw - b_surface_spw (heuristic surface blend)"),
        Derived("surface_rpw_diff", ("surface_rpw_diff",), lambda e: pl.col("surface_rpw_diff"),
                "a_surface_rpw - b_surface_rpw"),
        Derived("serve_adjusted_diff", ("serve_adjusted_diff",), lambda e: pl.col("serve_adjusted_diff"),
                "opponent-adjusted SPW vs an average returner, A - B"),
        Derived("return_adjusted_diff", ("return_adjusted_diff",), lambda e: pl.col("return_adjusted_diff"),
                "opponent-adjusted RPW vs an average server, A - B"),
        Derived("matchup_spw_diff", ("matchup_spw_diff",), lambda e: pl.col("matchup_spw_diff"),
                "a_spw_vs_b - b_spw_vs_a: expected serve-point edge in this matchup (antisymmetric)"),
        Derived("spw_decay_diff", ("a_spw_decay52w", "b_spw_decay52w"), _d("a_spw_decay52w", "b_spw_decay52w"),
                "52-week decay-weighted SPW, A - B"),
        Derived("rpw_decay_diff", ("a_rpw_decay52w", "b_rpw_decay52w"), _d("a_rpw_decay52w", "b_rpw_decay52w"),
                "52-week decay-weighted RPW, A - B"),
        Derived("decay_missing_diff", ("a_spw_decay52w", "b_spw_decay52w"), _nulldiff("a_spw_decay52w", "b_spw_decay52w"),
                "1[A has no decayed history] - 1[B has none]", False),
        Derived("log_service_ess_diff", ("a_service_ess_decay52w", "b_service_ess_decay52w"),
                _logdiff("a_service_ess_decay52w", "b_service_ess_decay52w"),
                "log1p(Kish ESS of A's decayed service rate) - same for B"),
        Derived("log_stat_matches_diff", ("a_stat_matches", "b_stat_matches"), _logdiff("a_stat_matches", "b_stat_matches"),
                "log1p(prior stats matches A) - log1p(B)"),
        Derived("has_history_diff", ("a_has_stats_history", "b_has_stats_history"),
                _flagdiff("a_has_stats_history", "b_has_stats_history"), "1[A has history] - 1[B has history]"),
        Derived("service_prior_dominated_diff", ("a_service_prior_dominated", "b_service_prior_dominated"),
                _flagdiff("a_service_prior_dominated", "b_service_prior_dominated"),
                "1[A's service estimate is prior-dominated] - same for B"),
    )
}  # fmt: skip

# Columns kept for evaluation and bookkeeping only (never model inputs).
KEYS = (
    "match_id", "prediction_batch", "release_batch", "tournament_date", "season", "tournament_id",
    "tournament_level", "surface", "player_a_id", "player_b_id",
)  # fmt: skip
LABELS = ("target_a_wins",)
EVAL_ONLY = ("baseline_elo_probability_a", "phase2_period", "min_prior_service_points")


def derived_inputs(names) -> list[str]:
    unknown = [n for n in names if n not in DERIVED]
    if unknown:
        raise DatasetError(f"unknown derived features {unknown}; registry: {sorted(DERIVED)}")
    base = sorted({b for n in names for b in DERIVED[n].base})
    return base


def assert_inputs_allowed(names, cfg: ModelingConfig) -> None:
    """Every input must be a registered derived feature built from guard-approved upstream columns."""
    check_model_inputs(derived_inputs(names), cfg.features, include_elo=True)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inputs(cfg: ModelingConfig) -> tuple[list[Path], list[Path], list[Path]]:
    elo_dir = cfg.features.elo.elo_dir / "predictions" / f"model_name={cfg.baseline}"
    feat_dir = cfg.features.features_dir / "serve_return"
    elo_files = sorted(elo_dir.rglob("*.parquet"))
    feat_files = sorted(feat_dir.rglob("*.parquet"))
    if not elo_files:
        raise FileNotFoundError(f"No Phase 2 predictions under {elo_dir}. Run `tennis-data elo-backtest`.")
    if not feat_files:
        raise FileNotFoundError(f"No Phase 3A features under {feat_dir}. Run `tennis-data features-build`.")
    validated = sorted((cfg.features.elo.data.paths.validated_dir / "matches").rglob("*.parquet"))
    if not validated:
        raise FileNotFoundError("No validated matches. Run `tennis-data build`.")
    return elo_files, feat_files, validated


def input_fingerprint(cfg: ModelingConfig) -> dict:
    elo_files, feat_files, validated = _inputs(cfg)

    def tag(files):
        return {f"{p.parent.name}/{p.name}": _sha(p) for p in files}

    return {
        "engine": MODELING_ENGINE_VERSION,
        "epsilon": cfg.epsilon,
        "elo": tag(elo_files),
        "features": tag(feat_files),
        "validated": tag(validated),
    }


def join_tables(elo: pl.DataFrame, feats: pl.DataFrame, labels: pl.DataFrame, cfg: ModelingConfig) -> pl.DataFrame:
    """Validated 1:1 join; raises DatasetError on any inconsistency."""
    for name, df in (("Elo predictions", elo), ("features", feats)):
        if df["match_id"].n_unique() != df.height:
            raise DatasetError(f"duplicate match_id in {name}")
    if elo.height != feats.height:
        raise DatasetError(f"row counts differ: Elo {elo.height} vs features {feats.height}")
    j = elo.join(feats, on="match_id", how="inner", suffix="_f", validate="1:1")
    if j.height != elo.height:
        raise DatasetError(f"{elo.height - j.height} Elo rows have no feature row")
    checks = {
        "player_a_id": "player_a_id_f", "player_b_id": "player_b_id_f", "prediction_batch": "prediction_batch_f",
        "tournament_level": "tournament_level_f", "surface": "surface_f", "tournament_date": "tournament_date_f",
    }  # fmt: skip
    for a, b in checks.items():
        bad = j.filter(pl.col(a).ne_missing(pl.col(b))).height
        if bad:
            raise DatasetError(f"{bad} rows disagree on {a} between Elo predictions and features")
    j = j.join(labels.select("match_id", "player_a_id", "player_a_won"), on="match_id", how="left", suffix="_v",
               validate="1:1")  # fmt: skip
    if j.filter(pl.col("player_a_won").is_null()).height:
        raise DatasetError("some rows are missing from the validated match table")
    if j.filter(pl.col("player_a_id") != pl.col("player_a_id_v")).height:
        raise DatasetError("player A differs from the validated table")
    if j.filter(pl.col("target_a_wins") != pl.col("player_a_won")).height:
        raise DatasetError("Elo target_a_wins disagrees with the validated outcome")
    return j


def build_from_frames(
    elo: pl.DataFrame, feats: pl.DataFrame, labels: pl.DataFrame, cfg: ModelingConfig
) -> pl.DataFrame:
    j = join_tables(elo, feats, labels, cfg)
    exprs = []
    for d in DERIVED.values():
        e = d.build(cfg.epsilon)
        exprs.append((e.fill_null(0.0) if d.fill_zero else e).cast(pl.Float64).alias(d.name))
    out = j.with_columns(
        pl.col("predicted_probability_a").alias("baseline_elo_probability_a"),
        pl.col("evaluation_period").alias("phase2_period"),
        pl.min_horizontal("a_service_points_observed", "b_service_points_observed").alias("min_prior_service_points"),
        *exprs,
    )
    # Record how many values were filled (for the reports) before they disappear.
    out = out.with_columns(
        *[DERIVED[n].build(cfg.epsilon).is_null().alias(f"_filled_{n}") for n in DERIVED if DERIVED[n].fill_zero]
    )
    cols = (
        list(KEYS)
        + list(LABELS)
        + list(EVAL_ONLY)
        + list(DERIVED)
        + [f"_filled_{n}" for n in DERIVED if DERIVED[n].fill_zero]
    )
    return out.select(cols).sort("prediction_batch", "match_id")


def build_dataset(cfg: ModelingConfig, use_cache: bool = True) -> tuple[pl.DataFrame, dict]:
    """Return (dataset, manifest). Reuses data/modeling/dataset.parquet when the inputs are unchanged."""
    out = cfg.modeling_dir
    out.mkdir(parents=True, exist_ok=True)
    fp = input_fingerprint(cfg)
    key = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    cache, manifest_path = out / "dataset.parquet", out / "dataset_manifest.json"
    if use_cache and cache.exists() and manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        if man.get("input_key") == key and man.get("sha256") == _sha(cache):
            return pl.read_parquet(cache), {**man, "_reused": True}
    elo_files, feat_files, validated = _inputs(cfg)
    elo = pl.concat([pl.read_parquet(p) for p in elo_files])
    feats = pl.concat([pl.read_parquet(p) for p in feat_files])
    labels = pl.concat([pl.read_parquet(p, columns=["match_id", "player_a_id", "player_a_won"]) for p in validated])
    _check_chronology(cfg)
    ds = build_from_frames(elo, feats, labels, cfg)
    write_single(ds, cache)
    man = {"input_key": key, "inputs": fp, "rows": ds.height, "sha256": _sha(cache)}
    manifest_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return ds, man


def _check_chronology(cfg: ModelingConfig) -> None:
    elo_sum = cfg.features.elo.reports_dir / "elo_summary.json"
    feat_sum = cfg.features.reports_dir / "feature_summary.json"
    if elo_sum.exists() and feat_sum.exists():
        a = json.loads(elo_sum.read_text()).get("chronology")
        b = json.loads(feat_sum.read_text()).get("configuration", {}).get("chronology")
        if a != b:
            raise DatasetError("Elo and feature outputs were built with different chronology settings")


def model_matrix(ds: pl.DataFrame, inputs, cfg: ModelingConfig):
    """The ONLY way inputs are extracted for fitting/prediction (guarded)."""
    import numpy as np

    assert_inputs_allowed(inputs, cfg)
    return ds.select(list(inputs)).to_numpy().astype(np.float64)

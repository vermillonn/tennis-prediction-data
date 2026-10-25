"""Walk-forward backtest: tune on the validation period, evaluate once on the test period.

1. Every candidate parameter set replays the full history (ratings only ever use earlier batches).
2. The global model and the surface model are each selected by log loss on validation-period
   predictions only. Test-period outcomes never influence any choice.
3. The selected models are replayed, written as the historical prediction dataset, and scored on
   the test period: on exactly the same matches for both models.
4. Sensitivity: the selected parameters are replayed with retirements added to the rating
   updates (not retuned), and scored on the same test matches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import polars as pl
import polars.selectors as cs

from tennis_data import __version__
from tennis_data.elo.config import Eligibility, EloConfig, EloParams
from tennis_data.elo.data import engine_frame, load_validated
from tennis_data.elo.engine import EloRun, run_elo, surface_probability
from tennis_data.elo.evaluation import calibration, expected_calibration_error, metrics, paired_bootstrap
from tennis_data.storage.parquet import tree_checksums, write_partitioned, write_single

GLOBAL, SURFACE = "global_elo", "surface_elo"

PREDICTION_COLUMNS = (
    "match_id", "tournament_date", "prediction_batch", "release_batch", "evaluation_period", "pool",
    "tournament_id", "tournament_level", "round", "surface", "surface_key", "match_status",
    "player_a_id", "player_b_id",
    "player_a_elo_before", "player_b_elo_before", "global_match_count_a", "global_match_count_b",
    "surface_elo_a_before", "surface_elo_b_before", "surface_match_count_a", "surface_match_count_b",
    "surface_effective_elo_a", "surface_effective_elo_b",
    "predicted_probability_a", "target_a_wins", "model_name", "model_version", "configuration_hash",
)  # fmt: skip


@dataclass
class Candidate:
    params: EloParams
    validation: dict


def period_expr(cfg: EloConfig) -> pl.Expr:
    ev = cfg.evaluation
    b = pl.col("prediction_batch")
    return (
        pl.when(b < ev.validation_start)
        .then(pl.lit("warmup"))
        .when(b < ev.test_start)
        .then(pl.lit("validation"))
        .when(b <= ev.test_end)
        .then(pl.lit("test"))
        .otherwise(pl.lit("after_test"))
        .alias("evaluation_period")
    )


def _scored(run: EloRun, frame: pl.DataFrame, shrinkage: float) -> pl.DataFrame:
    meta = frame.filter("is_target").select(
        "match_id", "tournament_date", "prediction_batch", "release_batch", "evaluation_period", "pool",
        "tournament_id", "tournament_level", "round", "surface", "surface_key", "match_status",
        "player_a_id", "player_b_id", "target_a_wins",
    )  # fmt: skip
    return surface_probability(meta.join(run.predictions, on="match_id", how="inner", validate="1:1"), shrinkage)


def _grid(cfg: EloConfig) -> list[EloParams]:
    base, t = cfg.params, cfg.tuning
    ks = [replace(base, k_mode="constant", k_constant=k) for k in t.k_constant]
    ks += [replace(base, k_mode="decay", k_numerator=a, k_offset=b, k_shape=c) for a, b, c in t.k_decay]
    return ks or [base]


def _row(model: str, p: EloParams, m: dict, surface: bool = True) -> dict:
    row = {k: v for k, v in asdict(p).items() if k != "initial_rating"}
    unused = ("k_numerator", "k_offset", "k_shape") if p.k_mode == "constant" else ("k_constant",)
    row.update(dict.fromkeys(unused))
    if not surface:  # global ratings do not depend on surface settings
        row.update(surface_init=None, carpet_policy=None, shrinkage_matches=None)
    return {"model": model, **row, **m}


def tune(cfg: EloConfig, frame: pl.DataFrame) -> tuple[pl.DataFrame, EloParams, EloParams]:
    """Grid search on the validation period. Returns (grid table, best global, best surface)."""
    rows = []
    best = {GLOBAL: None, SURFACE: None}
    for k_params in _grid(cfg):
        for i, init in enumerate(cfg.tuning.surface_init):
            p = replace(k_params, surface_init=init)
            run = run_elo(frame, p)
            scored = _scored(run, frame, 0.0).filter(pl.col("evaluation_period") == "validation")
            if i == 0:  # global ratings do not depend on surface settings
                gm = metrics(scored, "global_probability_a").row(0, named=True)
                rows.append(_row(GLOBAL, p, gm, surface=False))
                if best[GLOBAL] is None or gm["log_loss"] < best[GLOBAL][1]:
                    best[GLOBAL] = (p, gm["log_loss"])
            for m in cfg.tuning.shrinkage_matches:
                sm = metrics(surface_probability(scored, m), "surface_probability_a").row(0, named=True)
                ps = replace(p, shrinkage_matches=m)
                rows.append(_row(SURFACE, ps, sm))
                if best[SURFACE] is None or sm["log_loss"] < best[SURFACE][1]:
                    best[SURFACE] = (ps, sm["log_loss"])
    grid = pl.DataFrame(rows, infer_schema_length=None)
    g, s = best[GLOBAL][0], best[SURFACE][0]
    # The global model has no surface component; keep the surface model's surface settings so the
    # informational surface columns in its output are comparable.
    g = replace(g, surface_init=s.surface_init, shrinkage_matches=s.shrinkage_matches)
    return grid, g, s


def predictions_table(
    cfg: EloConfig, frame: pl.DataFrame, params: EloParams, model: str
) -> tuple[pl.DataFrame, EloRun]:
    mcfg = cfg.with_params(params)
    run = run_elo(frame, params, record_history=(model == SURFACE))
    scored = _scored(run, frame, params.shrinkage_matches)
    prob = "global_probability_a" if model.startswith(GLOBAL) else "surface_probability_a"
    out = scored.with_columns(
        pl.col(prob).alias("predicted_probability_a"),
        pl.lit(model).alias("model_name"),
        pl.lit(mcfg.full_model_version).alias("model_version"),
        pl.lit(mcfg.configuration_hash()).alias("configuration_hash"),
    ).select(PREDICTION_COLUMNS)
    return out.sort("prediction_batch", "match_id"), run


def _evaluate(cfg: EloConfig, preds: dict[str, pl.DataFrame], period: str) -> dict:
    ev = cfg.evaluation
    sets = {name: df.filter(pl.col("evaluation_period") == period) for name, df in preds.items()}
    ids = {name: set(df["match_id"]) for name, df in sets.items()}
    reference = ids[GLOBAL]
    for name, s in ids.items():
        if s != reference:
            raise RuntimeError(f"{name} is scored on different {period} matches than {GLOBAL}; comparison invalid")
    res = {"overall": [], "by_surface": [], "by_level": [], "calibration": [], "ece": {}}
    for name, df in sets.items():
        tag = pl.lit(name).alias("model")
        res["overall"].append(metrics(df, "predicted_probability_a").select(tag, pl.all()))
        res["by_surface"].append(
            metrics(
                df.with_columns(pl.col("surface").fill_null("(missing)")), "predicted_probability_a", ["surface"]
            ).select(tag, pl.all())
        )
        res["by_level"].append(metrics(df, "predicted_probability_a", ["tournament_level"]).select(tag, pl.all()))
        cal = calibration(df, "predicted_probability_a", ev.calibration_bins)
        res["calibration"].append(cal.select(tag, pl.all()))
        res["ece"][name] = expected_calibration_error(cal)
    for key in ("overall", "by_surface", "by_level", "calibration"):
        res[key] = pl.concat(res[key], how="vertical")
    joined = sets[GLOBAL].select(
        "match_id", "prediction_batch", "target_a_wins", pl.col("predicted_probability_a").alias("p_global")
    )
    comparisons = {}
    for name, df in sets.items():
        if name == GLOBAL:
            continue
        both = joined.join(
            df.select("match_id", pl.col("predicted_probability_a").alias("p_new")), on="match_id", validate="1:1"
        )
        comparisons[f"{name} - {GLOBAL}"] = paired_bootstrap(
            both, "p_new", "p_global", "prediction_batch", ev.bootstrap_samples, ev.bootstrap_seed
        )
    res["comparisons"] = comparisons
    return res


def exclusions(frame: pl.DataFrame, quarantine_path: Path) -> pl.DataFrame:
    rows = frame.group_by("evaluation_period", "exclusion_reason").len("matches")
    rows = rows.with_columns(
        pl.col("exclusion_reason").fill_null("eligible_target"), pl.lit("validated").alias("table")
    )
    if quarantine_path.exists():
        q = pl.read_parquet(quarantine_path, columns=["quarantine_categories"]).with_columns(
            pl.col("quarantine_categories").list.join("+").alias("exclusion_reason")
        )
        qrows = (
            q.group_by("exclusion_reason")
            .len("matches")
            .with_columns(pl.lit("n/a").alias("evaluation_period"), pl.lit("quarantine").alias("table"))
        )
        rows = pl.concat([rows, qrows.select(rows.columns)], how="vertical")
    return rows.select("table", "evaluation_period", "exclusion_reason", "matches").sort(
        "table", "evaluation_period", "exclusion_reason"
    )


def _write_tables(tables_dir: Path, tables: dict[str, pl.DataFrame]) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.with_columns(cs.float().round(6)).write_csv(tables_dir / f"{name}.csv")


def backtest(cfg: EloConfig) -> dict:
    matches, quarantine_ids = load_validated(cfg)
    frame = engine_frame(matches, cfg, quarantine_ids).with_columns(period_expr(cfg))

    grid, best_g, best_s = tune(cfg, frame)
    preds, runs = {}, {}
    for model, params in ((GLOBAL, best_g), (SURFACE, best_s)):
        preds[model], runs[model] = predictions_table(cfg, frame, params, model)

    sens = {}
    if cfg.eligibility.sensitivity_retirements and "retired" not in cfg.eligibility.update_statuses:
        e = Eligibility(cfg.eligibility.update_statuses + ("retired",), cfg.eligibility.target_statuses, False)
        scfg = cfg.with_eligibility(e)
        sframe = engine_frame(matches, scfg, quarantine_ids).with_columns(period_expr(scfg))
        for model, params in ((GLOBAL, best_g), (SURFACE, best_s)):
            name = f"{model}_with_retirements"
            sens[name], _ = predictions_table(scfg, sframe, params, name)

    test = _evaluate(cfg, {**preds, **sens}, "test")
    validation = _evaluate(cfg, preds, "validation")

    # --- outputs ---
    cfg.elo_dir.mkdir(parents=True, exist_ok=True)
    write_partitioned(pl.concat(list(preds.values())), cfg.elo_dir / "predictions", partition_col="model_name")
    if sens:
        write_partitioned(
            pl.concat(list(sens.values())), cfg.elo_dir / "sensitivity_predictions", partition_col="model_name"
        )
    write_single(runs[SURFACE].history, cfg.elo_dir / "rating_history.parquet")

    excl = exclusions(frame, cfg.data.paths.quarantine_dir / "matches.parquet")
    tables = {
        "tuning_grid": grid,
        "test_overall": test["overall"],
        "test_by_surface": test["by_surface"],
        "test_by_level": test["by_level"],
        "test_calibration": test["calibration"],
        "validation_overall": validation["overall"],
        "exclusions": excl,
    }
    _write_tables(cfg.reports_dir / "tables", tables)

    build_summary = cfg.data.paths.reports_dir / "build_summary.json"
    summary = {
        "package_version": __version__,
        "model_version": cfg.full_model_version,
        "source_commit": cfg.data.source.commit,
        "input_validated_checksums": json.loads(build_summary.read_text())["outputs"]["validated"]
        if build_summary.exists()
        else None,
        "selected": {
            GLOBAL: {"params": asdict(best_g), "configuration_hash": cfg.with_params(best_g).configuration_hash()},
            SURFACE: {"params": asdict(best_s), "configuration_hash": cfg.with_params(best_s).configuration_hash()},
        },
        "periods": {
            "validation": [str(cfg.evaluation.validation_start), str(cfg.evaluation.test_start)],
            "test": [str(cfg.evaluation.test_start), str(cfg.evaluation.test_end)],
        },
        "chronology": asdict(cfg.chronology),
        "test_comparisons": test["comparisons"],
        "validation_comparisons": validation["comparisons"],
        "test_ece": test["ece"],
        "outputs": {
            "predictions": tree_checksums(cfg.elo_dir / "predictions"),
            "sensitivity_predictions": tree_checksums(cfg.elo_dir / "sensitivity_predictions") if sens else {},
            "rating_history.parquet": tree_checksums(cfg.elo_dir)["rating_history.parquet"],
        },
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "elo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return {"summary": summary, "tables": tables, "test": test, "validation": validation, "frame": frame}

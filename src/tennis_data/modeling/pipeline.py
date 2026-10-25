"""Phase 3B pipeline: dataset -> development -> selection -> freeze -> exploratory -> production -> reports."""

from __future__ import annotations

import hashlib
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.modeling import experiments as X
from tennis_data.modeling.artifacts import prediction_frame, sha256, write_json, write_predictions
from tennis_data.modeling.audit import leakage_audit
from tennis_data.modeling.config import ModelingConfig
from tennis_data.modeling.dataset import build_dataset
from tennis_data.modeling.evaluation import (
    SUBGROUPS,
    add_groups,
    calibration,
    expected_calibration_error,
    paired_bootstrap,
    summarize,
)
from tennis_data.storage.parquet import write_single

CODE_FILES = ("config", "dataset", "folds", "preprocessing", "logistic", "calibration", "evaluation",
              "experiments", "artifacts", "audit", "pipeline", "report")  # fmt: skip


def code_hash(cfg: ModelingConfig) -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in CODE_FILES:
        p = here / f"{name}.py"
        h.update(name.encode() + b"\0" + (p.read_bytes() if p.exists() else b"") + b"\0")
    h.update(cfg.config_path.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# ------------------------------------------------------------------------------------------- evaluation helpers


def _metrics_block(p, y, cfg) -> dict:
    return summarize(p, y, cfg.calibration_bins)


def _boot(cfg, p_new, p_base, y, clusters, alphas=(0.05,)):
    return paired_bootstrap(p_new, p_base, y, clusters, cfg.bootstrap_samples, cfg.bootstrap_seed, alphas)


def _clusters(ds: pl.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    sub = ds[rows]
    return {
        "week": sub["prediction_batch"].cast(pl.Int32).to_numpy(),
        "tournament": sub.select(pl.format("{}|{}", "tournament_id", pl.col("prediction_batch").dt.year()))
        .to_series()
        .to_numpy(),
        "season": sub["prediction_batch"].dt.year().to_numpy(),
    }


def evaluate_block(cfg, ds, data, rows, models: dict[str, np.ndarray], base: str, comparisons, alphas) -> dict:
    """Metrics for every model on identical ``rows``, paired bootstraps, subgroups and calibration tables."""
    y = data.y[rows]
    cl = _clusters(ds, rows)
    out = {"rows": int(len(rows)), "weeks": int(len(np.unique(cl["week"]))), "metrics": {}, "boot": {},
           "sensitivity": {}, "calibration": {}, "subgroups": {}}  # fmt: skip
    for name, p in models.items():
        out["metrics"][name] = _metrics_block(p, y, cfg)
    for new, ref in comparisons:
        out["boot"][f"{new} vs {ref}"] = _boot(cfg, models[new], models[ref], y, cl["week"], alphas)
    return out


def subgroup_table(cfg, ds, data, rows, models: dict[str, np.ndarray], base: str, compare: list[str]):
    sub = add_groups(ds[rows], cfg)
    y = data.y[rows]
    weeks = sub["prediction_batch"].cast(pl.Int32).to_numpy()
    recs = []
    for g in SUBGROUPS:
        keys = sub[g].to_numpy()
        for level in sorted(set(keys.tolist())):
            m = keys == level
            rec = {"group": g, "level": str(level), "n": int(m.sum())}
            for name, p in models.items():
                rec[f"ll_{name}"] = summarize(p[m], y[m], cfg.calibration_bins)["log_loss"]
            for name in compare:
                b = _boot(cfg, models[name][m], models[base][m], y[m], weeks[m])
                rec[f"d_{name}"] = b["log_loss_diff"]
                rec[f"lo_{name}"], rec[f"hi_{name}"] = b["ll_ci95"]
            recs.append(rec)
    return pl.DataFrame(recs)


def calibration_table(cfg, data, rows, models: dict[str, np.ndarray]) -> pl.DataFrame:
    frames = []
    for name, p in models.items():
        df = pl.DataFrame({"p": p, "target_a_wins": data.y[rows].astype(bool)})
        frames.append(calibration(df, "p", cfg.calibration_bins).with_columns(pl.lit(name).alias("model")))
    return pl.concat(frames).select("model", pl.exclude("model"))


def ece_of(cfg, data, rows, p) -> float:
    df = pl.DataFrame({"p": p, "target_a_wins": data.y[rows].astype(bool)})
    return expected_calibration_error(calibration(df, "p", cfg.calibration_bins))


# --------------------------------------------------------------------------------------------------------- run


def run_modeling(cfg: ModelingConfig, use_cache: bool = True, write_reports: bool = True) -> dict:
    t0 = time.perf_counter()
    timings = {}
    ds, ds_manifest = build_dataset(cfg, use_cache=use_cache)
    timings["dataset"] = time.perf_counter() - t0
    data = X.Data.of(ds)
    audit = leakage_audit(cfg, ds)
    if not audit["passed"]:
        raise AssertionError(f"leakage audit failed: {audit}")
    provenance = {
        "dataset_input_key": ds_manifest["input_key"],
        "dataset_sha256": ds_manifest["sha256"],
        "source_manifest": {k: ds_manifest["inputs"][k] for k in ("elo", "features")},
        "validated_files": len(ds_manifest["inputs"]["validated"]),
        "upstream_feature_hash": cfg.features.configuration_hash(),
        "code_hash": code_hash(cfg),
    }
    d0, d1 = cfg.development_years
    e0, e1 = cfg.exploratory_years
    m0_all = ds["baseline_elo_probability_a"].to_numpy()

    # 1-3. development, selection, freeze ---------------------------------------------------------------
    t = time.perf_counter()
    dev = X.development(data, cfg)
    sel = X.select(dev, data, cfg)
    frozen = X.freeze(dev, sel, data, cfg, provenance)
    timings["development"] = time.perf_counter() - t
    mdir = cfg.modeling_dir
    write_json(frozen, mdir / "models" / "frozen_model.json")
    frozen_sha_at_freeze = sha256(mdir / "models" / "frozen_model.json")

    wf_rows = next(iter(dev["wfs"].values())).rows
    dev_pos = np.flatnonzero((data.year[wf_rows] >= d0) & (data.year[wf_rows] <= d1))
    dev_rows = wf_rows[dev_pos]
    sel_id, sel_cal = sel["selected"], sel["calibration"]

    def dev_pred(exp_id: str, method: str = "raw") -> np.ndarray:
        if method == "raw":
            return dev["wfs"][(exp_id, dev["chosen_c"][exp_id])].p[dev_pos]
        return dev["calibrated"][(exp_id, method)][dev_pos]

    dev_models = {"M0": m0_all[dev_rows]}
    for e in cfg.experiments:
        dev_models[e.id] = dev_pred(e.id)
    final_id = f"{sel_id}+{sel_cal}"
    dev_models[final_id] = dev_pred(sel_id, sel_cal)
    for method in cfg.calibration_methods:
        if method != "raw":
            dev_models[f"{sel_id}+{method}"] = dev_pred(sel_id, method)
            dev_models[f"M0R+{method}"] = dev_pred("M0R", method)
    n_cmp = len(cfg.experiments)
    bonf = 0.05 / n_cmp
    comps = [(e.id, "M0") for e in cfg.experiments]
    if final_id not in {e.id for e in cfg.experiments}:
        comps.append((final_id, "M0"))
    devres = evaluate_block(cfg, ds, data, dev_rows, dev_models, "M0", comps, (0.05, bonf))
    y_dev = data.y[dev_rows]
    cl_dev = _clusters(ds, dev_rows)
    devres["sensitivity"] = {
        f"{name} vs M0 [{c}]": _boot(cfg, dev_models[name], dev_models["M0"], y_dev, cl_dev[c])
        for name in dict.fromkeys([final_id, "M0R"])
        for c in ("week", "tournament", "season")
    }
    folds_out = []
    for a, b in cfg.folds:
        m = (data.year[dev_rows] >= a) & (data.year[dev_rows] <= b)
        rec = {"fold": f"{a}-{b}", "rows": int(m.sum()), "weeks": int(len(np.unique(cl_dev["week"][m])))}
        for name in ("M0", "M0R", sel_id, final_id):
            mm = _metrics_block(dev_models[name][m], y_dev[m], cfg)
            rec[f"ll_{name}"], rec[f"brier_{name}"], rec[f"ece_{name}"] = mm["log_loss"], mm["brier"], mm["ece"]
        for name in dict.fromkeys(["M0R", final_id]):
            bb = _boot(cfg, dev_models[name][m], dev_models["M0"][m], y_dev[m], cl_dev["week"][m])
            rec[f"d_{name}"], (rec[f"lo_{name}"], rec[f"hi_{name}"]) = bb["log_loss_diff"], bb["ll_ci95"]
        folds_out.append(rec)
    devres["folds"] = pl.DataFrame(folds_out)
    devres["subgroups"] = subgroup_table(cfg, ds, data, dev_rows, {k: dev_models[k] for k in ("M0", "M0R", final_id)},
                                         "M0", list(dict.fromkeys(["M0R", final_id])))  # fmt: skip
    devres["calibration"] = calibration_table(
        cfg, data, dev_rows, {k: v for k, v in dev_models.items() if k.startswith(("M0", sel_id))}
    )
    ablation_pairs = [
        ("M0R", "M0"), ("M1", "M0R"), ("A1_SPW", "M0R"), ("A1_RPW", "M0R"), ("M2", "M1"), ("A2_SURF", "M0R"),
        ("M3", "M0R"), ("M3", "A3_NOMATCHUP"), ("A3_NOMATCHUP", "M0R"), ("A3_MATCHUP", "M0R"), ("M4A", "M1"),
        ("M4B", "M1"), ("A4_DECAY", "M0R"), ("A4_ESS", "M0R"), ("M4A", "A4_DECAY"), ("A0_NOELO", "M0"),
    ]  # fmt: skip
    ablation_pairs = [(a, b) for a, b in ablation_pairs if a in dev_models and b in dev_models]
    devres["ablations"] = {f"{a} vs {b}": _boot(cfg, dev_models[a], dev_models[b], y_dev, cl_dev["week"])
                           for a, b in ablation_pairs}  # fmt: skip

    # 4. exploratory ------------------------------------------------------------------------------------
    t = time.perf_counter()
    expl = X.exploratory(data, cfg, frozen, dev)
    timings["exploratory"] = time.perf_counter() - t
    if sha256(mdir / "models" / "frozen_model.json") != frozen_sha_at_freeze:
        raise AssertionError("frozen model file changed during the exploratory evaluation")
    ex_wf_rows = expl["by_exp"][sel_id]["wf"].rows
    ex_pos = np.flatnonzero(data.year[ex_wf_rows] >= e0)
    ex_rows = ex_wf_rows[ex_pos]
    if not np.array_equal(ex_rows, expl["static_rows"]):
        raise AssertionError("exploratory rows mismatch")

    def ex_pred(exp_id: str, method: str = "raw") -> np.ndarray:
        if method == "raw":
            return expl["by_exp"][exp_id]["wf"].p[ex_pos]
        return expl["by_exp"][exp_id]["cal"][method][0][ex_pos]

    ex_models = {"M0": m0_all[ex_rows]}
    for e in cfg.experiments:
        ex_models[e.id] = ex_pred(e.id)
    ex_models[final_id] = ex_pred(sel_id, sel_cal)
    ex_models[f"{final_id} (static 2010 fit)"] = expl["static_p"]
    for method in cfg.calibration_methods:
        if method != "raw":
            ex_models[f"{sel_id}+{method}"] = ex_pred(sel_id, method)
            ex_models[f"M0R+{method}"] = ex_pred("M0R", method)
    primary = [(final_id, "M0"), ("M0R", "M0"), (f"{final_id} (static 2010 fit)", "M0")]
    secondary = [(e.id, "M0") for e in cfg.experiments if e.id not in (final_id, "M0R")]
    exres = evaluate_block(cfg, ds, data, ex_rows, ex_models, "M0", list(dict.fromkeys(primary + secondary)),
                           (0.05, bonf))  # fmt: skip
    y_ex = data.y[ex_rows]
    cl_ex = _clusters(ds, ex_rows)
    exres["sensitivity"] = {
        f"{name} vs M0 [{c}]": _boot(cfg, ex_models[name], ex_models["M0"], y_ex, cl_ex[c])
        for name in dict.fromkeys([final_id, "M0R"])
        for c in ("week", "tournament", "season")
    }
    periods = [(2010, 2014), (2015, 2019), (2020, 2023), (2024, e1)]
    per = []
    for a, b in periods:
        m = (data.year[ex_rows] >= a) & (data.year[ex_rows] <= b)
        if not m.any():
            continue
        rec = {"period": f"{a}-{b}", "rows": int(m.sum())}
        for name in ("M0", "M0R", final_id):
            rec[f"ll_{name}"] = _metrics_block(ex_models[name][m], y_ex[m], cfg)["log_loss"]
        bb = _boot(cfg, ex_models[final_id][m], ex_models["M0"][m], y_ex[m], cl_ex["week"][m])
        rec["d"], (rec["lo"], rec["hi"]) = bb["log_loss_diff"], bb["ll_ci95"]
        per.append(rec)
    exres["periods"] = pl.DataFrame(per)
    exres["subgroups"] = subgroup_table(cfg, ds, data, ex_rows, {k: ex_models[k] for k in ("M0", "M0R", final_id)},
                                        "M0", list(dict.fromkeys(["M0R", final_id])))  # fmt: skip
    exres["calibration"] = calibration_table(
        cfg, data, ex_rows, {k: v for k, v in ex_models.items() if k.startswith(("M0", sel_id))}
    )
    exres["ablations"] = {f"{a} vs {b}": _boot(cfg, ex_models[a], ex_models[b], y_ex, cl_ex["week"])
                          for a, b in ablation_pairs}  # fmt: skip

    # 5. production -------------------------------------------------------------------------------------
    prod = X.production(data, cfg, frozen, expl)

    # artifacts -----------------------------------------------------------------------------------------
    t = time.perf_counter()
    chash = cfg.configuration_hash()
    coef_rows, splits = [], []
    for (exp_id, c), wf in sorted(dev["wfs"].items()):
        for y, f in sorted(wf.fits.items()):
            for i, col in enumerate(wf.inputs):
                coef_rows.append({"phase": "development", "experiment": exp_id, "C": c, "year": y,
                                  "training_cutoff": f.cutoff, "train_rows": f.train_rows, "feature": col,
                                  "scale": f.scaler.scale[i], "coef_scaled": float(f.coef[i]),
                                  "coef_original_units": float(f.coef_original()[i]), "iterations": f.iterations,
                                  "converged": f.converged})  # fmt: skip
    for exp_id, v in sorted(expl["by_exp"].items()):
        wf = v["wf"]
        for y, f in sorted(wf.fits.items()):
            if y < e0:
                continue
            for i, col in enumerate(wf.inputs):
                coef_rows.append({"phase": "exploratory", "experiment": exp_id, "C": wf.c, "year": y,
                                  "training_cutoff": f.cutoff, "train_rows": f.train_rows, "feature": col,
                                  "scale": f.scaler.scale[i], "coef_scaled": float(f.coef[i]),
                                  "coef_original_units": float(f.coef_original()[i]), "iterations": f.iterations,
                                  "converged": f.converged})  # fmt: skip
    coefs = pl.DataFrame(coef_rows)
    sel_wf = expl["by_exp"][sel_id]["wf"]
    for y, f in sorted(sel_wf.fits.items()):
        splits.append({"year": y, "role": ("calibration history" if y < d0 else "development" if y <= d1
                                           else "exploratory"),  # fmt: skip
                       "training_cutoff_batch": f.cutoff, "train_rows": f.train_rows,
                       "first_train_batch": cfg.train_start, "last_train_release_batch": f.last_train_release,
                       "predict_rows": int((data.year == y).sum())})  # fmt: skip
    splits_df = pl.DataFrame(splits)
    pred_dir = mdir / "predictions"

    def frame_for(rows, model_id, p, cutoffs, cal_cutoffs, extra=None):
        meta = {"model_id": model_id, "model_version": cfg.full_version, "configuration_hash": chash}
        cols = {"predicted_probability_a": p, "training_cutoff": cutoffs, "calibration_cutoff": cal_cutoffs,
                **(extra or {})}  # fmt: skip
        return prediction_frame(ds, rows, cols, meta)

    def cutoff_arr(rows):
        return np.array([data.cutoffs[int(y)] for y in data.year[rows]], dtype="datetime64[D]")

    all_frames = []
    for e in cfg.experiments:
        wf = expl["by_exp"][e.id]["wf"]
        dwf = dev["wfs"][(e.id, dev["chosen_c"][e.id])]
        n_dev = len(dwf.rows)
        if not (np.array_equal(wf.rows[:n_dev], dwf.rows) and np.array_equal(wf.p[:n_dev], dwf.p)):
            raise AssertionError(f"{e.id}: exploratory walk-forward does not reproduce development predictions")
        cut = cutoff_arr(wf.rows)
        extra = {"raw_probability_a": wf.p, "C": np.full(len(wf.rows), wf.c)}
        before_dev = data.year[wf.rows] < d0
        for method in cfg.calibration_methods:
            if method == "raw":
                continue
            p = np.concatenate([dev["calibrated"][(e.id, method)], expl["by_exp"][e.id]["cal"][method][0][n_dev:]])
            p[before_dev] = np.nan  # no calibrated prediction before the development period
            extra[f"{method}_probability_a"] = p
        all_frames.append(frame_for(wf.rows, e.id, wf.p, cut, cut, extra))
    exp_preds = pl.concat(all_frames, how="vertical_relaxed").with_columns(
        pl.col("temperature_probability_a").fill_nan(None), pl.col("isotonic_probability_a").fill_nan(None)
    )
    write_predictions(exp_preds, pred_dir / "experiments.parquet")
    # The frozen selected model (development + exploratory), with its frozen calibration method.
    rows_sel = np.concatenate([dev_rows, ex_rows])
    p_sel = np.concatenate([dev_models[final_id], ex_models[final_id]])
    cut_sel = cutoff_arr(rows_sel)
    phase = np.where(data.year[rows_sel] <= d1, "development", "exploratory")
    sel_frame = frame_for(rows_sel, final_id, p_sel, cut_sel, cut_sel, {"phase": phase})
    static = frame_for(ex_rows, f"{final_id}@static{e0}", expl["static_p"],
                       np.full(len(ex_rows), np.datetime64(data.cutoffs[e0], "D")),
                       np.full(len(ex_rows), np.datetime64(data.cutoffs[e0], "D")),
                       {"phase": np.full(len(ex_rows), "exploratory")})  # fmt: skip
    write_predictions(pl.concat([sel_frame, static]), pred_dir / "selected.parquet")
    write_single(coefs.sort("phase", "experiment", "C", "year", "feature"), mdir / "models" / "coefficients.parquet")
    write_single(dev["grid"].sort("experiment", "C"), mdir / "models" / "development_grid.parquet")
    write_json(prod, mdir / "models" / "production_model.json")
    timings["artifacts"] = time.perf_counter() - t
    artifacts = {}
    for p in sorted(mdir.rglob("*")):
        if p.is_file() and p.suffix in (".parquet", ".json") and p.name not in ("manifest.json", "performance.json"):
            artifacts[p.relative_to(mdir).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    n_fits = dev["fits"] + expl["fits"] + 2
    manifest = {
        "model_version": cfg.full_version,
        "configuration_hash": chash,
        "configuration": cfg.hash_payload(),
        **provenance,
        "frozen_hash": frozen["frozen_hash"],
        "selected_model": final_id,
        "splits": splits_df.to_dicts(),
        "development_rows": int(len(dev_rows)),
        "exploratory_rows": int(len(ex_rows)),
        "fits": n_fits,
        "leakage_audit": audit,
        "artifacts": artifacts,
    }
    write_json(manifest, mdir / "manifest.json")
    timings["total"] = time.perf_counter() - t0
    perf = {
        "seconds": {k: round(v, 2) for k, v in timings.items()},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "logistic_fits": n_fits,
        "calibrator_fits": len(cfg.experiments) * (len(cfg.calibration_methods) - 1) * (d1 - d0 + 1 + e1 - e0 + 1),
        "matches_evaluated": {"development": int(len(dev_rows)), "exploratory": int(len(ex_rows))},
        "artifact_bytes": sum(a["bytes"] for a in artifacts.values()),
        "git_commit": git_commit(),
        "dataset_cache_reused": ds_manifest.get("_reused", False),
    }
    write_json(perf, mdir / "performance.json")

    results = {
        "cfg": cfg, "provenance": provenance, "dev": dev, "sel": sel, "frozen": frozen, "devres": devres,
        "exres": exres, "prod": prod, "splits": splits_df, "coefs": coefs, "audit": audit, "final_id": final_id,
        "manifest": manifest, "perf": perf, "bonferroni_alpha": bonf, "n_comparisons": n_cmp,
        "dataset": ds, "data": data, "dev_rows": dev_rows, "ex_rows": ex_rows,
    }  # fmt: skip
    if write_reports:
        from tennis_data.modeling.report import write_reports as _wr

        results["reports"] = _wr(results)
    return results

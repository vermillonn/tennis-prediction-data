"""Phase 3B experiment protocol: development grid, pre-specified selection, freeze, exploratory, production.

Order of operations (enforced by the code, not just by convention):

1. ``development``: walk-forward annual refits for prediction years ``first_prediction_year..dev_end`` only.
   No row with a prediction batch after the development period is ever read for fitting or scoring.
2. ``select``: C per experiment, then the candidate model, then its calibration method, all by pooled
   development log loss with the pre-specified simplicity rule.
3. ``freeze``: the selected specification and its 2010-cutoff fit are written to ``frozen_model.json`` with a
   hash. Nothing after this step may change it.
4. ``exploratory``: the frozen specification is applied walk-forward to 2010-2026. Exploratory results are
   reported only; they feed no decision.
5. ``production``: the frozen specification refitted on every released label, for prospective use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl

from tennis_data.modeling import folds
from tennis_data.modeling.calibration import Calibrator, fit_calibrator
from tennis_data.modeling.config import ModelingConfig
from tennis_data.modeling.dataset import DERIVED, assert_inputs_allowed
from tennis_data.modeling.evaluation import row_losses
from tennis_data.modeling.logistic import fit_logistic, predict
from tennis_data.modeling.preprocessing import Scaler, fit_scaler

SIMPLICITY_ORDER = ("raw", "temperature", "isotonic")


@dataclass
class Data:
    """Numpy view of the modeling dataset (rows sorted by prediction_batch, match_id)."""

    ds: pl.DataFrame
    x: dict[str, np.ndarray]
    y: np.ndarray
    batch: np.ndarray  # days since epoch
    release: np.ndarray
    year: np.ndarray
    cutoffs: dict[int, date]

    @classmethod
    def of(cls, ds: pl.DataFrame) -> Data:
        days = pl.col("prediction_batch").cast(pl.Int32)
        arr = ds.select(
            days.alias("b"), pl.col("release_batch").cast(pl.Int32).alias("r"), folds.year_col().alias("yr")
        )
        return cls(
            ds=ds,
            x={n: ds[n].to_numpy().astype(np.float64) for n in DERIVED},
            y=ds["target_a_wins"].cast(pl.Float64).to_numpy(),
            batch=arr["b"].to_numpy(),
            release=arr["r"].to_numpy(),
            year=arr["yr"].to_numpy(),
            cutoffs=folds.cutoffs(ds),
        )

    def matrix(self, inputs, rows: np.ndarray) -> np.ndarray:
        return np.column_stack([self.x[c][rows] for c in inputs]) if inputs else np.zeros((rows.sum(), 0))


def _days(d: date) -> int:
    return (d - date(1970, 1, 1)).days


@dataclass
class YearFit:
    year: int
    cutoff: date
    train_rows: int
    last_train_release: date | None
    scaler: Scaler
    coef: np.ndarray
    iterations: int
    converged: bool
    symmetry_max_dev: float

    def coef_original(self) -> np.ndarray:
        return self.coef / np.asarray(self.scaler.scale)


@dataclass
class WalkForward:
    """Out-of-fold raw predictions of one experiment at one C for a range of prediction years."""

    exp_id: str
    inputs: tuple[str, ...]
    c: float
    rows: np.ndarray  # dataset row indices, ascending
    p: np.ndarray
    fits: dict[int, YearFit] = field(default_factory=dict)


def fit_cutoff(data: Data, cfg: ModelingConfig, inputs, c: float, cutoff: date) -> tuple[Scaler, np.ndarray, dict]:
    tr = (data.batch >= _days(cfg.train_start)) & (data.release < _days(cutoff))
    x = data.matrix(inputs, tr)
    scaler = fit_scaler(x, inputs)
    fit = fit_logistic(scaler.transform(x), data.y[tr], c, cfg.max_iterations, cfg.tolerance)
    last = int(data.release[tr].max()) if tr.any() else None
    info = {
        "train_rows": int(tr.sum()),
        "last_train_release": date(1970, 1, 1) + timedelta(days=last) if last is not None else None,
        "iterations": fit.iterations,
        "converged": fit.converged,
    }
    return scaler, fit.coef, info


def walk_forward(data: Data, cfg: ModelingConfig, exp_id: str, inputs, c: float, years) -> WalkForward:
    assert_inputs_allowed(inputs, cfg)
    idx, ps, fits = [], [], {}
    for y in years:
        cutoff = data.cutoffs[y]
        scaler, coef, info = fit_cutoff(data, cfg, inputs, c, cutoff)
        rows = np.flatnonzero(data.year == y)
        x = scaler.transform(data.matrix(inputs, data.year == y))
        p = predict(coef, x)
        sym = float(np.max(np.abs(p + predict(coef, -x) - 1))) if len(p) else 0.0
        fits[y] = YearFit(y, cutoff, info["train_rows"], info["last_train_release"], scaler, coef,
                          info["iterations"], info["converged"], sym)  # fmt: skip
        idx.append(rows)
        ps.append(p)
    return WalkForward(exp_id, tuple(inputs), c, np.concatenate(idx), np.concatenate(ps), fits)


def calibrate(wf: WalkForward, data: Data, cfg: ModelingConfig, years, method: str):
    """Calibrated predictions for ``years``; calibrator for y fitted on OOF rows of y-window..y-1 released < cutoff."""
    out = wf.p.copy()
    cals: dict[int, Calibrator] = {}
    yr, rel = data.year[wf.rows], data.release[wf.rows]
    for y in years:
        cutoff = _days(data.cutoffs[y])
        fit_rows = (yr >= y - cfg.calibration_window) & (yr < y) & (rel < cutoff)
        if method != "raw" and int(fit_rows.sum()) == 0:
            raise ValueError(f"no calibration rows for {wf.exp_id} {y}")
        if method == "isotonic" and int(fit_rows.sum()) < cfg.isotonic_min_rows:
            raise ValueError(f"too few rows for isotonic calibration in {y}")
        cal = fit_calibrator(method, wf.p[fit_rows], data.y[wf.rows][fit_rows], cfg.epsilon)
        cals[y] = cal
        tgt = yr == y
        out[tgt] = cal.apply(wf.p[tgt])
        q = np.linspace(0.001, 0.999, 999)
        sym = float(np.max(np.abs(cal.apply(q) + cal.apply(1 - q) - 1)))
        if sym > 1e-9:
            raise AssertionError(f"calibrator {method} {y} breaks symmetry by {sym}")
    return out, cals


def pooled_ll(p: np.ndarray, y: np.ndarray) -> float:
    import math

    return math.fsum(row_losses(p, y)[0]) / len(p)


def choose_with_tolerance(scores: dict, order, tol: float):
    """Lowest score; then the earliest entry of ``order`` (simplest) within ``tol`` of it."""
    best = min(scores.values())
    for k in order:
        if k in scores and scores[k] <= best + tol:
            return k, best
    raise AssertionError


# --------------------------------------------------------------------------------------------- development


def development(data: Data, cfg: ModelingConfig) -> dict:
    d0, d1 = cfg.development_years
    years = list(range(cfg.first_prediction_year, d1 + 1))
    if max(data.year[data.year <= d1]) != d1:
        raise ValueError("development years missing from the dataset")
    dev_mask_all = (data.year >= d0) & (data.year <= d1)
    grid, wfs, n_fits = [], {}, 0
    for exp in cfg.experiments:
        for c in cfg.c_grid:
            wf = walk_forward(data, cfg, exp.id, exp.inputs, c, years)
            n_fits += len(years)
            dev = (data.year[wf.rows] >= d0) & (data.year[wf.rows] <= d1)
            row = {"experiment": exp.id, "C": c, "n_inputs": len(exp.inputs),
                   "dev_log_loss": pooled_ll(wf.p[dev], data.y[wf.rows][dev]), "dev_rows": int(dev.sum())}  # fmt: skip
            for a, b in cfg.folds:
                m = (data.year[wf.rows] >= a) & (data.year[wf.rows] <= b)
                row[f"ll_{a}_{b}"] = pooled_ll(wf.p[m], data.y[wf.rows][m])
            row["max_symmetry_dev"] = max(f.symmetry_max_dev for f in wf.fits.values())
            row["all_converged"] = all(f.converged for f in wf.fits.values())
            grid.append(row)
            wfs[(exp.id, c)] = wf
    grid_df = pl.DataFrame(grid)
    chosen_c = {}
    for exp in cfg.experiments:
        g = grid_df.filter(pl.col("experiment") == exp.id).sort("dev_log_loss", "C")
        chosen_c[exp.id] = float(g["C"][0])  # ties -> smaller C (stronger regularisation)
    dev_years = list(range(d0, d1 + 1))
    calibrated, calibrators = {}, {}
    for exp in cfg.experiments:
        wf = wfs[(exp.id, chosen_c[exp.id])]
        for method in cfg.calibration_methods:
            p, cals = calibrate(wf, data, cfg, dev_years, method)
            calibrated[(exp.id, method)] = p
            calibrators[(exp.id, method)] = cals
    m0 = data.ds["baseline_elo_probability_a"].to_numpy()
    expected = np.flatnonzero(np.isin(data.year, years))
    if any(not np.array_equal(wf.rows, expected) for wf in wfs.values()):
        raise AssertionError("models were not evaluated on identical rows")
    del dev_mask_all
    return {"grid": grid_df, "wfs": wfs, "chosen_c": chosen_c, "calibrated": calibrated, "calibrators": calibrators,
            "years": years, "dev_years": dev_years, "m0": m0, "fits": n_fits}  # fmt: skip


def select(dev: dict, data: Data, cfg: ModelingConfig) -> dict:
    d0, d1 = cfg.development_years
    raw_scores = {}
    for e in cfg.candidates:
        wf = dev["wfs"][(e, dev["chosen_c"][e])]
        m = (data.year[wf.rows] >= d0) & (data.year[wf.rows] <= d1)
        raw_scores[e] = pooled_ll(wf.p[m], data.y[wf.rows][m])
    by_simplicity = sorted(cfg.candidates, key=lambda e: (len(cfg.experiment(e).inputs), cfg.candidates.index(e)))
    lowest = min(raw_scores, key=lambda e: (raw_scores[e], cfg.candidates.index(e)))
    chosen, _ = choose_with_tolerance(raw_scores, by_simplicity, cfg.simplicity_tolerance)
    wf = dev["wfs"][(chosen, dev["chosen_c"][chosen])]
    m = (data.year[wf.rows] >= d0) & (data.year[wf.rows] <= d1)
    cal_scores = {k: pooled_ll(dev["calibrated"][(chosen, k)][m], data.y[wf.rows][m]) for k in cfg.calibration_methods}
    method, _ = choose_with_tolerance(cal_scores, SIMPLICITY_ORDER, cfg.simplicity_tolerance)
    return {"candidate_dev_log_loss": raw_scores, "lowest": lowest, "selected": chosen, "C": dev["chosen_c"][chosen],
            "calibration_scores": cal_scores, "calibration": method}  # fmt: skip


# ---------------------------------------------------------------------------------------------------- freeze


def _json_default(o):
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating | np.integer):
        return o.item()
    raise TypeError(type(o))


def canonical_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()


def fit_spec_at(data: Data, cfg: ModelingConfig, spec: dict, cutoff: date, cal_rows: np.ndarray | None,
                cal_p: np.ndarray | None) -> dict:  # fmt: skip
    scaler, coef, info = fit_cutoff(data, cfg, spec["inputs"], spec["C"], cutoff)
    cal = fit_calibrator(spec["calibration"], cal_p, data.y[cal_rows], cfg.epsilon) if cal_rows is not None else None
    return {
        "training_cutoff_batch": cutoff,
        "training_rule": f"prediction_batch >= {cfg.train_start} and release_batch < {cutoff}",
        **info,
        "scale": scaler.to_dict(),
        "coef_scaled": dict(zip(spec["inputs"], coef.tolist(), strict=True)),
        "coef_original_units": dict(zip(spec["inputs"], (coef / np.asarray(scaler.scale)).tolist(), strict=True)),
        "calibrator": None
        if cal is None
        else {
            **cal.summary(),
            "slope": cal.slope,
            "knots_x": list(cal.knots_x),
            "knots_y": list(cal.knots_y),
        },  # fmt: skip
    }


def freeze(dev: dict, sel: dict, data: Data, cfg: ModelingConfig, provenance: dict) -> dict:
    e = cfg.experiment(sel["selected"])
    spec = {"experiment": e.id, "description": e.description, "inputs": list(e.inputs), "C": sel["C"],
            "calibration": sel["calibration"], "calibration_window_years": cfg.calibration_window,
            "epsilon": cfg.epsilon, "derived_features": {n: DERIVED[n].description for n in e.inputs}}  # fmt: skip
    e0 = cfg.exploratory_years[0]
    cutoff = data.cutoffs[e0]
    wf = dev["wfs"][(e.id, sel["C"])]
    yr, rel = data.year[wf.rows], data.release[wf.rows]
    m = (yr >= e0 - cfg.calibration_window) & (yr < e0) & (rel < _days(cutoff))
    fit = fit_spec_at(data, cfg, spec, cutoff, wf.rows[m], wf.p[m])
    frozen = {
        "status": "frozen before any exploratory-period row was scored",
        "model_id": f"{e.id}+{sel['calibration']}",
        "model_version": cfg.full_version,
        "specification": spec,
        "fit_for_first_exploratory_year": {
            "year": e0,
            **fit,
            "calibration_rows": int(m.sum()),
            "calibration_years": [e0 - cfg.calibration_window, e0 - 1],
        },  # fmt: skip
        "selection": sel,
        "configuration_hash": cfg.configuration_hash(),
        **provenance,
    }
    frozen["frozen_hash"] = canonical_hash(frozen)
    return frozen


def verify_frozen(frozen: dict) -> None:
    body = {k: v for k, v in frozen.items() if k != "frozen_hash"}
    if canonical_hash(json.loads(json.dumps(body, default=_json_default))) != frozen["frozen_hash"]:
        raise AssertionError("frozen model specification was modified after freezing")


# ------------------------------------------------------------------------------------------------ exploratory


def exploratory(data: Data, cfg: ModelingConfig, frozen: dict, dev: dict) -> dict:
    """Frozen settings, walk-forward annual refits for every experiment (selected one + all others for context)."""
    verify_frozen(frozen)
    e0, e1 = cfg.exploratory_years
    years = [y for y in range(cfg.first_prediction_year, e1 + 1) if y in data.cutoffs]
    exp_years = [y for y in years if y >= e0]
    out, n_fits = {}, 0
    for exp in cfg.experiments:
        c = dev["chosen_c"][exp.id]  # frozen per-experiment C from development
        wf = walk_forward(data, cfg, exp.id, exp.inputs, c, years)
        n_fits += len(years)
        cal = {}
        for method in cfg.calibration_methods:
            cal[method] = calibrate(wf, data, cfg, exp_years, method)
        out[exp.id] = {"wf": wf, "cal": cal}
    sel = frozen["specification"]
    # Sanity: the walk-forward fit for e0 must reproduce the frozen coefficients exactly.
    f0 = out[sel["experiment"]]["wf"].fits[e0]
    frozen_coef = np.array([frozen["fit_for_first_exploratory_year"]["coef_scaled"][c] for c in sel["inputs"]])
    if not np.allclose(f0.coef, frozen_coef, rtol=0, atol=1e-12):
        raise AssertionError("exploratory refit does not reproduce the frozen model")
    # Static variant: the frozen 2010 coefficients and calibrator applied to all exploratory years without refits.
    wf = out[sel["experiment"]]["wf"]
    rows = np.flatnonzero(data.year >= e0)
    x = f0.scaler.transform(data.matrix(sel["inputs"], data.year >= e0))
    calib = frozen["fit_for_first_exploratory_year"]["calibrator"]
    static_cal = Calibrator(calib["method"], cfg.epsilon, calib["rows"], calib["slope"], tuple(calib["knots_x"]),
                            tuple(calib["knots_y"]))  # fmt: skip
    static = static_cal.apply(predict(f0.coef, x))
    del wf
    return {"by_exp": out, "years": exp_years, "static_rows": rows, "static_p": static, "fits": n_fits}


# ------------------------------------------------------------------------------------------------- production


def production(data: Data, cfg: ModelingConfig, frozen: dict, expl: dict) -> dict:
    """Frozen specification refitted on every released label, for prospective predictions."""
    verify_frozen(frozen)
    spec = frozen["specification"]
    last_release = date(1970, 1, 1) + timedelta(days=int(data.release.max()))
    cutoff = last_release + timedelta(days=7)  # first batch at which every label in the data is known
    wf = expl["by_exp"][spec["experiment"]]["wf"]
    rel = data.release[wf.rows]
    window_start = cutoff - timedelta(weeks=52 * cfg.calibration_window)
    m = (data.batch[wf.rows] >= _days(window_start)) & (rel < _days(cutoff))
    fit = fit_spec_at(data, cfg, spec, cutoff, wf.rows[m], wf.p[m])
    return {
        "model_id": frozen["model_id"],
        "model_version": cfg.full_version,
        "frozen_hash": frozen["frozen_hash"],
        "specification": spec,
        "fit": {**fit, "calibration_rows": int(m.sum()), "calibration_batches": [window_start, cutoff]},  # fmt: skip
        "valid_for_prediction_batches_from": cutoff,
    }

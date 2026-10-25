"""Phase 3B reports (deterministic: no wall-clock values, runtimes live in data/modeling/performance.json)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from tennis_data.modeling.artifacts import dumps
from tennis_data.modeling.dataset import DERIVED
from tennis_data.modeling.evaluation import verdict

LICENSE = "Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data."


def f(x, d=5) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{d}f}"


def s(x, d=5) -> str:
    return "–" if x is None else f"{x:+.{d}f}"


def ci(t, d=5) -> str:
    return f"[{t[0]:+.{d}f}, {t[1]:+.{d}f}]"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _bonf_key(r) -> str:
    return f"ll_ci{100 * (1 - r['bonferroni_alpha']):g}"


def _comparison_rows(res, block, names, ref="M0"):
    rows = []
    bkey = _bonf_key(res)
    for n in names:
        m = block["metrics"][n]
        b = block["boot"].get(f"{n} vs {ref}")
        rows.append([
            n, len(res["cfg"].experiment(n.split("+")[0]).inputs) if n not in ("M0",) else 0,
            f(m["log_loss"]), f(m["brier"]), f(m["accuracy"], 4), f(m["ece"], 4),
            s(b["log_loss_diff"]) if b else "–", ci(b["ll_ci95"]) if b else "–",
            ci(b[bkey]) if b and bkey in b else "–", s(b["brier_diff"]) if b else "–",
            ci(b["brier_ci95"]) if b else "–", verdict(*b["ll_ci95"]) if b else "baseline",
        ])  # fmt: skip
    return rows


COMP_HEADER = ["model", "inputs", "log loss", "Brier", "accuracy", "ECE", "ΔLL vs M0", "95% CI",
               "Bonferroni CI", "ΔBrier", "95% CI (Brier)", "ΔLL verdict (95%)"]  # fmt: skip


def _boot_rows(d: dict, keys=None) -> list[list]:
    rows = []
    for k, b in d.items():
        if keys and k not in keys:
            continue
        rows.append([k, b["rows"], b["clusters"], s(b["log_loss_diff"]), ci(b["ll_ci95"]), s(b["brier_diff"]),
                     ci(b["brier_ci95"]), f(b["share_ll_below_zero"], 3), verdict(*b["ll_ci95"])])  # fmt: skip
    return rows


BOOT_HEADER = ["comparison (new vs reference)", "rows", "blocks", "ΔLL", "95% CI", "ΔBrier", "95% CI",
               "P*(ΔLL<0)", "verdict"]  # fmt: skip


def _subgroup_rows(df: pl.DataFrame, final_id: str) -> list[list]:
    rows = []
    for r in df.iter_rows(named=True):
        rows.append([r["group"], r["level"], r["n"], f(r["ll_M0"]), f(r["ll_M0R"]), f(r[f"ll_{final_id}"]),
                     s(r[f"d_{final_id}"]), ci((r[f"lo_{final_id}"], r[f"hi_{final_id}"])), s(r["d_M0R"]),
                     ci((r["lo_M0R"], r["hi_M0R"]))])  # fmt: skip
    return rows


def _subgroup_header(final_id):
    return ["group", "level", "n", "LL M0", "LL M0R", f"LL {final_id}", f"Δ {final_id}", "95% CI", "Δ M0R",
            "95% CI"]  # fmt: skip


def _method_note(res) -> str:
    cfg = res["cfg"]
    return (
        f"Uncertainty: paired cluster bootstrap of ΔLL = LL(candidate) − LL(M0) on identical matches; "
        f"{cfg.bootstrap_samples} resamples of whole weekly prediction batches (seed {cfg.bootstrap_seed}), "
        f"percentile intervals. Negative Δ favours the candidate. "
        f"'Bonferroni CI' uses α = 0.05/{res['n_comparisons']} for the {res['n_comparisons']} registered "
        f"experiments compared with M0. An interval that includes 0 is reported as inconclusive."
    )


def development_report(res) -> str:
    cfg, dv, sel, fid = res["cfg"], res["devres"], res["sel"], res["final_id"]
    d0, d1 = cfg.development_years
    sp = res["splits"]
    lines = [
        "# Phase 3B development report: does serve/return information add to surface Elo?",
        "",
        LICENSE,
        "",
        f"Configuration hash `{cfg.configuration_hash()}`, code hash `{res['provenance']['code_hash']}`, model "
        f"version `{cfg.full_version}`, dataset key `{res['provenance']['dataset_input_key']}`, upstream feature "
        f"hash `{res['provenance']['upstream_feature_hash']}`.",
        "",
        "## 1. Question and answer",
        "",
        f"On the development years {d0}–{d1} ({dv['rows']:,} matches, {dv['weeks']} weekly batches) the "
        f"pre-specified rule selected **{fid}** "
        f"({cfg.experiment(sel['selected']).description}). Against the Phase 2 surface-Elo baseline (M0) its "
        f"development log-loss difference is {s(dv['boot'][f'{fid} vs M0']['log_loss_diff'])} "
        f"(95% CI {ci(dv['boot'][f'{fid} vs M0']['ll_ci95'])}).",
        "",
        "What carries the gain matters as much as the size (see the ablation report): the overall career-long "
        "shrunk serve/return rates (M1) do **not** improve on Elo; the 52-week recency-weighted rates and the "
        "recent-activity (effective sample size) features do. Opponent-adjusted and matchup features (M3) give a "
        "smaller, also non-zero improvement.",
        "",
        gain_source(res, "devres"),
        "",
        "**This is not an independent test.** The development years were also used to tune the Phase 2 Elo "
        "parameters (1991–2009 validation), so M0 is, if anything, favoured here. The 2010–2026 period was already "
        "inspected in Phase 2 and is reported separately as exploratory (`exploratory_report.md`).",
        "",
        "## 2. Protocol",
        "",
        "- **Models.** L2-penalised logistic regression without intercept on antisymmetric A−B inputs, objective "
        "Σ logloss + |β|²/(2C). M0 is the stored Phase 2 surface-Elo probability (not refitted, not retuned). "
        "M0R refits a single slope on the Elo logit z = log(p/(1−p)), p clipped to "
        f"[{cfg.epsilon:g}, 1 − {cfg.epsilon:g}]: it isolates what pure rescaling of Elo does.",
        "- **Player-order symmetry by construction.** Every input changes sign when A and B are swapped, there "
        "is no intercept, scaling divides by the training-fold RMS (no centring), and missing differences are "
        "0. Hence P(A beats B) = 1 − P(B beats A); checked numerically on every fit "
        f"(max deviation {max(r['max_symmetry_dev'] for r in res['dev']['grid'].to_dicts()):.1e}).",
        f"- **Walk-forward annual refits.** The model for year y is trained on rows with prediction_batch ≥ "
        f"{cfg.train_start} and **release_batch < first prediction batch of y** (a result is known only after its "
        "batch closes; Slams, 96+ draw Masters and the Olympics release a week later). The tournament start date "
        "is never used as evidence that a result was known. Per-year cutoffs and counts are in Section 3 and in "
        "`data/modeling/manifest.json`.",
        f"- **Folds.** Development years {d0}–{d1} grouped as "
        + ", ".join(f"{a}–{b}" for a, b in cfg.folds)
        + f". Years {cfg.first_prediction_year}–{d0 - 1} are predicted only to provide earlier out-of-fold "
        "predictions for calibration.",
        f"- **Selection rule (fixed before results).** (1) per experiment, C ∈ {list(cfg.c_grid)} by pooled "
        f"development log loss (ties → smaller C); (2) among candidates {list(cfg.candidates)} the lowest pooled "
        f"development log loss, replaced by the candidate with the fewest inputs within {cfg.simplicity_tolerance} "
        "of it; (3) calibration method (raw < temperature < isotonic in simplicity) by the same rule.",
        "- **Missing values.** Missing differences are set to 0 (the only antisymmetric value); the counts are "
        "in Section 4. No row is dropped.",
        "- **Leakage audit.** For a random sample of "
        f"{res['audit']['sampled_rows']} rows the released service history of player A was recomputed from the "
        "Phase 1 table and the Phase 3A eligibility table alone (release_batch < prediction_batch) and matched "
        f"the features exactly (max abs diff observed points {res['audit']['max_abs_diff_service_points_observed']}, "
        f"decay rate {res['audit']['max_abs_diff_decay_rate']:.1e}; current match included "
        f"{res['audit']['current_match_in_history']} times).",
        "",
        "## 3. Train/validation boundaries",
        "",
        table(
            [
                "prediction year",
                "role",
                "training cutoff (first batch of year)",
                "last training release batch",
                "training rows",
                "predicted rows",
            ],  # fmt: skip
            [
                [
                    r["year"],
                    r["role"],
                    r["training_cutoff_batch"],
                    r["last_train_release_batch"],
                    f"{r['train_rows']:,}",
                    f"{r['predict_rows']:,}",
                ]
                for r in sp.filter(pl.col("year") <= d1).iter_rows(named=True)
            ],
        ),  # fmt: skip
        "",
        "## 4. Inputs and missing values",
        "",
        table(
            ["input", "definition", "filled with 0 (all rows, 1994–2026)"],
            [[n, DERIVED[n].description, _filled(res, n)] for n in DERIVED],
        ),  # fmt: skip
        "",
        "## 5. Baseline M0 per fold",
        "",
        table(
            ["fold", "rows", "weeks", "LL M0", "Brier M0", "ECE M0", "LL M0R", f"LL {fid}"],
            [
                [
                    r["fold"],
                    f"{r['rows']:,}",
                    r["weeks"],
                    f(r["ll_M0"]),
                    f(r["brier_M0"]),
                    f(r["ece_M0"], 4),
                    f(r["ll_M0R"]),
                    f(r[f"ll_{fid}"]),
                ]
                for r in dv["folds"].iter_rows(named=True)
            ],
        ),  # fmt: skip
        "",
        "## 6. Model comparison (pooled development years, identical matches)",
        "",
        table(
            COMP_HEADER,
            _comparison_rows(
                res,
                dv,
                ["M0", *[e.id for e in cfg.experiments]]
                + ([fid] if fid not in [e.id for e in cfg.experiments] else []),
            ),
        ),
        "",
        _method_note(res),
        "",
        "## 7. Per-fold differences against M0",
        "",
        table(
            ["fold", "rows", "Δ M0R", "95% CI", f"Δ {fid}", "95% CI"],
            [
                [
                    r["fold"],
                    f"{r['rows']:,}",
                    s(r["d_M0R"]),
                    ci((r["lo_M0R"], r["hi_M0R"])),
                    s(r[f"d_{fid}"]),
                    ci((r[f"lo_{fid}"], r[f"hi_{fid}"])),
                ]
                for r in dv["folds"].iter_rows(named=True)
            ],
        ),  # fmt: skip
        "",
        "## 8. Hyperparameter grid (every setting tried)",
        "",
        table(
            ["experiment", "C", "inputs", "dev LL", *[f"LL {a}–{b}" for a, b in cfg.folds], "converged", "chosen"],
            [
                [
                    r["experiment"],
                    r["C"],
                    r["n_inputs"],
                    f(r["dev_log_loss"], 6),
                    *[f(r[f"ll_{a}_{b}"], 6) for a, b in cfg.folds],
                    r["all_converged"],
                    "✓" if res["dev"]["chosen_c"][r["experiment"]] == r["C"] else "",
                ]
                for r in res["dev"]["grid"].sort("experiment", "C").iter_rows(named=True)
            ],
        ),  # fmt: skip
        "",
        "## 9. Selection outcome",
        "",
        table(
            ["candidate", "chosen C", "inputs", "pooled dev LL"],
            [
                [e, res["dev"]["chosen_c"][e], len(cfg.experiment(e).inputs), f(v, 6)]
                for e, v in sel["candidate_dev_log_loss"].items()
            ],
        ),  # fmt: skip
        "",
        f"Lowest development log loss: **{sel['lowest']}**. Selected after the simplicity rule: "
        f"**{sel['selected']}** (C = {sel['C']}). Calibration scores: "
        + ", ".join(f"{k} {f(v, 6)}" for k, v in sel["calibration_scores"].items())
        + f" → **{sel['calibration']}**. Frozen model hash `{res['frozen']['frozen_hash'][:16]}`.",
        "",
        "## 10. Dependence sensitivity (block choice)",
        "",
        table(BOOT_HEADER, _boot_rows(dv["sensitivity"])),
        "",
        "Blocks: `week` = prediction batch (primary); `tournament` = event × year; `season` = calendar year "
        "(very few blocks, wide intervals). Matches also share players across blocks; no block scheme removes that "
        "dependence, so intervals may be somewhat too narrow.",
        "",
        "## 11. Subgroups (development years)",
        "",
        table(_subgroup_header(fid), _subgroup_rows(dv["subgroups"], fid)),
        "",
        "Subgroup intervals are unadjusted and descriptive; with this many groups some will exclude 0 by chance.",
        "",
        f"## 12. Coefficients of {sel['selected']} by prediction year (original input units)",
        "",
        _coef_table(res, sel["selected"], "development"),
        "",
        "## 13. Evaluation-design caveats",
        "",
        "1. **M0 was tuned on these years.** Phase 2 chose the Elo K-factors and shrinkage on 1991–2009 log loss; "
        "the development years are inside that window, so M0 is optimistically evaluated here and the "
        "comparison is conservative for the candidates.",
        "2. **Phase 3A priors.** The shrinkage κ values were set in Phase 3A from 1991–2009 player totals "
        "(moments of serve/return rates, not match outcomes). This is mild information from within the period.",
        "3. **Post-hoc diagnostics.** `A4_DECAY` and `A4_ESS` were added to the registry after the development grid "
        "was first inspected, only to locate where the M4 gain comes from. They are never selection candidates.",
        "4. **Not independent.** 2010–2026 results were examined in Phase 2; they are exploratory evidence only.",
        "",
    ]
    return "\n".join(lines)


def gain_source(res, key: str) -> str:
    """Where the selected model's gain comes from: sparse vs substantial serve-statistics history."""
    fid, block = res["final_id"], res[key]
    sub = block["subgroups"].filter(pl.col("group") == "history").sort("level")
    total = block["boot"][f"{fid} vs M0"]["log_loss_diff"] * block["rows"]
    parts = []
    for r in sub.iter_rows(named=True):
        share = r[f"d_{fid}"] * r["n"] / total if total else float("nan")
        parts.append(
            f"{r['level']}: {r['n']:,} matches ({100 * r['n'] / block['rows']:.0f}%), ΔLL {s(r[f'd_{fid}'])} "
            f"{ci((r[f'lo_{fid}'], r[f'hi_{fid}']))}, {100 * share:.0f}% of the total log-loss reduction"
        )
    return (
        "**Where the gain comes from.** A large part of the improvement is a cold-start correction: when a player "
        "has little or no released serve/return history (often Davis Cup or newly arrived players, whom Elo "
        "starts at the initial rating) the recency inputs and their missingness indicator carry most information. "
        + "; ".join(parts)
        + ". The improvement on matches between players with substantial history is the more relevant test of "
        "serve/return skill information and is smaller."
    )


def _filled(res, n) -> str:
    ds = res["dataset"]
    col = f"_filled_{n}"
    if col not in ds.columns:
        return "never null"
    sub = ds.filter(pl.col("prediction_batch").dt.year() >= res["cfg"].first_prediction_year)
    k = int(sub[col].sum())
    return f"{k:,} ({100 * k / sub.height:.2f}%)"


def _coef_table(res, exp_id, phase) -> str:
    c = res["coefs"].filter((pl.col("experiment") == exp_id) & (pl.col("phase") == phase)
                            & (pl.col("C") == res["dev"]["chosen_c"][exp_id]))  # fmt: skip
    if phase == "development":
        c = c.filter(pl.col("year") >= res["cfg"].development_years[0])
    feats = list(res["cfg"].experiment(exp_id).inputs)
    wide = c.pivot(on="feature", index=["year", "train_rows"], values="coef_original_units").sort("year")
    return table(
        ["year", "training rows", *feats],
        [[r["year"], f"{r['train_rows']:,}", *[f(r[x], 4) for x in feats]] for r in wide.iter_rows(named=True)],
    )


def exploratory_report(res) -> str:
    cfg, ex, fid = res["cfg"], res["exres"], res["final_id"]
    e0, e1 = cfg.exploratory_years
    fr = res["frozen"]
    static = f"{fid} (static 2010 fit)"
    primary = ["M0", "M0R", fid, static]
    others = [e.id for e in cfg.experiments if e.id not in ("M0R", fid.split("+")[0])]
    sp = res["splits"].filter(pl.col("year") >= e0)
    lines = [
        "# Phase 3B exploratory report: frozen model on 2010–2026",
        "",
        LICENSE,
        "",
        "**Read this first.** 2010–2026 is *not* an untouched holdout: Phase 2 reported surface-Elo results on it. "
        "Everything below was computed only after the development selection was frozen "
        f"(`data/modeling/models/frozen_model.json`, hash `{fr['frozen_hash'][:16]}`; the code verifies the hash "
        "and that the file did not change during this evaluation). No parameter, feature, C or calibration method "
        "was changed after these numbers were seen. They are exploratory, not a final test.",
        "",
        "## 1. Frozen specification",
        "",
        f"- Model `{fid}`: inputs {', '.join(f'`{x}`' for x in fr['specification']['inputs'])}; C = "
        f"{fr['specification']['C']}; calibration `{fr['specification']['calibration']}`.",
        "- Refitted every year with the frozen specification (coefficients re-estimated on the labels released "
        f"before the first batch of each year). `{static}` instead applies the {e0}-cutoff coefficients to every "
        "exploratory year without refitting.",
        f"- {ex['rows']:,} matches in {ex['weeks']} weekly batches ({e0}–{e1}; the source ends in "
        f"{res['dataset']['prediction_batch'].max()}).",
        "",
        gain_source(res, "exres"),
        "",
        "## 2. Primary comparison",
        "",
        table(COMP_HEADER, _comparison_rows(res, ex, primary)),
        "",
        _method_note(res),
        "",
        "## 3. Other registered experiments (frozen per-experiment C; reported for completeness, not used for any "
        "decision)",
        "",
        table(COMP_HEADER, _comparison_rows(res, ex, others)),
        "",
        "## 4. Sub-periods",
        "",
        table(
            ["period", "rows", "LL M0", "LL M0R", f"LL {fid}", f"Δ {fid}", "95% CI", "verdict"],
            [
                [
                    r["period"],
                    f"{r['rows']:,}",
                    f(r["ll_M0"]),
                    f(r["ll_M0R"]),
                    f(r[f"ll_{fid}"]),
                    s(r["d"]),
                    ci((r["lo"], r["hi"])),
                    verdict(r["lo"], r["hi"]),
                ]
                for r in ex["periods"].iter_rows(named=True)
            ],
        ),
        "",
        "## 5. Dependence sensitivity",
        "",
        table(BOOT_HEADER, _boot_rows(ex["sensitivity"])),
        "",
        "## 6. Subgroups",
        "",
        table(_subgroup_header(fid), _subgroup_rows(ex["subgroups"], fid)),
        "",
        "## 7. Walk-forward training cutoffs",
        "",
        table(
            ["prediction year", "training cutoff", "last training release batch", "training rows", "predicted rows"],
            [
                [
                    r["year"],
                    r["training_cutoff_batch"],
                    r["last_train_release_batch"],
                    f"{r['train_rows']:,}",
                    f"{r['predict_rows']:,}",
                ]
                for r in sp.iter_rows(named=True)
            ],
        ),  # fmt: skip
        "",
        f"## 8. Coefficients of {fid.split('+')[0]} by year (original units)",
        "",
        _coef_table(res, fid.split("+")[0], "exploratory"),
        "",
        "## 9. Prospective evaluation protocol (matches after 2026-10-25)",
        "",
        prospective_protocol(res),
        "",
    ]
    return "\n".join(lines)


def prospective_protocol(res) -> str:
    p = res["prod"]
    return "\n".join([
        "The only genuinely untouched evaluation is prospective. The protocol is fixed now:",
        "",
        f"1. **Model.** `{p['model_id']}` exactly as frozen (hash `{p['frozen_hash'][:16]}`): same inputs, C, "
        "calibration method, epsilon and walk-forward rule. Upstream Elo and feature parameters stay as merged.",
        f"2. **Production fit.** `data/modeling/models/production_model.json`: trained on every label released by "
        f"the end of the data (training cutoff batch {p['fit']['training_cutoff_batch']}, "
        f"{p['fit']['train_rows']:,} rows). Valid from prediction batch {p['valid_for_prediction_batches_from']}.",
        "3. **Population.** Every Phase 2 target match (same pool, statuses and eligibility rules) whose prediction "
        "batch is after 2026-10-25, i.e. from batch 2026-10-31 on.",
        "4. **Timing.** Predictions for a batch are produced from Elo and feature state frozen at batch start and "
        "stored (with model id, version, training cutoff, configuration hash) *before* any result of that batch is "
        "ingested. Refits follow the same annual rule (first batch of each calendar year), using only labels "
        "released before it. Labels released between the end of the current data and 2026-10-31 may be ingested "
        "through the normal Phase 1→3A pipeline before the first prospective batch; they are training data, not "
        "evaluation data.",
        "5. **Primary endpoint.** ΔLL = LL(model) − LL(M0) on identical matches, weekly-block bootstrap, 95% "
        "interval, reported with N and the number of blocks, whatever the sign. Secondary: Brier, ECE, surfaces, "
        "Grand Slams vs other events.",
        "6. **Analysis plan.** First analysis after at least 26 weekly batches, final after 52; no interim decisions "
        "about the model. Any change to the model starts a new, separately registered prospective period.",
    ])  # fmt: skip


def calibration_report(res) -> str:
    cfg, sel = res["cfg"], res["sel"]
    sid = sel["selected"]
    dv, ex = res["devres"], res["exres"]
    names = ["M0", "M0R", "M0R+temperature", "M0R+isotonic", sid, f"{sid}+temperature", f"{sid}+isotonic"]

    def mrows(block):
        return [[n, f(block["metrics"][n]["log_loss"]), f(block["metrics"][n]["brier"]),
                 f(block["metrics"][n]["ece"], 4)] for n in names if n in block["metrics"]]  # fmt: skip

    slopes = []
    for y, cal in sorted(res["dev"]["calibrators"][(sid, "temperature")].items()):
        slopes.append([y, "development", cal.rows, f(cal.slope, 4)])
    lines = [
        "# Phase 3B calibration report",
        "",
        LICENSE,
        "",
        "## 1. Methods and data separation",
        "",
        "Both calibrators are symmetric, g(1 − p) = 1 − g(p), so they cannot break player-order symmetry:",
        "",
        "- `temperature`: σ(b · logit p), Platt scaling without an intercept (an intercept would favour whichever "
        "player is labelled A).",
        "- `isotonic`: pool-adjacent-violators on the symmetrised sample {(p, y)} ∪ {(1 − p, 1 − y)} with linear "
        f"interpolation; requires ≥ {cfg.isotonic_min_rows} rows.",
        "",
        table(
            ["item", "rows used"],
            [
                ["base model for year y", "trained on labels released before the first batch of y"],
                [
                    "calibrator for year y",
                    f"the same model's out-of-fold predictions for years y−{cfg.calibration_window}"
                    "…y−1 whose label was released before the first batch of y",
                ],
                ["evaluated", "year y only; never rows used to fit the calibrator or the model"],
            ],
        ),  # fmt: skip
        "",
        "The release filter matters for multi-week events that straddle New Year: their results are predicted in "
        "year y−1 but known only in y.",
        "",
        "## 2. Development years",
        "",
        table(["model", "log loss", "Brier", "ECE"], mrows(dv)),
        "",
        f"Selection: {', '.join(f'{k} {f(v, 6)}' for k, v in sel['calibration_scores'].items())} → "
        f"**{sel['calibration']}** (simplest within {cfg.simplicity_tolerance}).",
        "",
        "## 3. Exploratory years (frozen method)",
        "",
        table(["model", "log loss", "Brier", "ECE"], mrows(ex)),
        "",
        "## 4. Temperature slope fitted for each development year",
        "",
        table(["year", "phase", "calibration rows", "slope b"], slopes),
        "",
        "A slope below 1 means the raw model was over-confident on the preceding years.",
        "",
        "## 5. Reliability tables (10 equal-width bins of P(A wins))",
        "",
        "### Development",
        "",
        _cal_table(dv["calibration"], ["M0", "M0R", sid, f"{sid}+temperature"]),
        "",
        "### Exploratory",
        "",
        _cal_table(ex["calibration"], ["M0", "M0R", sid, f"{sid}+temperature"]),
        "",
        "ECE = Σ_bins n_bin |observed − mean predicted| / n. Because player A is chosen by a hash of the match id, "
        "the table is roughly symmetric around 0.5.",
        "",
    ]
    return "\n".join(lines)


def _cal_table(df: pl.DataFrame, models) -> str:
    rows = []
    for r in df.filter(pl.col("model").is_in(models)).sort("model", "bin").iter_rows(named=True):
        rows.append([r["model"], f"{r['bin_lower']:.1f}–{r['bin_upper']:.1f}", f"{r['n']:,}", f(r["mean_predicted"], 4),
                     f(r["observed"], 4), s(r["observed_minus_predicted"], 4)])  # fmt: skip
    return table(["model", "bin", "n", "mean predicted", "observed", "obs − pred"], rows)


ABLATION_NOTES = {
    "M0R vs M0": "Does refitting a slope on the Elo logit help?",
    "M1 vs M0R": "Overall shrunk serve + return rates on top of rescaled Elo",
    "A1_SPW vs M0R": "Serve rate only",
    "A1_RPW vs M0R": "Return rate only",
    "M2 vs M1": "Adding surface-specific rates",
    "A2_SURF vs M0R": "Surface-specific rates only",
    "M3 vs M0R": "Opponent-adjusted strengths + matchup",
    "M3 vs A3_NOMATCHUP": "Contribution of the matchup feature",
    "A3_NOMATCHUP vs M0R": "Opponent-adjusted strengths without matchup",
    "A3_MATCHUP vs M0R": "Matchup expected serve-point edge only",
    "M4A vs M1": "Adding 52-week recency-weighted rates",
    "M4B vs M1": "Adding reliability/activity features",
    "A4_DECAY vs M0R": "Post-hoc: recency-weighted rates only",
    "A4_ESS vs M0R": "Post-hoc: recent activity (log ESS) only",
    "M4A vs A4_DECAY": "Do career shrunk rates add anything to recency-weighted ones?",
    "A0_NOELO vs M0": "Serve/return rates without Elo, against Elo",
}


def ablation_report(res) -> str:
    dv, ex = res["devres"]["ablations"], res["exres"]["ablations"]

    def rows(d):
        return [[ABLATION_NOTES.get(k, ""), *r] for k, r in zip(d, _boot_rows(d), strict=True)]

    lines = [
        "# Phase 3B ablation report",
        "",
        LICENSE,
        "",
        "Each line compares two registered experiments on identical matches (ΔLL = new − reference, weekly-block "
        "bootstrap, 95% percentile interval, unadjusted). Each experiment uses its own C chosen on development "
        "data. Exploratory rows are shown for transparency only; no decision used them.",
        "",
        "## 1. Development years",
        "",
        table(["question", *BOOT_HEADER], rows(dv)),
        "",
        "## 2. Exploratory years (2010–2026, not used for any decision)",
        "",
        table(["question", *BOOT_HEADER], rows(ex)),
        "",
        "## 3. Reading",
        "",
        _ablation_reading(res),
        "",
    ]
    return "\n".join(lines)


def _ablation_reading(res) -> str:
    dv = res["devres"]["ablations"]

    def v(k):
        b = dv.get(k)
        return f"{s(b['log_loss_diff'])} {ci(b['ll_ci95'])}" if b else "n/a"

    return "\n".join([
        f"- **Career shrunk rates add nothing to Elo.** M1 vs M0R: {v('M1 vs M0R')}; serve-only and return-only "
        f"differences are negligible either way ({v('A1_SPW vs M0R')}, {v('A1_RPW vs M0R')}). Long-run serve/return levels are largely "
        "already in Elo.",
        f"- **Surface-specific rates:** {v('A2_SURF vs M0R')} alone; adding them to M1: {v('M2 vs M1')}.",
        f"- **Opponent adjustment and matchup:** M3 vs M0R {v('M3 vs M0R')}; the matchup feature alone carries "
        f"most of it ({v('A3_MATCHUP vs M0R')}; M3 vs A3_NOMATCHUP {v('M3 vs A3_NOMATCHUP')}).",
        f"- **Recency is where the information is.** M4A vs M1 {v('M4A vs M1')}; recency-weighted rates alone "
        f"(post-hoc A4_DECAY) {v('A4_DECAY vs M0R')}, and career rates add nothing beyond them "
        f"({v('M4A vs A4_DECAY')}). Recent activity alone (A4_ESS, post-hoc) {v('A4_ESS vs M0R')}: part of the "
        "recency signal is how much a player has played lately (Elo has no notion of inactivity or form).",
        f"- **Without Elo** the serve/return rates are much worse than Elo ({v('A0_NOELO vs M0')}).",
        "- These are development-period statements with unadjusted intervals; A4_* were defined after the grid was "
        "first seen.",
    ])  # fmt: skip


def model_card(res) -> str:
    cfg, fr, p, fid = res["cfg"], res["frozen"], res["prod"], res["final_id"]
    dv, ex = res["devres"], res["exres"]
    bd, be = dv["boot"][f"{fid} vs M0"], ex["boot"][f"{fid} vs M0"]
    coefs = p["fit"]["coef_original_units"]
    return "\n".join([
        f"# Model card: {fid} (Phase 3B, `{cfg.full_version}`)",
        "",
        LICENSE,
        "",
        "## Purpose",
        "",
        "Pre-match probability that player A wins an ATP tour-level singles match, for research. Not for betting or "
        "any decision with financial consequences.",
        "",
        "## Model",
        "",
        f"- Logistic regression without intercept, L2 (C = {fr['specification']['C']}), calibration "
        f"`{fr['specification']['calibration']}`.",
        "- Inputs (A − B differences): " + "; ".join(f"`{k}` — {DERIVED[k].description}" for k in coefs),
        "- Production coefficients (original units): " + ", ".join(f"`{k}` {v:+.4f}" for k, v in coefs.items()),
        f"- Production fit: labels released before batch {p['fit']['training_cutoff_batch']} "
        f"({p['fit']['train_rows']:,} matches since {cfg.train_start}); frozen hash `{fr['frozen_hash'][:16]}`; "
        f"configuration hash `{cfg.configuration_hash()}`.",
        "- Symmetric in player order by construction (antisymmetric inputs, no intercept, symmetric calibration).",
        "",
        "## Performance (log loss; Δ vs Phase 2 surface Elo on identical matches, weekly-block bootstrap 95% CI)",
        "",
        table(["period", "status", "matches", "LL model", "LL surface Elo", "ΔLL", "95% CI", "ΔBrier"], [
            [f"{cfg.development_years[0]}–{cfg.development_years[1]}", "development (selection)", f"{bd['rows']:,}",
             f(dv["metrics"][fid]["log_loss"]), f(dv["metrics"]["M0"]["log_loss"]), s(bd["log_loss_diff"]),
             ci(bd["ll_ci95"]), s(bd["brier_diff"])],
            [f"{cfg.exploratory_years[0]}–{cfg.exploratory_years[1]}", "exploratory (seen in Phase 2)",
             f"{be['rows']:,}", f(ex["metrics"][fid]["log_loss"]), f(ex["metrics"]["M0"]["log_loss"]),
             s(be["log_loss_diff"]), ci(be["ll_ci95"]), s(be["brier_diff"])],
        ]),  # fmt: skip
        "",
        "Neither period is an independent test. The prospective protocol is in `exploratory_report.md` §9.",
        "",
        "Development: " + gain_source(res, "devres"),
        "",
        "Exploratory: " + gain_source(res, "exres"),
        "",
        "## Limitations",
        "",
        "- No match dates in the source: chronology is weekly; recency is quantised to weeks.",
        "- Development period overlaps the Phase 2 Elo tuning period; exploratory period was seen in Phase 2.",
        "- Block bootstrap ignores dependence through shared players across weeks; intervals may be too narrow.",
        "- Tour-level ATP men only, 1991 onward for statistics; no qualifying/challenger matches.",
        "- Missing feature differences are 0 (neutral); players without recent statistics are treated as average "
        "relative to their opponent on that input, with a missingness indicator for the recency rates.",
        "- Data license CC BY-NC-SA 4.0 (non-commercial).",
        "",
    ])  # fmt: skip


def summary_json(res) -> dict:
    cfg, fid = res["cfg"], res["final_id"]

    def comp(block):
        return {k: {kk: vv for kk, vv in b.items()} for k, b in block["boot"].items()}

    return {
        "model_version": cfg.full_version,
        "configuration_hash": cfg.configuration_hash(),
        "code_hash": res["provenance"]["code_hash"],
        "dataset_input_key": res["provenance"]["dataset_input_key"],
        "frozen_hash": res["frozen"]["frozen_hash"],
        "selection": res["sel"],
        "selected_model": fid,
        "development": {"rows": res["devres"]["rows"], "weeks": res["devres"]["weeks"],
                        "metrics": res["devres"]["metrics"], "comparisons_vs_M0": comp(res["devres"])},
        "exploratory": {"rows": res["exres"]["rows"], "weeks": res["exres"]["weeks"],
                        "metrics": res["exres"]["metrics"], "comparisons_vs_M0": comp(res["exres"])},
        "leakage_audit": res["audit"],
        "production": {"training_cutoff_batch": res["prod"]["fit"]["training_cutoff_batch"],
                       "train_rows": res["prod"]["fit"]["train_rows"],
                       "coef_original_units": res["prod"]["fit"]["coef_original_units"]},
        "logistic_fits": res["manifest"]["fits"],
    }  # fmt: skip


def write_reports(res) -> dict[str, Path]:
    out = res["cfg"].reports_dir
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    docs = {
        "development_report.md": development_report(res),
        "exploratory_report.md": exploratory_report(res),
        "calibration_report.md": calibration_report(res),
        "ablation_report.md": ablation_report(res),
        "model_card.md": model_card(res),
    }
    written = {}
    for name, text in docs.items():
        p = out / name
        p.write_text(text)
        written[name] = p
    p = out / "modeling_summary.json"
    p.write_text(dumps(summary_json(res)))
    written["modeling_summary.json"] = p
    csvs = {
        "development_grid.csv": res["dev"]["grid"].sort("experiment", "C"),
        "development_folds.csv": res["devres"]["folds"],
        "development_subgroups.csv": res["devres"]["subgroups"],
        "exploratory_subgroups.csv": res["exres"]["subgroups"],
        "exploratory_periods.csv": res["exres"]["periods"],
        "calibration_development.csv": res["devres"]["calibration"],
        "calibration_exploratory.csv": res["exres"]["calibration"],
        "splits.csv": res["splits"],
    }
    for name, df in csvs.items():
        p = tables / name
        df.write_csv(p, float_precision=8)
        written[f"tables/{name}"] = p
    return written

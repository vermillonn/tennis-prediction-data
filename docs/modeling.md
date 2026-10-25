# Phase 3B: predictive modeling, calibration and backtesting

Question: do the Phase 3A serve/return features add predictive information beyond the Phase 2 surface Elo
(M0)? A credible "no" was an acceptable outcome. Results: `reports/modeling/` (start with
`development_report.md`).

```bash
uv run tennis-data model-backtest            # dataset -> development -> select -> freeze -> exploratory -> reports
uv run tennis-data query "SELECT model_id, count(*) FROM model_selected_predictions GROUP BY ALL"
```

## 1. Dataset (`modeling/dataset.py`)

- 1:1 join of `data/elo/predictions/model_name=surface_elo` and `data/features/serve_return` on `match_id`.
  The join is rejected (`DatasetError`) on duplicate ids, different row counts, or any disagreement in player
  A/B, prediction batch, tournament date, level or surface. The label is checked against the Phase 1 validated
  table. Both upstream outputs must use the same chronology settings.
- No row is dropped. Every model input is a registered **derived antisymmetric feature** (`DERIVED`): swapping
  A and B negates it. Missing differences are 0, the only value consistent with antisymmetry. For the
  recency-weighted rates a missingness-indicator difference is also available.
- Each derived feature lists the upstream columns it reads. Those columns must pass the Phase 3A guard
  (`features.model_inputs.check_model_inputs`), so labels, current-match statistics, `stats_eligible`,
  `stats_exclusion_reason`, release batches, ids and version fields can never become inputs. Tests check this
  guard and a perturbation of `stats_eligible`.
- The joined table is cached in `data/modeling/dataset.parquet`. It is reused only when the SHA-256 of every
  upstream file and the modeling engine version are unchanged.
- `elo_logit = log(p / (1 - p))` with p clipped to [1e-6, 1 - 1e-6].

## 2. Chronology (`modeling/folds.py`)

- Walk-forward annual refits. The model for calendar year y predicts rows whose `prediction_batch` is in y. It
  is trained on rows with `prediction_batch >= 1991-01-01` and `release_batch < cutoff(y)`, where cutoff(y)
  is the first prediction batch of y. A result is known only when its batch closes, and multi-week events
  release a week later. The tournament start date is never treated as proof that a result was known.
- 1994–2001 are predicted only so that development years have earlier out-of-fold predictions for
  calibration. Development years are 2002–2009 (folds 2002–04, 2005–07, 2008–09). 2010–2026 is exploratory.
- Every cutoff, its last included release batch and the row counts are written to
  `data/modeling/manifest.json` and the reports.

## 3. Models (`modeling/preprocessing.py`, `modeling/logistic.py`)

- L2-penalised logistic regression **without intercept**, minimising `Σ logloss + |β|² / (2C)` by damped
  Newton steps with backtracking. Tests check it against `scipy.optimize`.
- Inputs are scaled by their training-fold RMS. They are not centred, because centring would break
  antisymmetry.
- These choices give `P(A beats B) = 1 − P(B beats A)` exactly, and every fit checks it numerically.
- Experiment registry: `config/modeling.toml` (M0R = Elo logit only; M1–M4B; ablations). `A4_*` are post-hoc
  diagnostics that were added after the development grid was first seen. They are never selection candidates.
- LightGBM was not fitted. The task made it optional and conditional on evidence of non-linearity, and no
  such diagnostic was run. Isotonic calibration did not beat raw or temperature-scaled output. That is weak
  evidence only: it tests the link of the combined score, not interactions between features. A non-linear
  learner is left for a later phase.

## 4. Calibration (`modeling/calibration.py`)

`temperature` (σ(b·logit p), Platt scaling without an intercept) and symmetrised `isotonic` both satisfy
`g(1 − p) = 1 − g(p)`. The calibrator for year y is fitted on the same model's out-of-fold predictions for
y−3…y−1, using only rows whose label was released before cutoff(y). It is never fitted on the rows it is
evaluated on. Tests check this by flipping the labels of the evaluated year.

## 5. Selection, freezing and evaluation (`modeling/experiments.py`, `modeling/evaluation.py`)

1. C ∈ {0.01, 0.1, 1, 10} per experiment by pooled development log loss (pooling weights each fold by its size).
2. The candidate with the lowest pooled development log loss, replaced by the candidate with the fewest inputs
   within 0.0005 of it.
3. The calibration method by the same rule (raw < temperature < isotonic).
4. The specification and its fit for the 2010 cutoff are written to `data/modeling/models/frozen_model.json`
   with a hash. The exploratory stage verifies the hash before it runs and checks that the file has not
   changed after it runs. A test scrambles every exploratory label and shows that selection and the frozen
   hash do not change.
5. Exploratory evaluation applies the frozen specification walk-forward to 2010–2026. A "static" variant
   applies the 2010 fit to every year without refitting.
6. The production fit uses every released label and goes to `models/production_model.json`. The prospective
   protocol for matches after 2026-10-25 is in `exploratory_report.md` §9.

Uncertainty: paired bootstrap of ΔLL = LL(candidate) − LL(M0) on identical matches. Whole weekly batches are
resampled (2000 resamples, seed 20261025). Sensitivity runs use tournament × year blocks and season blocks.
Bonferroni intervals cover the registered experiments. An interval that includes 0 is labelled inconclusive.

## 6. Leakage audit (`modeling/audit.py`)

For 300 random rows, the released service history of player A is recomputed from the Phase 1 table and the
Phase 3A eligibility table alone (`release_batch < prediction_batch`). The recomputed observed points and
52-week decayed rate must equal the feature values, and the current match must never appear. The pipeline
stops if the audit fails.

## 7. Outputs

| path | content |
|---|---|
| `data/modeling/dataset.parquet` | joined modeling table (cache; git-ignored) |
| `data/modeling/predictions/experiments.parquet` | every experiment at its chosen C, 1994–2026: raw, temperature, isotonic |
| `data/modeling/predictions/selected.parquet` | frozen selected model (required columns first) + static variant |
| `data/modeling/models/{frozen_model,production_model}.json` | specification, coefficients, scales, calibrator, cutoffs |
| `data/modeling/models/coefficients.parquet`, `development_grid.parquet` | every fit and every grid setting |
| `data/modeling/manifest.json` | hashes, splits, cutoffs, leakage audit, SHA-256 and size of every artifact |
| `data/modeling/performance.json` | runtime, peak memory, fit counts, git commit (not reproducible by design) |
| `reports/modeling/*.md`, `tables/*.csv`, `modeling_summary.json` | reports (byte-reproducible) |

DuckDB views: `model_predictions` and `model_selected_predictions`.

## 8. Evidence tiers

| tier | years | status | what it can support |
|---|---|---|---|
| development | 2002–2009 | used for every choice: C, candidate, calibration | the selection only; the Elo baseline (M0) was also tuned on 1991–2009, so it is favoured here |
| exploratory | 2010–2026 | scored once, after `frozen_model.json` was written and hashed; Phase 2 had already reported on it | consistency checks, never decisions; it is not a final or independent test |
| prospective | prediction batches after 2026-10-25 | not yet observed; protocol fixed in `exploratory_report.md` §9 | the only independent evaluation of the frozen model |

No parameter, input, C or calibration method was changed after the exploratory numbers were seen.

## 9. Selection decision: M4A versus M4B

The pre-specified rule took the candidate with the lowest pooled development log loss. That was M4B
(0.594852, 7 inputs). The rule then replaced it with the candidate with the fewest inputs within
`simplicity_tolerance = 0.0005`. That was M4A (0.595120, 6 inputs, difference 0.00027). M4A's calibration
followed the same rule. Temperature scaling (0.594663) was within 0.0005 of raw output, so the simpler `raw`
was kept. The tolerance and the simplicity order were in `config/modeling.toml` before any result was
computed.

M4B was worse in development, but in the exploratory period it is better than M4A (ΔLL vs M0 −0.00715 against
−0.00541). That observation **cannot** be used to switch models: doing so would select on the exploratory
period. A future phase may register M4B (or a combination) as a new candidate. It would then need a new
development protocol and a new prospective period. `tests/test_modeling_published.py` checks that the
published selection follows the rule and that the configuration and code have not changed since the reports
were generated.

## 10. Open research questions

These are documented limitations, not settled findings. Phase 3B made no correction for any of them.

1. **Grand Slam underperformance.** M4A is worse than M0 on Grand Slams in development (ΔLL +0.0037,
   95% CI [+0.0005, +0.0068]). In exploratory it is +0.0017 [−0.0001, +0.0035], inconclusive. M0R shows the
   same pattern. A single slope on the Elo logit is shared across best-of-three and best-of-five matches, while
   favourites win best-of-five more often. A format-specific slope or a format-aware Elo transform is untested.
2. **Elo cold start.** 81% (development) and 54% (exploratory) of M4A's log-loss reduction comes from matches
   where a player has fewer than 1,000 released service points. Many of these are Davis Cup matches or newly
   arrived players, whom Elo starts at the initial rating. Part of the gain is therefore a correction of Elo's
   initialisation, not information about serve/return skill. Between players with substantial history the
   gain is −0.0015 [−0.0026, −0.0004] (development) and −0.0030 [−0.0038, −0.0023] (exploratory). Whether a
   better Elo prior (for example, ranking-based initial ratings) would absorb most of the gain is open.
3. **Dependence in the bootstrap intervals.** The intervals resample whole weekly batches. Tournament × year
   and season blocks give similar intervals. No block scheme captures dependence through players who appear
   in many weeks, so the intervals may be too narrow. Two-way (week × player) or player-cluster methods are
   untested. Subgroup intervals are unadjusted. Only the comparisons with M0 have Bonferroni versions.

## 11. Known limitations

- The development years overlap the Phase 2 Elo tuning period (1991–2009), so M0 is optimistic there. 2010–2026
  was seen in Phase 2. Neither period is an independent test. Only the prospective protocol gives one.
- Much of the gain is a cold-start correction for players with little released statistics history. See the
  subgroup tables.
- The bootstrap blocks do not remove dependence through players shared across weeks.
- Grand Slam matches (best of five) are predicted slightly worse than by M0. The model has no format-specific
  slope.

# Phase 3A: historical serve/return features

Code: `src/tennis_data/features/`. Config: `config/features.toml`. Command: `uv run tennis-data features-build`.
Column definitions: `docs/feature_dictionary.md` (generated). Coverage: `reports/features/coverage_report.md`.

Phase 3A builds features only. It trains no model and makes no claim that the features improve match
prediction. That needs a separately specified evaluation (Phase 3B).

## 1. Inputs and populations

- **Rows:** one per Phase 2 Elo target match (validated, `completed`, in a rating pool): 193,634 rows that join
  1:1 to `elo_predictions` on `match_id`. The Phase 2 match population is reused unchanged
  (`elo.data.engine_frame`), as are its pools and its eligibility.
- **Statistics:** validated Phase 1 columns `player_{a,b}_svpt`, `_first_won`, `_second_won` (nullable Int64).
  Per match, for player A:
  - service points played = `a_svpt`; won = `a_1stWon + a_2ndWon`
  - return points played = `b_svpt`; won = `b_svpt - (b_1stWon + b_2ndWon)`

  Player B is symmetric. Missing values stay null, and nothing is set to 0 or to an average.
- **`stats_eligible`:** independent of Elo eligibility. It requires:
  - the match is in a rating pool, with valid identities;
  - status `completed` (plus `retired` if `include_retirements = true`);
  - `stats_status = complete`, and none of the Phase 2 stat warning flags (`service_games_imbalance`, ...);
  - both service-point denominators > 0, and points won ≤ points played.

  Every excluded match carries a `stats_exclusion_reason` (`data/features/match_stats_eligibility.parquet`).
  A match can update Elo without contributing statistics: 94,620 such matches exist, almost all with no
  statistics at all. Rows Phase 1 quarantined (including 15 with impossible statistics) never reach this pipeline.
- **Scope:** ATP only (the tour-level pool). There is no WTA data in the pipeline, so no WTA priors exist.
  Adding WTA requires its own pool and its own priors.

## 2. Chronology (identical to Phase 2)

Features use the Phase 2 batches unchanged (`prediction_batch`, `release_batch`; Saturday-to-Friday weeks;
two-week release for Slams, 96+ draw Masters and the Olympics). For each batch, in date order:

1. If due, refit the opponent model on observations with `release_batch` < batch.
2. Freeze: every target match of the batch gets snapshots of both players from the state left by earlier
   batches. Nothing from the batch itself is visible, including the match's own statistics.
3. Release: observations whose `release_batch` is this batch enter the state.

The source has no match dates. `tournament_date` is the event start, and no match timestamp is created.
Consequences:

- **Calendar windows are weekly:** `last13w` / `last52w` (≈ 90 / 365 days) include observations whose
  *event start batch* lies within the last 13 / 52 batches before the prediction batch. An observation enters a
  window when it is released and leaves it when its event start batch falls out of the range. Day-exact recency
  is not claimed.
- **Match-count windows** (`last5`, `last20`) use the most recent N released observations, ordered by release
  batch, then event start, then `round_order`, then `match_id`. Within one batch this order is a deterministic
  convention, not an observed sequence.
- **Decay ages** are measured in weeks from the observation's event start batch to the prediction batch. They are
  quantised to weeks, and for a two-week event they may overstate the true age by up to a week.

Order independence: rows are sorted by that canonical content key before processing. Point counts are integers
(exact), and decay sums add each batch's terms with `math.fsum`. Tests show bit-identical output under
shuffling, truncation at batch boundaries, perturbation of future statistics and future winners, and that a
player's own match statistics never influence their pre-match snapshot.

## 3. Estimators

All rates are **point-weighted** (sum won / sum played). No feature averages match percentages. Every rate
is exported with its denominator.

### Raw and windowed rates

`spw_raw`, `rpw_raw` (all prior history), `*_last5`, `*_last20`, `*_last13w`, `*_last52w`. With no points
the rate is null, never 0%.

### Recency weighting

`*_decay52w`: `sum_i w_i won_i / sum_i w_i played_i` with `w_i = 2^(-age_i / h)`, where h = 52 weeks and i
indexes matches. The half-life is not tuned. Several values can be exported (`[decay].half_life_weeks`) for
selection in Phase 3B.

Two different sample-size quantities are exported. They are not interchangeable:

| column | definition | meaning |
|---|---|---|
| `service_points_decay52w` | `sum_i w_i n_i` | total decay-weighted points: the denominator of the decayed rate. **Not** an effective sample size. Its scale depends on the arbitrary normalisation w = 1 at age 0. |
| `service_ess_decay52w` | `(sum_i w_i n_i)^2 / sum_i w_i^2 n_i` | Kish effective sample size, in points |
| `matches_ess_decay52w` | `(sum_i w_i)^2 / sum_i w_i^2` | Kish effective number of matches |

Return-point versions are analogous. The opponent model exports the same pair: `adjusted_serve_points_weighted`
(a weight sum) and `adjusted_serve_ess` (Kish), with return versions.

**What Kish ESS measures here.** If every point were an independent Bernoulli trial with the same variance,
the variance of the decayed rate would equal that of an unweighted rate over ESS points. ESS is unchanged when
all weights are rescaled, equals the raw count when all weights are equal, and never exceeds the raw count. The
decay weights are a deliberate down-weighting of old evidence, not sampling weights. So ESS describes how
concentrated the weights are, not how much information about *current* ability the data carry.

**Limitation: within-match dependence.** Points within a match are not independent. They share an opponent,
conditions, form and match situation. With an intra-match correlation ρ and m points per match, the
variance is inflated by the design effect `1 + (m - 1) ρ` (about 75 service points per match here), so
the point-level ESS (`service_ess_*`) is an **upper bound** on the information. The match-level ESS
(`matches_ess_*`) treats each match as one unit and is robust to within-match dependence, but ignores how
many points each match had. The true information lies in between. ρ is not estimated in Phase 3A. The same
caveat applies to the raw point counts and to the beta-binomial shrinkage, which treat points as exchangeable;
the κ values were estimated from player totals, which partly absorbs this.

### Bayesian shrinkage (beta-binomial)

`spw_shrunk = (won + kappa_s * prior_spw) / (played + kappa_s)`, and similarly for return.

- **Prior policy: chronological.** `prior_spw` is the pool's service-point win rate over all observations
  released before the prediction batch. It is never computed from the whole dataset. Until 20,000 service
  points have been released, the prior and the shrunk features are null. The return prior is
  `1 - prior_spw`, because every point is one player's serve and the other's return.
- **Prior strengths:** `kappa_service = 200`, `kappa_return = 300` (equivalent points). These are round values
  inside beta-binomial method-of-moments estimates computed only on 1991-2009 (validation-period) player
  totals: service 170-236, return 274-384, depending on the minimum-sample filter. They were not tuned against
  any prediction metric, and the 2010-2026 period was not used.
- **Diagnostics:** `*_prior_weight = kappa / (n + kappa)` and `*_prior_dominated` (weight > 0.5) flag estimates
  that mostly reflect the prior. `has_stats_history`, `stat_matches` and the point counts let downstream models
  tell estimated values from observed data.

### Surface estimates (heuristic hierarchical blend)

For the match surface s (Hard, Clay, Grass or Carpet; Carpet is a separate surface here):

1. `p_off` = the player's **off-surface** shrunk rate (all points minus surface points, toward `prior_spw`).
2. `base = sigmoid(logit(p_off) + logit(prior_s) - logit(prior_all))`: `p_off` moved to surface s by the
   population surface effect. `prior_s` is the chronological surface prior.
3. `surface_spw = (won_s + kappa_surf * base) / (n_s + kappa_surf)`, with `kappa_surface_service = 600` and
   `kappa_surface_return = 500` (validation-period method-of-moments: about 570 and 470).

Using off-surface evidence in step 1 means each point is counted exactly once. The surface points inform the
surface estimate, and the other points inform its prior. A player with few grass points therefore gets mostly
their transported off-surface ability, never a confident 0% or 100%. Without surface points, the estimate equals
`base`; with many, it converges to the raw surface rate. This is an empirical-Bayes style **heuristic**, not a
fitted hierarchical model. Surface sample sizes are exported separately (`surface_service_points`, ...).

### Opponent adjustment (Option A: regularised weighted binomial model)

```
logit P(server i wins a point against returner j on surface s) = mu_s + alpha_i + a_is - beta_j - b_js
```

- **Data and weights:** server-returner point counts from released observations. Each observation is weighted
  `2^(-age/104 weeks)` relative to the fit batch. Matches with a missing surface are left out of this model
  (they still count in the raw and shrunk features).
- **Regularisation:** ridge (zero-mean Gaussian priors) with λ = κ·p̄(1 − p̄). This makes κ an equivalent-points
  strength on the same footing as the beta-binomial κ: serve 200, return 300, surface deviations 550.
  The surface baseline mu_s is unpenalised.
- **Identifiability:** the ridge fixes the location. At the optimum, the active alphas, betas and per-surface
  deviations each average exactly zero; the real data give mean alpha = -7e-8. An unseen player has
  alpha = beta = 0, which is exactly the population baseline (`adjusted_in_fit = false`).
- **Objective:** `F = sum w [k softplus(-eta) + (n - k) softplus(eta)] + lam_serve/2 |alpha|^2 +
  lam_return/2 |beta|^2 + lam_surface/2 (|a|^2 + |b|^2)`, the weighted binomial negative log-likelihood plus
  the ridge terms (`opponent.penalized_objective`).
- **Fitting:** block coordinate Newton steps (diagonal per block), plus a closed-form re-centring along every
  direction in which the likelihood is exactly flat (global shifts against mu_s, and each player's overall
  strength against their surface deviations). Without re-centring, convergence took over 2,000 iterations; with
  it, 13.8 on average (maximum 25), and all 429 fits converged (tolerance 1e-6 on the logit scale).
- **Guaranteed monotone descent:** within a block, F is separable across the block's parameters. Every Newton
  step is halved (at most 30 times, then dropped) until that parameter's own part of F does not increase beyond
  a 1e-12 relative rounding tolerance. Re-centring minimises exactly along flat directions. So F is
  non-increasing by construction, not only empirically.
- **Independent validation** (`tests/test_features_opponent_validation.py`):
  - The penalised objective and its gradient are re-implemented from the model specification in the test and
    minimised with scipy L-BFGS-B, a generic optimiser and a dev-only dependency.
  - Datasets: random, sparse (including all-won and all-lost records) and decay-weighted.
  - The custom solution reaches the same optimum: objective difference at most 1e-10 relative (observed about
    1e-12), and fitted probabilities agree within 1e-7 (observed about 1e-8). The module's own objective
    function agrees with the independent one.
  - A fit stopped after 2 sweeps is rejected by the same tolerances, so the comparison is discriminating.
  - Further tests check monotone objective traces, convergence from a deliberately bad random initialisation
    to the same optimum, finite and bounded estimates across data scales from 1e-6 to 1e4, and extreme records.
- **Chronology:** refitted every 4 batches, warm-started from the previous fit, on observations released before
  the fit batch only. A snapshot uses the latest fit at or before its batch (`adjustment_fit_batch`), so adjusted
  features can be up to 3 batches staler than the cumulative counts. No model is fitted on the full history and
  applied backwards.
- **Exported values:**
  - `serve_adjusted = sigmoid(mu_overall + alpha_i)`: SPW against an average returner.
  - `return_adjusted = 1 - sigmoid(mu_overall - beta_i)`.
  - surface versions of both.
  - `a_spw_vs_b` / `b_spw_vs_a`: the model's expected SPW in this specific matchup.
  - the strengths and their weighted point counts.
- **Face validity**, using fitted parameters only and no outcomes: the November 2015 fit ranks Karlovic,
  Federer, Raonic, Isner and Djokovic as the top servers, and Djokovic, Murray, Nadal, Ferrer and Federer as the
  top returners.
- **Cost:** 429 fits take about 189 s in total on this machine (at most 1.3 s each, up to 197,726 observations),
  about 2.5x the unsafeguarded 74 s, because of the extra objective evaluations of the descent safeguard.

### Differences

`spw_diff`, `rpw_diff`, `surface_spw_diff`, `surface_rpw_diff`, `serve_adjusted_diff`, `return_adjusted_diff`
and `matchup_spw_diff`, always as A minus B. The individual player values are kept as well.

## 4. Model-input guard (for Phase 3B)

The feature table keeps audit fields for traceability, but a predictive model matrix may contain only
pre-match information. `tennis_data.features.model_inputs` enforces this:

- **Allowlist:** registry columns with role `feature` (all player features, differences, matchup) or `context`
  (`surface`, `tournament_level`), plus the Phase 2 pre-match Elo state (`player_a_elo_before`, ...,
  `predicted_probability_a`).
- **Always forbidden**, even if listed explicitly:
  - the label and outcome fields: `target_a_wins`, `player_a_won`, `winner_id`, `score`, `match_status`, `minutes`;
  - the current match's own statistics: Phase 1 `player_*_svpt` etc. and Phase 3A `a_sp_played` etc.;
  - post-match audit fields: `stats_eligible`, `stats_exclusion_reason`, `stats_status`, `quality_flags`, and
    the outcome-derived eligibility flags `is_target` / `is_update`;
  - `release_batch` and evaluation-period labels.
- **Not inputs:** keys (`match_id`, `prediction_batch`, ids, ...) and audit or meta columns
  (`adjustment_fit_batch`, `configuration_hash`, ...).
- `model_matrix(df, cfg)` selects the allowlisted columns, and `check_model_inputs` raises `ModelInputError`
  on any forbidden or unregistered column. Phase 3B must build every model matrix through it.

## 5. Outputs

| path | content |
|---|---|
| `data/features/serve_return/season=YYYY/part-0.parquet` | feature table (142 columns), DuckDB view `serve_return_features` |
| `data/features/match_stats_eligibility.parquet` | per validated match: point counts, `stats_eligible`, exclusion reason, Elo flags |
| `data/features/opponent_fit_log.parquet` | per fit: batch, observations, iterations, convergence, surface baselines |
| `data/features/opponent_fit_parameters.parquet` | alpha / beta per player per fit (players with data) |
| `data/features/performance.json` | runtimes and peak memory of the last run (not reproducible) |
| `reports/features/coverage_report.md`, `tables/*.csv` | coverage and quality report |
| `reports/features/feature_summary.json` | configuration, configuration hash, SHA-256 of every data output |

`configuration_hash` covers the feature settings, the Phase 2 chronology, pools and targets, and the source
commit. Building twice with the same inputs and configuration gives byte-identical Parquet files (62 files on
the real data).

## 6. Readiness and limitations

- **Coverage:**
  - Statistics start in 1991: warm-up rows have no history, and early-1990s rows are prior-heavy.
  - Davis Cup coverage is 8.8% and Carpet 28%.
  - In the test period, 94.4% of rows have prior statistics for both players; in validation, 91.6%.
  - Missingness is not at random. Players with more data tend to be stronger and more active, so missingness
    and sample-size indicators should be modelled, not imputed.
- **Not tuned:** the half-lives (52 weeks for features, 104 weeks inside the opponent model), the refit
  interval and the window lengths. The κ values come from validation-period moments, not from predictive
  performance.
- **Approximate recency:** windows and decay are weekly and based on event start dates (Section 2).
- **Excluded evidence:** retirements are excluded by default, and stats-quarantined matches are unavailable.
  Both slightly bias histories toward fully played matches.
- **Sample sizes assume independent points.** The point-level Kish ESS is an upper bound under within-match
  dependence (Section 3); the intra-match correlation is not estimated.
- **The surface blend is a heuristic.** The opponent model uses one global strength plus surface deviations and
  has no time dynamics beyond observation decay.
- **The 2010-2026 period is not a pristine hold-out** (see `docs/elo.md`). Feature choices made in Phase 3B must
  be selected on validation data, with a separate, untouched final hold-out.

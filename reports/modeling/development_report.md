# Phase 3B development report: does serve/return information add to surface Elo?

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

Configuration hash `5e72010100065889`, code hash `029557fed53c1204`, model version `m3b-1+engine1`, dataset key `dbb3f87467f6edf5`, upstream feature hash `3d889b94456995d1`.

## 1. Question and answer

On the development years 2002–2009 (24,622 matches, 322 weekly batches) the pre-specified rule selected **M4A+raw** (M1 + recency-weighted (52-week decay) serve/return differences). Against the Phase 2 surface-Elo baseline (M0) its development log-loss difference is -0.00630 (95% CI [-0.00833, -0.00439]).

What carries the gain matters as much as the size (see the ablation report): the overall career-long shrunk serve/return rates (M1) do **not** improve on Elo; the 52-week recency-weighted rates and the recent-activity (effective sample size) features do. Opponent-adjusted and matchup features (M3) give a smaller, also non-zero improvement.

**Where the gain comes from.** A large part of the improvement is a cold-start correction: when a player has little or no released serve/return history (often Davis Cup or newly arrived players, whom Elo starts at the initial rating) the recency inputs and their missingness indicator carry most information. sparse (<1000 prior service points, either player): 5,323 matches (22%), ΔLL -0.02374 [-0.03018, -0.01718], 81% of the total log-loss reduction; substantial (>=1000 for both): 19,299 matches (78%), ΔLL -0.00149 [-0.00262, -0.00038], 19% of the total log-loss reduction. The improvement on matches between players with substantial history is the more relevant test of serve/return skill information and is smaller.

**This is not an independent test.** The development years were also used to tune the Phase 2 Elo parameters (1991–2009 validation), so M0 is, if anything, favoured here. The 2010–2026 period was already inspected in Phase 2 and is reported separately as exploratory (`exploratory_report.md`).

## 2. Protocol

- **Models.** L2-penalised logistic regression without intercept on antisymmetric A−B inputs, objective Σ logloss + |β|²/(2C). M0 is the stored Phase 2 surface-Elo probability (not refitted, not retuned). M0R refits a single slope on the Elo logit z = log(p/(1−p)), p clipped to [1e-06, 1 − 1e-06]: it isolates what pure rescaling of Elo does.
- **Player-order symmetry by construction.** Every input changes sign when A and B are swapped, there is no intercept, scaling divides by the training-fold RMS (no centring), and missing differences are 0. Hence P(A beats B) = 1 − P(B beats A); checked numerically on every fit (max deviation 2.2e-16).
- **Walk-forward annual refits.** The model for year y is trained on rows with prediction_batch ≥ 1991-01-01 and **release_batch < first prediction batch of y** (a result is known only after its batch closes; Slams, 96+ draw Masters and the Olympics release a week later). The tournament start date is never used as evidence that a result was known. Per-year cutoffs and counts are in Section 3 and in `data/modeling/manifest.json`.
- **Folds.** Development years 2002–2009 grouped as 2002–2004, 2005–2007, 2008–2009. Years 1994–2001 are predicted only to provide earlier out-of-fold predictions for calibration.
- **Selection rule (fixed before results).** (1) per experiment, C ∈ [0.01, 0.1, 1.0, 10.0] by pooled development log loss (ties → smaller C); (2) among candidates ['M0R', 'M1', 'M2', 'M3', 'M4A', 'M4B'] the lowest pooled development log loss, replaced by the candidate with the fewest inputs within 0.0005 of it; (3) calibration method (raw < temperature < isotonic in simplicity) by the same rule.
- **Missing values.** Missing differences are set to 0 (the only antisymmetric value); the counts are in Section 4. No row is dropped.
- **Leakage audit.** For a random sample of 300 rows the released service history of player A was recomputed from the Phase 1 table and the Phase 3A eligibility table alone (release_batch < prediction_batch) and matched the features exactly (max abs diff observed points 0, decay rate 4.4e-16; current match included 0 times).

## 3. Train/validation boundaries

| prediction year | role | training cutoff (first batch of year) | last training release batch | training rows | predicted rows |
|---|---|---|---|---|---|
| 1994 | calibration history | 1994-01-01 | 1993-12-04 | 11,103 | 3,916 |
| 1995 | calibration history | 1995-01-07 | 1994-12-31 | 15,019 | 3,715 |
| 1996 | calibration history | 1996-01-06 | 1995-12-30 | 18,734 | 3,693 |
| 1997 | calibration history | 1997-01-04 | 1996-12-28 | 22,427 | 3,461 |
| 1998 | calibration history | 1998-01-03 | 1997-11-22 | 25,888 | 3,491 |
| 1999 | calibration history | 1999-01-02 | 1998-11-28 | 29,379 | 3,246 |
| 2000 | calibration history | 2000-01-01 | 1999-11-27 | 32,625 | 3,367 |
| 2001 | calibration history | 2001-01-06 | 2000-12-30 | 35,992 | 3,222 |
| 2002 | development | 2002-01-05 | 2001-12-29 | 39,214 | 3,097 |
| 2003 | development | 2003-01-04 | 2002-12-28 | 42,311 | 2,997 |
| 2004 | development | 2004-01-03 | 2003-11-22 | 45,308 | 3,186 |
| 2005 | development | 2005-01-01 | 2004-11-27 | 48,494 | 3,257 |
| 2006 | development | 2006-01-07 | 2005-12-31 | 51,751 | 3,155 |
| 2007 | development | 2007-01-06 | 2006-12-30 | 54,906 | 3,075 |
| 2008 | development | 2008-01-05 | 2007-12-29 | 57,981 | 2,901 |
| 2009 | development | 2009-01-03 | 2008-11-15 | 60,882 | 2,954 |

## 4. Inputs and missing values

| input | definition | filled with 0 (all rows, 1994–2026) |
|---|---|---|
| elo_logit | log(p/(1-p)) of the Phase 2 surface-Elo probability, p clipped to [eps, 1-eps] | never null |
| spw_shrunk_diff | a_spw_shrunk - b_spw_shrunk | 0 (0.00%) |
| rpw_shrunk_diff | a_rpw_shrunk - b_rpw_shrunk | 0 (0.00%) |
| surface_spw_diff | a_surface_spw - b_surface_spw (heuristic surface blend) | 0 (0.00%) |
| surface_rpw_diff | a_surface_rpw - b_surface_rpw | 0 (0.00%) |
| serve_adjusted_diff | opponent-adjusted SPW vs an average returner, A - B | 0 (0.00%) |
| return_adjusted_diff | opponent-adjusted RPW vs an average server, A - B | 0 (0.00%) |
| matchup_spw_diff | a_spw_vs_b - b_spw_vs_a: expected serve-point edge in this matchup (antisymmetric) | 0 (0.00%) |
| spw_decay_diff | 52-week decay-weighted SPW, A - B | 6,596 (6.74%) |
| rpw_decay_diff | 52-week decay-weighted RPW, A - B | 6,596 (6.74%) |
| decay_missing_diff | 1[A has no decayed history] - 1[B has none] | never null |
| log_service_ess_diff | log1p(Kish ESS of A's decayed service rate) - same for B | 0 (0.00%) |
| log_stat_matches_diff | log1p(prior stats matches A) - log1p(B) | 0 (0.00%) |
| has_history_diff | 1[A has history] - 1[B has history] | 0 (0.00%) |
| service_prior_dominated_diff | 1[A's service estimate is prior-dominated] - same for B | 0 (0.00%) |

## 5. Baseline M0 per fold

| fold | rows | weeks | LL M0 | Brier M0 | ECE M0 | LL M0R | LL M4A+raw |
|---|---|---|---|---|---|---|---|
| 2002-2004 | 9,280 | 121 | 0.62001 | 0.21556 | 0.0138 | 0.61965 | 0.61366 |
| 2005-2007 | 9,487 | 121 | 0.59546 | 0.20503 | 0.0128 | 0.59629 | 0.58977 |
| 2008-2009 | 5,855 | 80 | 0.58162 | 0.19976 | 0.0098 | 0.58272 | 0.57440 |

## 6. Model comparison (pooled development years, identical matches)

| model | inputs | log loss | Brier | accuracy | ECE | ΔLL vs M0 | 95% CI | Bonferroni CI | ΔBrier | 95% CI (Brier) | ΔLL verdict (95%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 0 | 0.60142 | 0.20775 | 0.6691 | 0.0066 | – | – | – | – | – | baseline |
| M0R | 1 | 0.60187 | 0.20791 | 0.6691 | 0.0116 | +0.00045 | [-0.00011, +0.00096] | [-0.00029, +0.00120] | +0.00016 | [-0.00002, +0.00034] | inconclusive (interval includes 0) |
| M1 | 3 | 0.60237 | 0.20817 | 0.6682 | 0.0111 | +0.00094 | [+0.00016, +0.00170] | [-0.00019, +0.00205] | +0.00042 | [+0.00012, +0.00072] | candidate worse |
| M2 | 5 | 0.60129 | 0.20768 | 0.6697 | 0.0112 | -0.00013 | [-0.00113, +0.00086] | [-0.00159, +0.00130] | -0.00007 | [-0.00049, +0.00035] | inconclusive (interval includes 0) |
| M3 | 4 | 0.59868 | 0.20656 | 0.6712 | 0.0109 | -0.00274 | [-0.00419, -0.00132] | [-0.00491, -0.00077] | -0.00118 | [-0.00181, -0.00058] | candidate better |
| M4A | 6 | 0.59512 | 0.20486 | 0.6778 | 0.0116 | -0.00630 | [-0.00833, -0.00439] | [-0.00937, -0.00372] | -0.00289 | [-0.00376, -0.00206] | candidate better |
| M4B | 7 | 0.59485 | 0.20483 | 0.6779 | 0.0159 | -0.00657 | [-0.00843, -0.00468] | [-0.00952, -0.00389] | -0.00292 | [-0.00372, -0.00209] | candidate better |
| A1_SPW | 2 | 0.60204 | 0.20801 | 0.6690 | 0.0114 | +0.00062 | [-0.00009, +0.00129] | [-0.00038, +0.00163] | +0.00026 | [-0.00000, +0.00052] | inconclusive (interval includes 0) |
| A1_RPW | 2 | 0.60178 | 0.20787 | 0.6699 | 0.0124 | +0.00036 | [-0.00021, +0.00087] | [-0.00036, +0.00112] | +0.00012 | [-0.00006, +0.00030] | inconclusive (interval includes 0) |
| A2_SURF | 3 | 0.60195 | 0.20800 | 0.6676 | 0.0113 | +0.00053 | [-0.00053, +0.00161] | [-0.00099, +0.00200] | +0.00025 | [-0.00021, +0.00072] | inconclusive (interval includes 0) |
| A3_NOMATCHUP | 3 | 0.60039 | 0.20733 | 0.6702 | 0.0093 | -0.00103 | [-0.00222, +0.00009] | [-0.00275, +0.00067] | -0.00042 | [-0.00092, +0.00005] | inconclusive (interval includes 0) |
| A3_MATCHUP | 2 | 0.59870 | 0.20657 | 0.6710 | 0.0101 | -0.00272 | [-0.00416, -0.00132] | [-0.00487, -0.00074] | -0.00118 | [-0.00180, -0.00057] | candidate better |
| A0_NOELO | 2 | 0.64022 | 0.22480 | 0.6297 | 0.0084 | +0.03879 | [+0.03391, +0.04360] | [+0.03164, +0.04571] | +0.01705 | [+0.01492, +0.01918] | candidate worse |
| A4_DECAY | 4 | 0.59513 | 0.20486 | 0.6778 | 0.0120 | -0.00629 | [-0.00832, -0.00440] | [-0.00936, -0.00366] | -0.00289 | [-0.00377, -0.00206] | candidate better |
| A4_ESS | 2 | 0.59742 | 0.20584 | 0.6771 | 0.0149 | -0.00400 | [-0.00559, -0.00248] | [-0.00640, -0.00188] | -0.00191 | [-0.00260, -0.00125] | candidate better |
| M4A+raw | 6 | 0.59512 | 0.20486 | 0.6778 | 0.0116 | -0.00630 | [-0.00833, -0.00439] | [-0.00937, -0.00372] | -0.00289 | [-0.00376, -0.00206] | candidate better |

Uncertainty: paired cluster bootstrap of ΔLL = LL(candidate) − LL(M0) on identical matches; 2000 resamples of whole weekly prediction batches (seed 20261025), percentile intervals. Negative Δ favours the candidate. 'Bonferroni CI' uses α = 0.05/14 for the 14 registered experiments compared with M0. An interval that includes 0 is reported as inconclusive.

## 7. Per-fold differences against M0

| fold | rows | Δ M0R | 95% CI | Δ M4A+raw | 95% CI |
|---|---|---|---|---|---|
| 2002-2004 | 9,280 | -0.00035 | [-0.00132, +0.00060] | -0.00635 | [-0.00905, -0.00370] |
| 2005-2007 | 9,487 | +0.00083 | [+0.00004, +0.00160] | -0.00569 | [-0.00874, -0.00297] |
| 2008-2009 | 5,855 | +0.00110 | [+0.00017, +0.00201] | -0.00722 | [-0.01261, -0.00221] |

## 8. Hyperparameter grid (every setting tried)

| experiment | C | inputs | dev LL | LL 2002–2004 | LL 2005–2007 | LL 2008–2009 | converged | chosen |
|---|---|---|---|---|---|---|---|---|
| A0_NOELO | 0.01 | 2 | 0.640281 | 0.652795 | 0.635842 | 0.627638 | True |  |
| A0_NOELO | 0.1 | 2 | 0.640222 | 0.652841 | 0.635705 | 0.627538 | True |  |
| A0_NOELO | 1.0 | 2 | 0.640217 | 0.652847 | 0.635692 | 0.627528 | True |  |
| A0_NOELO | 10.0 | 2 | 0.640216 | 0.652848 | 0.635691 | 0.627527 | True | ✓ |
| A1_RPW | 0.01 | 2 | 0.601947 | 0.619688 | 0.596452 | 0.582730 | True |  |
| A1_RPW | 0.1 | 2 | 0.601794 | 0.619612 | 0.596278 | 0.582492 | True |  |
| A1_RPW | 1.0 | 2 | 0.601780 | 0.619606 | 0.596261 | 0.582469 | True |  |
| A1_RPW | 10.0 | 2 | 0.601779 | 0.619605 | 0.596259 | 0.582467 | True | ✓ |
| A1_SPW | 0.01 | 2 | 0.602220 | 0.620056 | 0.596813 | 0.582712 | True |  |
| A1_SPW | 0.1 | 2 | 0.602057 | 0.619954 | 0.596606 | 0.582522 | True |  |
| A1_SPW | 1.0 | 2 | 0.602041 | 0.619945 | 0.596586 | 0.582504 | True |  |
| A1_SPW | 10.0 | 2 | 0.602040 | 0.619944 | 0.596584 | 0.582502 | True | ✓ |
| A2_SURF | 0.01 | 3 | 0.602139 | 0.619837 | 0.596425 | 0.583346 | True |  |
| A2_SURF | 0.1 | 3 | 0.601969 | 0.619732 | 0.596233 | 0.583111 | True |  |
| A2_SURF | 1.0 | 3 | 0.601953 | 0.619722 | 0.596214 | 0.583088 | True |  |
| A2_SURF | 10.0 | 3 | 0.601951 | 0.619721 | 0.596213 | 0.583085 | True | ✓ |
| A3_MATCHUP | 0.01 | 2 | 0.598781 | 0.615465 | 0.592388 | 0.582696 | True |  |
| A3_MATCHUP | 0.1 | 2 | 0.598708 | 0.615436 | 0.592291 | 0.582592 | True |  |
| A3_MATCHUP | 1.0 | 2 | 0.598701 | 0.615433 | 0.592282 | 0.582582 | True |  |
| A3_MATCHUP | 10.0 | 2 | 0.598700 | 0.615433 | 0.592281 | 0.582580 | True | ✓ |
| A3_NOMATCHUP | 0.01 | 3 | 0.600509 | 0.617832 | 0.595064 | 0.581876 | True |  |
| A3_NOMATCHUP | 0.1 | 3 | 0.600401 | 0.617785 | 0.594926 | 0.581720 | True |  |
| A3_NOMATCHUP | 1.0 | 3 | 0.600391 | 0.617781 | 0.594912 | 0.581704 | True |  |
| A3_NOMATCHUP | 10.0 | 3 | 0.600390 | 0.617781 | 0.594911 | 0.581703 | True | ✓ |
| A4_DECAY | 0.01 | 4 | 0.595260 | 0.613785 | 0.589833 | 0.574691 | True |  |
| A4_DECAY | 0.1 | 4 | 0.595141 | 0.613730 | 0.589684 | 0.574519 | True |  |
| A4_DECAY | 1.0 | 4 | 0.595129 | 0.613725 | 0.589669 | 0.574502 | True |  |
| A4_DECAY | 10.0 | 4 | 0.595128 | 0.613724 | 0.589667 | 0.574500 | True | ✓ |
| A4_ESS | 0.01 | 2 | 0.597533 | 0.615858 | 0.591768 | 0.577829 | True |  |
| A4_ESS | 0.1 | 2 | 0.597430 | 0.615830 | 0.591631 | 0.577662 | True |  |
| A4_ESS | 1.0 | 2 | 0.597420 | 0.615828 | 0.591618 | 0.577645 | True |  |
| A4_ESS | 10.0 | 2 | 0.597419 | 0.615828 | 0.591617 | 0.577644 | True | ✓ |
| M0R | 0.01 | 1 | 0.602015 | 0.619714 | 0.596478 | 0.582934 | True |  |
| M0R | 0.1 | 1 | 0.601882 | 0.619656 | 0.596310 | 0.582742 | True |  |
| M0R | 1.0 | 1 | 0.601870 | 0.619651 | 0.596293 | 0.582723 | True |  |
| M0R | 10.0 | 1 | 0.601869 | 0.619651 | 0.596292 | 0.582721 | True | ✓ |
| M1 | 0.01 | 3 | 0.602618 | 0.620560 | 0.597036 | 0.583226 | True |  |
| M1 | 0.1 | 3 | 0.602392 | 0.620389 | 0.596794 | 0.582936 | True |  |
| M1 | 1.0 | 3 | 0.602369 | 0.620372 | 0.596770 | 0.582907 | True |  |
| M1 | 10.0 | 3 | 0.602367 | 0.620371 | 0.596768 | 0.582904 | True | ✓ |
| M2 | 0.01 | 5 | 0.601599 | 0.619395 | 0.595872 | 0.582672 | True |  |
| M2 | 0.1 | 5 | 0.601320 | 0.619156 | 0.595579 | 0.582353 | True |  |
| M2 | 1.0 | 5 | 0.601293 | 0.619133 | 0.595550 | 0.582321 | True |  |
| M2 | 10.0 | 5 | 0.601290 | 0.619131 | 0.595547 | 0.582318 | True | ✓ |
| M3 | 0.01 | 4 | 0.598768 | 0.615460 | 0.592453 | 0.582544 | True |  |
| M3 | 0.1 | 4 | 0.598692 | 0.615433 | 0.592334 | 0.582462 | True |  |
| M3 | 1.0 | 4 | 0.598686 | 0.615431 | 0.592323 | 0.582454 | True |  |
| M3 | 10.0 | 4 | 0.598685 | 0.615431 | 0.592322 | 0.582454 | True | ✓ |
| M4A | 0.01 | 6 | 0.595294 | 0.613791 | 0.589952 | 0.574630 | True |  |
| M4A | 0.1 | 6 | 0.595137 | 0.613672 | 0.589791 | 0.574423 | True |  |
| M4A | 1.0 | 6 | 0.595122 | 0.613660 | 0.589775 | 0.574402 | True |  |
| M4A | 10.0 | 6 | 0.595120 | 0.613659 | 0.589774 | 0.574400 | True | ✓ |
| M4B | 0.01 | 7 | 0.594945 | 0.613499 | 0.589630 | 0.574151 | True |  |
| M4B | 0.1 | 7 | 0.594852 | 0.613486 | 0.589474 | 0.574033 | True | ✓ |
| M4B | 1.0 | 7 | 0.594877 | 0.613542 | 0.589479 | 0.574039 | True |  |
| M4B | 10.0 | 7 | 0.594880 | 0.613549 | 0.589480 | 0.574040 | True |  |

## 9. Selection outcome

| candidate | chosen C | inputs | pooled dev LL |
|---|---|---|---|
| M0R | 10.0 | 1 | 0.601869 |
| M1 | 10.0 | 3 | 0.602367 |
| M2 | 10.0 | 5 | 0.601290 |
| M3 | 10.0 | 4 | 0.598685 |
| M4A | 10.0 | 6 | 0.595120 |
| M4B | 0.1 | 7 | 0.594852 |

Lowest development log loss: **M4B**. Selected after the simplicity rule: **M4A** (C = 10.0). Calibration scores: raw 0.595120, temperature 0.594663, isotonic 0.597700 → **raw**. Frozen model hash `bc4e5c870c6e21b9`.

## 10. Dependence sensitivity (block choice)

| comparison (new vs reference) | rows | blocks | ΔLL | 95% CI | ΔBrier | 95% CI | P*(ΔLL<0) | verdict |
|---|---|---|---|---|---|---|---|---|
| M4A+raw vs M0 [week] | 24622 | 322 | -0.00630 | [-0.00833, -0.00439] | -0.00289 | [-0.00376, -0.00206] | 1.000 | candidate better |
| M4A+raw vs M0 [tournament] | 24622 | 1200 | -0.00630 | [-0.00788, -0.00468] | -0.00289 | [-0.00355, -0.00219] | 1.000 | candidate better |
| M4A+raw vs M0 [season] | 24622 | 8 | -0.00630 | [-0.00718, -0.00528] | -0.00289 | [-0.00333, -0.00240] | 1.000 | candidate better |
| M0R vs M0 [week] | 24622 | 322 | +0.00045 | [-0.00011, +0.00096] | +0.00016 | [-0.00002, +0.00034] | 0.054 | inconclusive (interval includes 0) |
| M0R vs M0 [tournament] | 24622 | 1200 | +0.00045 | [-0.00005, +0.00095] | +0.00016 | [-0.00001, +0.00033] | 0.033 | inconclusive (interval includes 0) |
| M0R vs M0 [season] | 24622 | 8 | +0.00045 | [-0.00006, +0.00097] | +0.00016 | [+0.00000, +0.00032] | 0.049 | inconclusive (interval includes 0) |

Blocks: `week` = prediction batch (primary); `tournament` = event × year; `season` = calendar year (very few blocks, wide intervals). Matches also share players across blocks; no block scheme removes that dependence, so intervals may be somewhat too narrow.

## 11. Subgroups (development years)

| group | level | n | LL M0 | LL M0R | LL M4A+raw | Δ M4A+raw | 95% CI | Δ M0R | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| surface | Carpet | 1463 | 0.63670 | 0.63646 | 0.62684 | -0.00985 | [-0.01880, -0.00060] | -0.00024 | [-0.00225, +0.00170] |
| surface | Clay | 8479 | 0.60436 | 0.60475 | 0.59963 | -0.00473 | [-0.00789, -0.00167] | +0.00039 | [-0.00052, +0.00125] |
| surface | Grass | 2445 | 0.59512 | 0.59650 | 0.58757 | -0.00755 | [-0.01131, -0.00398] | +0.00138 | [-0.00040, +0.00307] |
| surface | Hard | 12235 | 0.59643 | 0.59681 | 0.58971 | -0.00672 | [-0.00903, -0.00443] | +0.00038 | [-0.00036, +0.00110] |
| event_type | Grand Slam | 3888 | 0.53888 | 0.54279 | 0.54254 | +0.00365 | [+0.00051, +0.00679] | +0.00391 | [+0.00282, +0.00489] |
| event_type | other tour-level | 20734 | 0.61315 | 0.61295 | 0.60498 | -0.00817 | [-0.01034, -0.00591] | -0.00020 | [-0.00077, +0.00034] |
| tournament_level | A | 13622 | 0.61394 | 0.61348 | 0.60916 | -0.00478 | [-0.00664, -0.00292] | -0.00046 | [-0.00111, +0.00016] |
| tournament_level | D | 2534 | 0.60513 | 0.60888 | 0.57028 | -0.03485 | [-0.04501, -0.02581] | +0.00375 | [+0.00272, +0.00477] |
| tournament_level | F | 120 | 0.62873 | 0.62784 | 0.63217 | +0.00344 | [-0.00893, +0.01596] | -0.00089 | [-0.00734, +0.00609] |
| tournament_level | G | 3888 | 0.53888 | 0.54279 | 0.54254 | +0.00365 | [+0.00051, +0.00679] | +0.00391 | [+0.00282, +0.00489] |
| tournament_level | M | 4458 | 0.61487 | 0.61324 | 0.61119 | -0.00368 | [-0.00614, -0.00119] | -0.00163 | [-0.00278, -0.00058] |
| sample_size | 0: no prior points (<1) | 2051 | 0.60525 | 0.60956 | 0.54565 | -0.05960 | [-0.07104, -0.04869] | +0.00431 | [+0.00332, +0.00536] |
| sample_size | 1: 1-999 | 3272 | 0.58430 | 0.58475 | 0.58304 | -0.00126 | [-0.00673, +0.00418] | +0.00045 | [-0.00092, +0.00187] |
| sample_size | 2: 1000-4999 | 6645 | 0.61395 | 0.61250 | 0.61004 | -0.00391 | [-0.00577, -0.00198] | -0.00145 | [-0.00250, -0.00038] |
| sample_size | 3: >= 5000 | 12654 | 0.59865 | 0.59946 | 0.59843 | -0.00023 | [-0.00151, +0.00103] | +0.00081 | [+0.00014, +0.00141] |
| history | sparse (<1000 prior service points, either player) | 5323 | 0.59237 | 0.59431 | 0.56863 | -0.02374 | [-0.03018, -0.01718] | +0.00194 | [+0.00099, +0.00286] |
| history | substantial (>=1000 for both) | 19299 | 0.60392 | 0.60395 | 0.60243 | -0.00149 | [-0.00262, -0.00038] | +0.00004 | [-0.00053, +0.00058] |
| elo_confidence_pct | 50-60 | 8570 | 0.68635 | 0.68645 | 0.67495 | -0.01140 | [-0.01505, -0.00793] | +0.00010 | [-0.00013, +0.00031] |
| elo_confidence_pct | 60-70 | 6693 | 0.64408 | 0.64470 | 0.63834 | -0.00574 | [-0.00838, -0.00332] | +0.00062 | [-0.00007, +0.00129] |
| elo_confidence_pct | 70-80 | 5236 | 0.56007 | 0.56137 | 0.55778 | -0.00229 | [-0.00487, +0.00021] | +0.00130 | [+0.00010, +0.00243] |
| elo_confidence_pct | 80-90 | 3229 | 0.45779 | 0.45663 | 0.45474 | -0.00306 | [-0.00632, -0.00004] | -0.00116 | [-0.00358, +0.00118] |
| elo_confidence_pct | 90-100 | 894 | 0.22893 | 0.23220 | 0.23208 | +0.00315 | [-0.00252, +0.00843] | +0.00327 | [-0.00060, +0.00696] |
| year | 2002 | 3097 | 0.62667 | 0.62580 | 0.61945 | -0.00722 | [-0.01108, -0.00332] | -0.00087 | [-0.00235, +0.00055] |
| year | 2003 | 2997 | 0.61340 | 0.61376 | 0.60578 | -0.00762 | [-0.01239, -0.00289] | +0.00036 | [-0.00147, +0.00195] |
| year | 2004 | 3186 | 0.61974 | 0.61921 | 0.61544 | -0.00430 | [-0.00977, +0.00103] | -0.00053 | [-0.00212, +0.00101] |
| year | 2005 | 3257 | 0.59665 | 0.59784 | 0.59015 | -0.00650 | [-0.01153, -0.00170] | +0.00119 | [-0.00009, +0.00243] |
| year | 2006 | 3155 | 0.59367 | 0.59450 | 0.58986 | -0.00380 | [-0.00756, -0.00024] | +0.00084 | [-0.00058, +0.00205] |
| year | 2007 | 3075 | 0.59605 | 0.59649 | 0.58929 | -0.00677 | [-0.01302, -0.00119] | +0.00043 | [-0.00111, +0.00191] |
| year | 2008 | 2901 | 0.59306 | 0.59353 | 0.58597 | -0.00709 | [-0.01549, +0.00015] | +0.00047 | [-0.00106, +0.00189] |
| year | 2009 | 2954 | 0.57039 | 0.57210 | 0.56304 | -0.00735 | [-0.01458, -0.00148] | +0.00171 | [+0.00045, +0.00280] |

Subgroup intervals are unadjusted and descriptive; with this many groups some will exclude 0 by chance.

## 12. Coefficients of M4A by prediction year (original input units)

| year | training rows | elo_logit | spw_shrunk_diff | rpw_shrunk_diff | spw_decay_diff | rpw_decay_diff | decay_missing_diff |
|---|---|---|---|---|---|---|---|
| 2002 | 39,214 | 0.6733 | 1.1008 | -0.3022 | 4.6359 | 4.5572 | -0.7533 |
| 2003 | 42,311 | 0.6719 | 1.1789 | -0.3439 | 4.5398 | 4.6587 | -0.7701 |
| 2004 | 45,308 | 0.6797 | 1.0643 | -0.5976 | 4.5284 | 4.7409 | -0.7987 |
| 2005 | 48,494 | 0.6866 | 0.9189 | -0.9890 | 4.4469 | 4.9293 | -0.7933 |
| 2006 | 51,751 | 0.6965 | 0.9118 | -1.1591 | 4.3775 | 5.0594 | -0.8182 |
| 2007 | 54,906 | 0.7039 | 0.8029 | -1.0469 | 4.4254 | 4.9196 | -0.8182 |
| 2008 | 57,981 | 0.7099 | 0.5691 | -0.7491 | 4.5509 | 4.6183 | -0.8418 |
| 2009 | 60,882 | 0.7176 | 0.4803 | -0.7769 | 4.5206 | 4.4839 | -0.8606 |

## 13. Evaluation-design caveats

1. **M0 was tuned on these years.** Phase 2 chose the Elo K-factors and shrinkage on 1991–2009 log loss; the development years are inside that window, so M0 is optimistically evaluated here and the comparison is conservative for the candidates.
2. **Phase 3A priors.** The shrinkage κ values were set in Phase 3A from 1991–2009 player totals (moments of serve/return rates, not match outcomes). This is mild information from within the period.
3. **Post-hoc diagnostics.** `A4_DECAY` and `A4_ESS` were added to the registry after the development grid was first inspected, only to locate where the M4 gain comes from. They are never selection candidates.
4. **Not independent.** 2010–2026 results were examined in Phase 2; they are exploratory evidence only.

# Research summary: do serve and return statistics improve on surface Elo?

This report summarises the research findings of this repository. All numbers are copied from the committed,
generated reports in `reports/`, and nothing here is a new experiment. The detailed sources are:

- the Phase 2 Elo evaluation: [elo/evaluation_report.md](../reports/elo/evaluation_report.md);
- the Phase 3B modelling reports:
  - [development](../reports/modeling/development_report.md);
  - [exploratory](../reports/modeling/exploratory_report.md);
  - [calibration](../reports/modeling/calibration_report.md);
  - [ablation](../reports/modeling/ablation_report.md);
  - [model card](../reports/modeling/model_card.md).

The methodology documents are [elo.md](elo.md), [features.md](features.md) and [modeling.md](modeling.md).

## 1. Motivation

Elo-type ratings are a strong, simple baseline for tennis. They summarise who beat whom, adjusted for opponent
strength. Point-level serve and return statistics measure *how* matches were won, so they could add information:
about form, about player style, or about players whose rating is still unreliable. The research tests whether such
information improves pre-match probabilities. It insists on two things:

- **explicit point-in-time construction**: deterministic, batch-level rules decide which results a feature may use;
- **an honest account of how much independent evidence the evaluation provides.**

## 2. Research questions

1. Can historical serving and returning performance improve match-winner prediction beyond surface Elo?
2. How large is the improvement under historical walk-forward evaluation?
3. How do performance and calibration vary by surface, tournament type and player experience?
4. What limits the assessment when the historical record has no match-level timestamps?

## 3. Data and sample construction

- **Source.** Jeff Sackmann's ATP tour-level match files, fetched at a pinned mirror commit (`83733587…`) with
  SHA-256 checksums.
- **Validation.** 199,389 source rows from 1968–2026 give 199,280 validated matches and 109 quarantined rows:
  - quarantined rows include duplicates, impossible statistics and invalid identifiers;
  - 5,769 validated rows carry a warning flag;
  - `validated + quarantined = source` is enforced ([data-quality report](../reports/data_quality_report.md)).
- **Modelling population.** 193,634 completed matches in the tour-level rating pool form the Elo targets and
  feature rows.
  - Serve/return statistics exist from 1991.
  - Predictions are evaluated on 2002–2026: 24,622 development matches and 45,191 exploratory matches.
  - Years 1994–2001 supply out-of-fold history for calibration.
- **Player order.** Players are labelled A and B from a hash of the match ID, never from the result. Every model is
  exactly symmetric under swapping A and B.

## 4. Chronological information constraints

The archive records each event's start date, but no reliable match dates or times. The project therefore uses
deterministic, frozen, **batch-level** information rules:

- Matches are grouped into **weekly batches** (Saturday to Friday) by event start.
- Every prediction in a batch uses the state frozen at the batch start.
- A result enters the state only when its nominal **release batch** closes:
  - one week for ordinary events;
  - two weeks for Grand Slams, 96+-draw Masters and the Olympics.
- The tournament start date is never treated as evidence that a result was known.
- A leakage audit recomputed the released service history of 300 random rows from the Phase 1 table and the
  eligibility table alone. It matched exactly, and the current match was included 0 times.

Under these rules, no historical feature includes its own target match. This mitigates temporal leakage. It does
**not** prove that every result was actually known before every prediction, and the rules have known costs:

- **Ratings can be staler** than a match-by-match replay would allow, by up to one or two weeks.
- **Within-event order is a convention.**
- **Some events overrun their nominal release batch.** The Phase 5A chronology audit
  ([report](../reports/operations/backfill_authorization_owner-2026-10-26.md)) found events whose last matches were
  played on or after the day their results enter the state. These include two-week events finishing a few days
  after release, events starting mid-week, and events longer than their nominal weeks. For such events a late-round
  result can be treated as available before it was known. The archive's event start dates cannot reveal these cases.
- **The stronger safeguard is disabled.** The proposed `state_availability_v1` would restrict prospective inference
  only. It remains disabled, and the historical results were not recomputed under it.

## 5. Elo baseline

Global Elo and surface-specific Elo were tuned on 1991–2009 validation log loss:

- **Global Elo:** K = 24.
- **Surface Elo:** K = 32, constant initial rating, carpet rated separately. Its prediction shrinks the surface
  rating toward the global rating with weight n_s / (n_s + 40), where n_s is the player's prior matches on that
  surface.

Updates within a batch are order-independent and zero-sum.

On 2010–2026 (45,191 matches), surface Elo beats global Elo:

| | log loss |
|---|---|
| surface Elo | 0.6040 |
| global Elo | 0.6074 |

The difference is −0.00333, with 95% CI [−0.00458, −0.00210]. Surface Elo (**M0**) is the baseline for all model
comparisons, and it was not refitted or retuned in Phase 3B.

## 6. Point-in-time serve and return features

For each player and match, features are computed from released statistics only, under the batch rules of §4:

- **Service and return points won**, point-weighted: total won divided by total played, never an average of match
  percentages.
- **Career rates** with beta-binomial shrinkage toward a pool prior that is itself computed chronologically.
- **52-week decay-weighted rates**, with weight 2^(−age/52 weeks), plus Kish effective sample sizes.
- **Surface blends, opponent-adjusted strengths** (a ridge-binomial model refitted on released data) and a
  matchup edge.

Missing values stay null in the feature table. In model inputs, a missing A − B difference is set to 0, the only
value consistent with player-order symmetry, and indicator variables mark missing recent history. An allowlist
guard prevents outcome, current-match and audit columns from entering a model matrix.

## 7. The M4A model

The model is an L2-penalised logistic regression without intercept, on antisymmetric A − B inputs, fitted by
damped Newton's method. It is symmetric by construction: P(A beats B) = 1 − P(B beats A), and numerical checks
deviate by at most 2.2e-16.

| input | meaning | production coefficient (original units) |
|---|---|---|
| `elo_logit` | log-odds of the surface-Elo probability | +0.7508 |
| `spw_shrunk_diff` | career service points won, shrunk | −0.1944 |
| `rpw_shrunk_diff` | career return points won, shrunk | −0.8708 |
| `spw_decay_diff` | service points won, 52-week decay | +4.3836 |
| `rpw_decay_diff` | return points won, 52-week decay | +3.7954 |
| `decay_missing_diff` | 1[A has no recent history] − 1[B has none] | −0.9047 |

The model uses C = 10 and `raw` probabilities, with no post-hoc calibration.

The career and recent rates are strongly correlated, so their opposite signs should not be read as separate causal
effects. The ablations show that career rates add nothing once the recent rates are present: ΔLL −0.00001,
inconclusive.

**Selection.** The selection rule was fixed before any results were computed:

1. For each experiment, choose C from {0.01, 0.1, 1, 10} by pooled development log loss.
2. Among six candidates (M0R, M1, M2, M3, M4A, M4B), take the lowest development log loss. Then replace it by the
   candidate with the fewest inputs within 0.0005.
3. Choose the calibration method (raw, then temperature, then isotonic) the same way.

M4B had the lowest development log loss (0.59485). M4A (0.59512) was within the tolerance with one input fewer, so
M4A was selected. Temperature scaling (0.59466) was within the tolerance of raw (0.59512), so raw was kept. The
specification was then frozen and hashed (`bc4e5c870c6e21b9`, configuration `5e72010100065889`) before any
2010–2026 row was scored.

## 8. Evaluation design

- **Walk-forward annual refits.** The model for season y is trained on all results released before the first batch
  of y. Calibration, where used, comes from out-of-fold predictions of earlier years.
- **Periods.**
  - **Development** (2002–2009, three folds) is where models were compared and selected.
  - **Exploratory** (2010–2026) was scored once, after the freeze.
- **Metrics.**
  - **Log loss** is the average of −log(probability given to the actual winner). It penalises confident mistakes
    heavily, and lower is better.
  - **Brier score** is the mean squared error of the probability.
  - **Accuracy** is how often the favourite won.
  - **ECE** (expected calibration error) is the average gap between predicted and observed frequencies across ten
    probability bins.
- **Paired comparison.** Every metric is computed on identical matches for M4A and M0. Uncertainty comes from a
  paired cluster bootstrap that resamples whole weekly batches (2,000 resamples, percentile intervals). The 14
  registered experiments also receive Bonferroni-adjusted intervals.

**Why the exploratory period is not an independent confirmation.** The 2010–2026 matches were scored with a frozen
specification, and no parameter changed after they were seen. But the same period served as the Phase 2 Elo "test"
period, and its results were known when the Phase 3 features and candidate models were designed. The development
period also overlaps the Elo tuning window, which favours M0 there. Both periods are therefore *retrospective
evidence under a disciplined protocol*, not an untouched holdout. Only the pre-specified prospective protocol
would provide that, and it has not been run (§13).

## 9. Main results

| period | matches | weeks | LL M4A | LL M0 | ΔLL (95% CI) | Bonferroni CI | ΔBrier (95% CI) | accuracy M4A / M0 |
|---|---|---|---|---|---|---|---|---|
| development 2002–2009 | 24,622 | 322 | 0.59512 | 0.60142 | −0.00630 [−0.00833, −0.00439] | [−0.00937, −0.00372] | −0.00289 [−0.00376, −0.00206] | 67.8% / 66.9% |
| exploratory 2010–2026 | 45,191 | 645 | 0.59861 | 0.60403 | −0.00541 [−0.00662, −0.00423] | [−0.00730, −0.00361] | −0.00231 [−0.00284, −0.00180] | 67.0% / 66.5% |

- The improvement is about 1% of the baseline log loss. It is consistent:
  - M4A is better than M0 in each development fold and in each exploratory sub-period: 2010–14, 2015–19, 2020–23
    and 2024–26 (Δ from −0.00728 to −0.00414);
  - the point estimate is negative in every exploratory season.
- **What carries the gain (development ablations).**
  - Career-long shrunk rates on top of rescaled Elo are slightly worse: ΔLL +0.00050.
  - The 52-week recency-weighted rates add −0.00725 over M1.
  - Opponent-adjusted strengths with a matchup term add −0.00318 over rescaled Elo.
  - Serve/return features **without** Elo are far worse than Elo: +0.03879. Elo remains the dominant input.
- **Cold start.** Matches where either player has fewer than 1,000 released service points (sparse history)
  account for most of the gain:

  | period | sparse share of matches | ΔLL, sparse | share of the gain | ΔLL, substantial history |
  |---|---|---|---|---|
  | development | 22% | −0.02374 | 81% | −0.00149 [−0.00262, −0.00038] |
  | exploratory | 19% | −0.01585 | 54% | −0.00303 [−0.00375, −0.00234] |

  Part of the improvement therefore corrects Elo's initial rating for new or rarely seen players (often Davis Cup)
  rather than measuring serve/return skill. The gain between established players is the more relevant test of
  skill information. It is smaller, but its interval excludes zero in both periods.

## 10. Calibration

| period | ECE M4A | ECE M0 |
|---|---|---|
| development | 0.0116 | 0.0066 |
| exploratory | 0.0064 | 0.0130 |

The reliability diagrams (dashboard *Model evaluation*, and `reports/modeling/tables/calibration_*.csv`) lie close
to the diagonal for both models.

- **Post-hoc calibration was tested and not adopted.**
  - Temperature scaling (Platt without intercept) slightly improved development log loss, from 0.59512 to
    0.59466, but fell within the selection tolerance.
  - Isotonic regression was worse (0.59770).
- **Fitted temperature slopes** on the development years were 0.99–1.11. A slope above 1 means the raw
  probabilities were, if anything, slightly under-confident.
- M4A therefore uses uncalibrated (`raw`) probabilities. Exploratory results with temperature scaling (0.59840)
  were reported for completeness, not used.

## 11. Surface and tournament-level findings

Exploratory period, ΔLL of M4A against M0 with 95% CI. Development values are in the development report.

| subgroup | matches | ΔLL | 95% CI |
|---|---|---|---|
| hard | 26,281 | −0.00716 | [−0.00862, −0.00567] |
| grass | 4,644 | −0.00498 | [−0.00739, −0.00283] |
| clay | 14,152 | −0.00202 | [−0.00423, +0.00019] |
| Masters (M) | 9,112 | −0.00651 | [−0.00806, −0.00496] |
| ATP 250/500 (A) | 23,882 | −0.00497 | [−0.00616, −0.00387] |
| Davis Cup (D) | 3,933 | −0.02037 | [−0.02871, −0.01092] |
| Grand Slam (G) | 7,889 | +0.00168 | [−0.00013, +0.00345] |

- **Grand Slams.** M4A is worse than M0 in development: +0.00365 [+0.00051, +0.00679]. In the exploratory period it
  is inconclusive with a positive point estimate. The rescaled-Elo model M0R shows the same pattern. A plausible
  explanation, not tested, is that one slope on the Elo logit is shared between best-of-three and best-of-five
  matches, although favourites win best-of-five matches more often.
- **Clay** gains are smaller and inconclusive in the exploratory period.
- **Davis Cup** gains are large. This is consistent with the cold-start explanation: many players are new to the
  tour-level rating pool.
- **Elo confidence.** Gains are largest when Elo is uncertain (50–80% favourite probability) and inconclusive above
  90%.

## 12. Uncertainty and robustness

- **Block bootstrap.** The intervals resample whole weekly batches. Resampling tournament-years (2,251 blocks) or
  seasons (17 blocks) gives similar exploratory intervals:
  - tournament-years: [−0.00649, −0.00438];
  - seasons: [−0.00677, −0.00413].
- **Dependence through players.** Players appear across many weeks, and no block scheme used here captures that
  dependence. The intervals may be too narrow. Subgroup intervals are not adjusted for multiplicity.
- **Stability.**
  - A static 2010 fit applied to all exploratory years without refitting performs almost identically: ΔLL −0.00532.
  - Across the 17 annual refits, the Elo coefficient is stable (0.73–0.75). The recency coefficients vary
    moderately: 4.29–4.67 for service and 3.78–4.90 for return.
- **Reproducibility.**
  - Every report is regenerated deterministically from hashed inputs, and repeated engine reconstructions are
    bit-identical.
  - The dashboard dataset reproduces the published development and exploratory metrics exactly.

## 13. Limitations

1. **No untouched holdout.** Development data selected the model, the exploratory period was seen in Phase 2, and
   the Elo tuning window overlaps development.
2. **No prospective results.** The prospective protocol is fixed in advance (exploratory report §9): matches after
   2026-10-25, primary endpoint ΔLL with a weekly-block bootstrap, first analysis after 26 weeks. The operational
   gate is closed, however:
   - backfill INCOMPLETE;
   - prospective predictions DISABLED;
   - no verified live data provider, and no point-level statistics after May 2026.

   No prospective forecast has been made, so none has been evaluated.
3. **Timing approximations.** There are no match timestamps, chronology is weekly, and late-round release
   semantics are imperfect (§4).
4. **Cold-start share.** Much of the gain may disappear with a better Elo initialisation, such as ranking-based
   priors. This is untested.
5. **Format.** There is no best-of-five term, and Grand Slam performance is weaker.
6. **Coverage.** The scope is ATP tour level only, with statistics from 1991.
   - Missing statistics are treated as neutral differences.
   - Height and hand are current values, not time-stamped.
   - An owner-supplied 2025–2026 export failed the statistical compatibility check (agreement 94–98% against a 98%
     threshold), so post-archive data adds results only.
7. **Uncertainty** intervals may be too narrow (§12).

## 14. Possible extensions

These are not implemented, and would each need a new, separately registered evaluation:

- Run the pre-specified prospective evaluation once a verified data source and complete backfill exist. This is
  the only route to an independent result.
- Use a ranking-based or hierarchical Elo initialisation, to test how much of the gain survives a better
  cold-start prior.
- Add a format-specific (best-of-five) term or Elo transform.
- Use two-way (week × player) or player-cluster bootstrap intervals.
- Extend to qualifying and Challenger matches, or to WTA data, with their own pools and priors.

# Phase 4A inference readiness

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

## 1. Assessment

**Ready for genuine prospective predictions: NO.**

Status: historical backfill **INCOMPLETE** · prospective predictions **DISABLED** · real-provider integrations **UNVERIFIED** (gate: `config/prospective.toml` `[status]`).

| component | status |
|---|---|
| frozen model integrity | pass |
| historical replay (inference path = published Phase 3B) | pass |
| state reconstruction (Phase 2 + 3A outputs reproduced) | pass |
| historical backfill complete to the prospective start | **not met** |
| live schedule and result provider verified | **not met** |
| status gate: backfill INCOMPLETE, prospective predictions DISABLED | **not met** |

Blockers:

- historical gap not filled: 22 weekly batches (2026-05-30 to 2026-10-24) have no results; no authorized source was accessible
- feature freshness fails for a match after the prospective start: last_eligible_result, last_point_statistics, missing_backfill_weeks
- no verified live provider: schedules for upcoming matches cannot be ingested automatically
- status gate: backfill INCOMPLETE, prospective predictions DISABLED (config/prospective.toml [status])

The inference machinery is verified; the data it needs is not available. M4A must not be used for prospective forecasts until the backfill reaches the prediction date and a verified provider supplies schedules.

## 2. Frozen model integrity

| check | value |
|---|---|
| integrity checks passed | True |
| model | M4A+raw |
| frozen hash | bc4e5c870c6e21b9 |
| expected frozen hash | bc4e5c870c6e21b9 |
| modeling configuration hash | 5e72010100065889 |
| production artifact sha256 | 24be84b7af161dc5 |
| inputs | elo_logit, spw_shrunk_diff, rpw_shrunk_diff, spw_decay_diff, rpw_decay_diff, decay_missing_diff |
| calibration | raw |
| production training cutoff | 2026-06-06 |

Before every prediction the inference path verifies the SHA-256 of `frozen_model.json` and `production_model.json` against the Phase 3B manifest, the frozen hash (and that it equals the configured one), and that the modeling configuration hash is unchanged. Any mismatch raises `ArtifactIntegrityError` and no probability is produced. Nothing is fitted at inference time.

## 3. Historical replay

274 historical matches were re-submitted as pending upcoming matches (no outcome, no updates) and priced through the new inference path with the frozen M4A annual fit of their year, then compared with the published Phase 3B predictions and Phase 2 baselines:

| metric | value |
|---|---|
| max abs(p − p_published), M4A | 1.11e-16 |
| max abs(p_elo − p_published), surface Elo | 0.00e+00 |
| max abs(P(A) + P(B) − 1) | 2.22e-16 |
| tolerance | 1e-12 |
| years covered | 2002–2025 |
| passed | True |

| category | cases | max M4A error | max Elo error |
|---|---|---|---|
| carpet | 32 | 1.1e-16 | 0.0e+00 |
| clay | 75 | 1.1e-16 | 0.0e+00 |
| development_period | 137 | 1.1e-16 | 0.0e+00 |
| established | 116 | 1.1e-16 | 0.0e+00 |
| exploratory_period | 137 | 1.1e-16 | 0.0e+00 |
| grand_slam | 58 | 1.1e-16 | 0.0e+00 |
| grass | 50 | 1.1e-16 | 0.0e+00 |
| hard | 117 | 1.1e-16 | 0.0e+00 |
| missing_recent_form | 35 | 1.1e-16 | 0.0e+00 |
| multi_week_boundary | 25 | 5.6e-17 | 0.0e+00 |
| new_player | 35 | 1.1e-16 | 0.0e+00 |
| sparse_history | 55 | 1.1e-16 | 0.0e+00 |

Categories overlap (a match can be both clay and Grand Slam). `multi_week_boundary` = a player has an earlier match whose result is not yet released at the prediction batch (week two of a two-week event).

State reconstruction over the whole archive in the same run:

| check | value |
|---|---|
| historical targets with an Elo prediction | 193,634 (unmatched 0) |
| max abs(Elo − published Phase 2) | 0.0e+00 |
| Phase 3A feature table identical | True (193,634 rows × 142 columns) |

## 4. Feature freshness for a match after the prospective start

Diagnostic probe (no probability is reported): a tour-level hard-court match on 2026-10-26 between two players from the archive's last event, with the state as it stands.

Overall status: **FAIL**

| check | value | status |
|---|---|---|
| last_eligible_result | {"value": "2026-06-06", "age_days": 140} | fail |
| last_point_statistics | {"value": "2026-05-23", "age_days": 154} | fail |
| opponent_fit_age_weeks | {"value": 0} | ok |
| missing_backfill_weeks | {"value": 21} | fail |
| unresolved_identities | {"value": 0, "players_in_match": 0} | ok |
| source_age_days | {"value": null} | warning |
| player_recent_history | {"max_weeks_since_last_stats": 25} | ok |

`missing_backfill_weeks` counts weekly batches after the last week with results and before the probe's batch. `opponent_fit_age_weeks` measures the refit schedule only: the opponent model refits when due, but on the same stale observations, so a fresh fit does not mean fresh data. `source_age_days` is null because no provider snapshot exists.

Thresholds (`[freshness]` in `config/prospective.toml`): warn_missing_weeks 1, fail_missing_weeks 4, warn_result_age_days 10, fail_result_age_days 35, warn_fit_age_weeks 5, fail_fit_age_weeks 12, warn_source_age_days 2, fail_source_age_days 14, fail_unresolved_identities 1, warn_player_inactive_weeks 26. A `fail` blocks prospective predictions. Freshness never changes a probability: model probability and operational reliability are reported separately.

## 5. Inference contract

Input: canonical player IDs, surface, tournament level and ID, event start date, round, draw size, best-of, a verified scheduled start time if any, and the state snapshot. Output per match: P(A), P(B) (= 1 − P(A), checked to 1e-12), the surface-Elo baseline, model ID, frozen hash, artifact checksum, feature-state hash, state snapshot and source snapshot IDs, feature availability (which inputs used the frozen 0 fallback), freshness, `prediction_mode` (`historical_reconstruction`, `exploratory_retrospective`, `prospective`) and the generation time.

## 6. Runtime

Runtime and memory of the run that produced this report are in `runtime.json` (not part of the reproducible report).

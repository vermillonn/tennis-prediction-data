# Phase 2: leakage-safe historical Elo

Code: `src/tennis_data/elo/`. Config: `config/elo.toml`. Command: `uv run tennis-data elo-backtest`.
Results: `reports/elo/evaluation_report.md`.

## 1. Chronology

### What the data does and does not contain

The source has **no match dates and no match times**. The only temporal field is `tournament_date`,
the event's start date (usually a Monday). `match_num` is not chronological. The Elo engine never
creates or stores a match timestamp. Every time-like column it produces is a *batch label* derived
from `tournament_date`.

### Weekly batches

- **Week boundary:** Saturday 00:00 to Friday 24:00 (`[chronology].week_start = "saturday"`).
  `prediction_batch` is the start date (the Saturday) of the batch containing `tournament_date`.
  Saturday was chosen over Monday because weekend-dated events (404 Sunday-start and 330 Saturday-start
  events in the tour files, e.g. Sunday-start Masters and weekend Davis Cup ties) belong to the ATP week that
  *follows*. With a Monday
  boundary they would share a batch with the previous week's events, whose finals they overlap.
- **Freeze:** every match of a batch is predicted from the rating state left by the closes of all
  earlier batches. No result from the batch itself, or any later batch, is visible.
- **Close:** after all of a batch's predictions are made, the results *released* in that batch are applied.
- **Release:** a one-week event releases in its prediction batch. Events that span two weeks release
  one batch later (`release_batch = prediction_batch + 7 days`). The multi-week events are Grand
  Slams, Masters with a draw of at least 96, and the Olympics. This rule is conservative: a result
  is never applied before its event can have finished.

Consequence: a match's prediction uses only results from events released in strictly earlier
batches, never its own result or any result from its own event. The tests check this directly. The
truncation test shows that removing every later match leaves each batch's predictions bit-identical.
The perturbation test shows that flipping every result after a cutoff leaves every earlier
prediction bit-identical. Both checks were also run on the full real dataset (cutoff 2015-01-03,
6 random truncation points); both passed.

**This is an approximation.** True match-by-match replay is impossible without dates. Ratings are up
to one week (two weeks after long events) staler than a replay with true dates would allow. In
exchange, no batch can see the future. One residual assumption remains: a player does not play two
events at once. If a player played a match in a later-starting event *before* finishing an earlier
event, the release rule would still apply the earlier event first. In the validated tour data from
1995 on, only 3 player-weeks place a player in two non-Davis-Cup events of one batch (2012 Olympics).
In 1968-1984 the source's event dates are approximate and such overlaps are common. Those years
fall almost entirely in the unscored warm-up period.

### Order-independent batch update

At a close, every released, update-eligible match contributes

```
E_a   = 1 / (1 + 10 ** ((R_b - R_a) / 400))        (standard Elo expectation)
delta = K_match * (S_a - E_a),   S_a = 1 if A won else 0
R_a  += delta,   R_b -= delta
```

R, the match counts and K_match are all read from the state **frozen at the start of the close**.
Each player's total change is `math.fsum` of their deltas (exactly rounded, so the result does not
depend on summation order), and all changes are applied at once.

Assumptions: within one batch, a player's strength is constant, so all of the batch's results are
evaluated against the same pre-batch ratings. This is one simultaneous gradient step on the Elo
log-likelihood, not a chain of sequential steps whose outcome would depend on an arbitrary ordering.
The tests show bit-identical outputs under any shuffle of the input. A player with several matches
in a batch (up to 7 at a Slam) gets the sum of the individual steps. Sequential Elo would shrink
later steps after early wins; the batch update does not.

Zero-sum: each match adds `+delta` and `-delta`, so the rating total of a pool is conserved.
With K decay, `K_match = (K(n_a) + K(n_b)) / 2` is shared by both players for exactly this reason.
On the full data the mean final global rating is exactly 1500.

## 2. Ratings

- **Global Elo:** initial rating 1500. K is either constant (`[model.k].mode = "constant"`) or decaying,
  `K(n) = numerator / (n + offset) ** shape` with n = the player's prior rated matches.
- **Surface Elo:** independent pools for Hard, Clay, Grass and Carpet, each updated with its own
  ratings, expected scores and match counts. Carpet (indoor, discontinued on tour after 2009) is a
  separate pool by default; `carpet_policy = "hard"` rates it as Hard, and `"none"` gives carpet
  matches no surface rating. Matches with a missing surface (50 in the test period) update only
  the global rating, and the surface model predicts them with global ratings.
- **Surface initialization:** a player's first rating on a surface is 1500 (`surface_init = "constant"`)
  or their global rating at that moment (`"global"`).
- **Shrinkage:** the surface model predicts with
  `effective = w * surface_rating + (1 - w) * global_rating`, where `w = n_s / (n_s + m)` and n_s is
  the player's prior matches on that surface. `m = 0` is pure surface Elo; `w = 0` with no surface
  history; large `m` tends to global Elo.
- **Pools:** players are rated separately per `[[pools]]` entry (families x levels). The default is one
  pool, `atp_tour`, covering the tour-level files including Davis Cup and the Olympics. Lower-tier
  files (`qual_chall`, `futures`) are outside it. Adding a pool for them keeps their very different
  player population from diluting tour ratings.
- **Not modelled (yet):** inactivity decay, margin of victory, best-of-5 adjustment, home advantage.

## 3. Eligibility (`elo/eligibility.py`)

| predicate | meaning | default |
|---|---|---|
| `target_eligible` | match is a prediction target and is scored | `match_status == "completed"`, valid identity, in a pool |
| `update_eligible` | result updates ratings | `match_status == "completed"`, valid identity, in a pool |
| `serve_return_eligible` | serve/return stats usable later (Phase 3) | completed, `stats_status == "complete"`, no stats warning flags |

- Walkover, default, abandoned, unfinished and unknown are never eligible; the config loader rejects them.
- **Retirements** are excluded by default: the recorded winner is real, but the match was not played out.
  The backtest also runs a sensitivity experiment, `*_with_retirements`, which adds retirements to the
  *updates* only. Targets stay the same, so the comparison uses identical matches.
- **Quarantined rows** are never loaded. The engine reads only the validated table, and
  `assert_no_quarantined` raises if any quarantined `match_id` reaches it. This includes the 15 rows
  quarantined only for impossible serve statistics, whose results may well be right; excluding them is
  the conservative choice for version 1.
- Validated rows flagged `score_inconsistent_with_best_of` are eligible: the winner is not in doubt.

## 4. Walk-forward evaluation

- Warm-up: batches before 1991-01-01 build ratings and are never scored (84,607 targets).
- Validation: 1991-01-01 to 2009-12-31 (63,836 targets). This is the only period used to choose parameters
  (`[tuning]` grid). Each model is selected by validation log loss.
- Test: 2010-01-01 to 2026 (45,191 targets). Scored once with the selected parameters; no choice depends on it.
  A test checks that flipping every test-period result changes neither the selected parameters
  nor any pre-test prediction.
- Metrics: log loss, Brier score, accuracy (p = 0.5 counts as half correct), calibration in 10 bins with
  expected calibration error, and results by surface and by level.
- Comparisons are paired on identical matches. 95% intervals come from a bootstrap that resamples whole
  weekly batches, because predictions within a batch share one frozen state. The report says "improvement"
  only when the interval excludes zero.

## 5. Outputs

| path | content |
|---|---|
| `data/elo/predictions/model_name=<model>/part-0.parquet` | one row per target-eligible match, for `global_elo` and `surface_elo` |
| `data/elo/sensitivity_predictions/model_name=<model>_with_retirements/` | the same, for the retirement sensitivity runs |
| `data/elo/rating_history.parquet` | state after each batch close for each player it changed (surface model run): `batch, pool, player_id, rating_type` (`global` or surface), `rating_after`, `match_count_after` |
| `reports/elo/evaluation_report.md` | evaluation report |
| `reports/elo/tables/*.csv` | tuning grid, test/validation metrics, calibration, exclusions |
| `reports/elo/elo_summary.json` | selected parameters, configuration hashes, input and output SHA-256 checksums |

DuckDB views (`tennis-data query`): `elo_predictions`, `elo_sensitivity_predictions`, `elo_rating_history`.

Prediction columns:

| column | meaning |
|---|---|
| `match_id`, `tournament_id`, `tournament_date`, `tournament_level`, `round`, `surface`, `match_status` | from the validated table |
| `prediction_batch` | start date of the batch whose frozen state produced the prediction (a batch label, not a match time) |
| `release_batch` | batch whose close applies this match's result |
| `evaluation_period` | `warmup`, `validation`, `test` or `after_test` |
| `pool`, `surface_key` | rating pool, and surface pool used (null = none) |
| `player_a_id`, `player_b_id` | Phase 1 winner-independent ordering |
| `player_a_elo_before`, `player_b_elo_before`, `global_match_count_a/b` | frozen global state |
| `surface_elo_a/b_before`, `surface_match_count_a/b` | frozen raw surface state (null if no surface key) |
| `surface_effective_elo_a/b` | shrunk ratings used by the surface model |
| `predicted_probability_a` | the model's P(player A wins) |
| `target_a_wins` | outcome (only for scoring, never an input to the row's own prediction) |
| `model_name`, `model_version`, `configuration_hash` | `configuration_hash` covers parameters, chronology, eligibility, pools, engine version and source commit |

The surface columns are present in the `global_elo` rows too, for reference; the global model does not use them.

Example: ratings as of a date (state after every batch closed before it):

```sql
SELECT player_id, arg_max(rating_after, batch) AS rating
FROM elo_rating_history WHERE rating_type = 'global' AND batch < DATE '2024-01-06'
GROUP BY player_id ORDER BY rating DESC LIMIT 10;
```

## 6. Known limitations

- Weekly batching approximates chronology (Section 1). Ratings are staler than true replay would allow,
  and the multi-week rules are by level and draw size, not observed event lengths.
- Simultaneous updates can overshoot for players with many matches in one batch.
- The global model is overconfident at the extremes (test: predicted 0.85 -> observed 0.80). The surface model is
  better calibrated but has the same tendency. No recalibration is applied in Phase 2.
- Retirements and stats-quarantined matches are left out of the default ratings. The sensitivity run shows that
  including retirements helps slightly, but that was not part of the pre-registered default.
- The tuning grid is small and fixed in the config, and parameters are not re-estimated over time.
- Rank and ranking-point features are deliberately not used.

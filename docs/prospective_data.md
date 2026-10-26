# Phase 4A: current data, historical backfill and inference readiness

Phase 4A prepares the frozen M4A model (configuration `5e72010100065889`, frozen hash `bc4e5c870c6e21b9`, raw
output) for upcoming matches. M4A, Elo, the Phase 3A feature formulas and the selection are unchanged. Phase 4A
adds a way to get new verified results into the player state and a verified inference path.

```bash
uv run tennis-data prospective-import export.csv --format sackmann_csv        # store an authorized file immutably
uv run tennis-data prospective-import feed.json --format tennis-fixture-provider-1
uv run tennis-data prospective-backfill                                         # identity -> reconciliation -> Phase 1
uv run tennis-data prospective-run                                              # audit, backfill, replay, reports
uv run tennis-data prospective-predict upcoming.json --mode prospective        # refuses stale or late requests
```

## 1. Components (`src/tennis_data/prospective/`)

| module | role |
|---|---|
| `schema.py` | versioned provider-neutral frames (`provider-match-1`): players, results, fixtures; structural checks |
| `providers/raw_store.py` | immutable content-addressed raw snapshots with retrieval time, redacted parameters, snapshot IDs |
| `providers/http.py` | keys from environment variables, rate limiter, bounded exponential backoff on 429 and 5xx, no retry on 401 and 403 |
| `providers/fixture.py` | documented JSON exchange format, used by CI and by converted authorized exports |
| `providers/sackmann_csv.py` | authorized Sackmann-format CSV import (archive IDs, `sackmann_v1` statistics) |
| `providers/sportradar.py` | raw retrieval only; **unverified**, the response mapping is deliberately not implemented |
| `identity.py` | provider player ID → canonical ID with categories and persisted decisions |
| `reconciliation.py` | versions, conflicts, precedence and tournament identity; conversion to Sackmann rows |
| `backfill.py` | parse store → identity → reconcile → **unchanged Phase 1 validation** → ledger → state snapshot |
| `state.py` | Phase 2 Elo and Phase 3A engines over snapshot + pending rows |
| `inference.py` | artifact integrity, Phase 3B derived inputs, frozen scaling and coefficients, P(A) + P(B) = 1 |
| `freshness.py` | operational reliability checks with warning and failure thresholds |
| `replay.py` | reproduces published Phase 2, 3A and 3B values through the new path |
| `gap.py` | historical gap audit |

The only change to earlier phases is a refactor: `pipeline.validate_frame` was extracted from Phase 1
`validate`. Behaviour is the same, so new rows pass exactly the Phase 1 rules. The Phase 1 to 3B outputs are
byte-identical after the change (verified).

## 2. Time fields: release-time assumptions versus acquisition timestamps

There are two different kinds of time in this system, and they are never substituted for each other.

**A. Historical release-time assumptions (inferred, not observed).** The source has no match times. When a result
"became available" is an *assumption* derived from the event start date by the frozen Phase 2 rule:

| field | meaning | how it is obtained |
|---|---|---|
| `prediction_batch` | weekly batch (Saturday–Friday) containing the event start date | inferred from `event_start_date` |
| `release_batch` | batch whose close releases the result (one week later for two-week events) | inferred (`event_weeks`) |
| `state_eligible_at` | when the result may update player state: 00:00 UTC on the day after the release batch closes | **inferred**; ledger `eligibility_basis = inferred_weekly_batch` |

These are conservative bounds, not observations. The historical results were not necessarily known at these times,
only no earlier than the rule allows. They hold for the archive and for every backfilled row.

**B. Source and acquisition timestamps (recorded, never inferred).**

| field | meaning | how it is obtained |
|---|---|---|
| `event_start_date` | tournament start date | stated by the source |
| `event_start_at` | match start time | stated by the source, otherwise null (never invented) |
| `event_completed_at` | match end time | stated by the source, otherwise null |
| `source_published_at` | when the source says it published this version | stated by the source, otherwise null |
| `ingested_at` (`retrieved_at`) | when *this system* downloaded or imported the snapshot | recorded by the raw store at acquisition |
| `source_version` | the source's record version (corrections have higher versions) | stated by the source (default 1) |

All of these are timezone-aware UTC. A timestamp without an offset is rejected rather than assumed. An acquisition
time says when *we* learned a result, not when it was historically knowable. A May–October 2026 result acquired in
November 2026 has `ingested_at` in November and an inferred `state_eligible_at` in its own week. It is used only as
historical reconstruction (section 3), and never as evidence that a forecast made then could have known it.

Phase 4A uses only the inferred release times (A) to decide state updates. The recorded source timestamps (B) are
stored for audit and for a possible future timestamp-based policy, which would need its own version and tests.

## 3. Chronology

Phase 4A keeps the frozen Phase 2 policy (`weekly_batch_v1`). Predictions use the state frozen at the start of the
weekly batch (Saturday to Friday). A result updates the state at the close of its release batch. Grand Slams, 96+ draw
Masters and the Olympics release one week later. Provider timestamps are stored but **not** used to release results
earlier, because mixing timestamp-based and weekly updates would change what the model saw historically. The ledger
records `eligibility_basis = inferred_weekly_batch` for every row. A timestamp policy would need its own documented
and tested version.

Backfill is historical reconstruction. A May–October 2026 result discovered now updates the state for later matches.
It is never treated as known on its event date, and never presented as a forecast (`prediction_mode =
historical_reconstruction`).

## 4. Identity

See `reports/prospective/identity_reconciliation.md`. The categories are verified, high_confidence, ambiguous and
unmatched. Ambiguous identities are never resolved automatically. Matches involving them are held out of the state
(`identity_unresolved`) until a reviewer adds an entry to `config/identity_overrides.toml`. New players get stable
IDs from 9,000,000 upward. The registry is append-only.

## 5. Reconciliation and precedence (`rec-1`)

- **Precedence**, highest first: `historical_archive`, `sackmann_csv`, `sportradar`, `fixture`.
- **The archive is never overwritten.** A new record that matches an archive match with the same outcome is
  dropped as a duplicate. A different outcome or score is quarantined.
- **Duplicates and versions.** Identical copies collapse into one. A higher `source_version` supersedes the earlier
  one, and both stay in `versions`. A different record under the same version is quarantined.
- **Cross-source disagreement.** Different winners, statuses, or start dates more than 7 days apart quarantine every
  record involved. If the outcomes agree but the statistics differ, the highest-precedence record is kept and the
  difference is logged.
- **Statistics.** Only records with `stats_definition = sackmann_v1` keep their statistics. Others keep only the
  outcome until their convention has been tested.
- **Match keys and numbers.** The match key is (tournament, round, unordered player pair). `match_num` is a hash of
  the key, so IDs do not change when other matches arrive.

## 6. Corrections and replay

The state is a pure function of the accepted table and the frozen configurations. A correction therefore rebuilds it
deterministically from the start. That way, downstream effects of a changed result are never skipped, and there are
no "latest rating only" patches. `replay_from_batch` (in `backfill/performance.json`) records the earliest batch whose
state can differ. Rebuilding from the stored table is the checkpoint restoration. It is verified against the recorded
`state_snapshot_id`.

## 7. Inference

An upcoming match enters the engines as a *pending* row: a target that never updates Elo or statistics and has no
outcome. This is the mechanism every historical prediction used, so M4A receives the same kinds of information.

- **Inputs** are the Phase 3B `DERIVED` expressions. The only missing-data rule is the frozen one (a missing
  difference becomes 0, plus the decay-missingness indicator).
- **Scaling and coefficients** come from `production_model.json`. Its SHA-256, the frozen hash and the modeling
  configuration hash are checked first.
- **Symmetry.** P(B) is computed from the negated inputs and must equal 1 − P(A) to within 1e-12.

**Prediction modes**

- `prospective`: only while the status gate is ENABLED (currently DISABLED), for matches starting after 2026-10-25,
  generated before the match start (the verified time,
  otherwise 00:00 UTC of the event start date), from a state with no later results, and with freshness not `fail`.
  Phase 4A does not persist predictions; the immutable ledger is Phase 4B.
- `exploratory_retrospective`: anything computed after the fact.
- `historical_reconstruction`: replays.

## 8. Freshness

Seven checks are reported per match: the latest eligible result, the latest usable point statistics, the age of the
opponent fit, the number of weekly batches with no results, unresolved identities, the age of the source snapshot,
and the players' recent statistics. Each is graded `ok`, `warning` or `fail` against `[freshness]`. Freshness never
changes a probability. A `fail` blocks prospective use. The off-season counts as missing weeks, which is conservative
and produces warnings in January.

## 9. Fallback design

1. Primary licensed API for schedules and results, once verified.
2. Secondary licensed source for verification and gaps.
3. Manual import of authorized Sackmann-format or exchange-format files.
4. Local immutable raw snapshots, so every state can be rebuilt without any provider.

All sources pass the same identity, Phase 1 validation and reconciliation rules. Statistics with an untested
counting convention never update the serve/return state.

## 10. Provider verification status

**No real-provider integration is verified.** "Verified" below means only that the code is tested against
documented or synthetic fixtures. None of them has been run against a live provider.

| adapter | real access tested | response format verified | statistics convention verified | may update state |
|---|---|---|---|---|
| `sackmann_csv` (manual import) | no new file was available | yes (archive format) | yes (`sackmann_v1`) | yes, after identity and Phase 1 validation |
| `tennis-fixture-provider-1` (exchange format) | n/a (format for converted exports and CI) | yes (documented here) | only if a record declares `sackmann_v1` | yes, after identity and Phase 1 validation |
| `sportradar` (HTTP) | **no** (host blocked; no key) | **no**: `normalize` raises `NotVerifiedError` | **no** | no |
| tennisdata.app, tennis-api.com, others | **no** (hosts blocked) | no adapter | no | no |

## 11. Status and limitations (2026-10-26)

- **Historical backfill: INCOMPLETE.** **Prospective predictions: DISABLED.** Both are set in
  `config/prospective.toml` `[status]` and enforced in code. `prospective` mode refuses every request while the
  gate is DISABLED, and the configuration cannot enable predictions while the backfill is INCOMPLETE. Freshness
  checks are never bypassed.
- **No current source was accessible**, because the environment's network policy blocks every provider host. The
  archive mirror has no data after 2026-05-25, and the 22-week gap to the prospective start is **not filled**
  (`reports/prospective/backfill_coverage.md`, `inference_readiness.md`).
- **Real-provider integrations are unverified** (section 10). The Sportradar response mapping is not implemented.
- Identity matching quality on a real provider is unmeasured.
- The last archive event (Roland Garros 2026) has no serve statistics.
- Corrections rebuild the state deterministically and never modify recorded predictions. Phase 4A itself stores no
  predictions; the immutable prediction ledger is Phase 4B.

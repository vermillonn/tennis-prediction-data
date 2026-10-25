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
| `providers/sportradar.py` | raw retrieval only; the response mapping is deliberately not implemented until verified |
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

## 2. Time fields

| field | meaning | source |
|---|---|---|
| `event_start_date` | tournament start date (the only date the archive has) | provider |
| `event_start_at` | match start time | provider, only if stated; never inferred |
| `event_completed_at` | match end time | provider, only if stated |
| `source_published_at` | when the provider published this version | provider, if stated |
| `ingested_at` | when this system retrieved or imported the snapshot | raw store |
| `state_eligible_at` | when the result may update player state | **inferred**: 00:00 UTC on the Saturday after the release batch |
| `source_version` | provider record version (corrections have higher versions) | provider |

All times are timezone-aware UTC. A timestamp without an offset is rejected rather than assumed.

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

- `prospective`: only for matches starting after 2026-10-25, generated before the match start (the verified time,
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

## 10. Status and limitations (2026-10-25)

- **No current source was accessible**, because the environment's network policy blocks every provider host. The
  archive mirror has no data after 2026-05-25. The 22-week gap to the prospective start is **not filled**, so the
  system is **not ready** for prospective predictions (`reports/prospective/inference_readiness.md`).
- The Sportradar response mapping is not implemented, because it was not verified.
- Identity matching quality on a real provider is unmeasured.
- The last archive event (Roland Garros 2026) has no serve statistics.

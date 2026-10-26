# Phase 4A verification record

Hand-written record of the final verification on 2026-10-26, implementation commit `480596a`. The generated reports
in this directory come from the same runs.

## Status

- Historical backfill (May–October 2026): **INCOMPLETE**. 0 records are available for the 22 missing weekly batches.
- Prospective predictions: **DISABLED** by the `[status]` gate in `config/prospective.toml`, enforced in code.
- Real-provider integrations: **UNVERIFIED**. External provider connectivity is unresolved: the environment's
  network policy blocks every provider host (`source_feasibility.md`).
- No immutable prospective prediction ledger exists yet (Phase 4B). Phase 4A stores no predictions.

## Results

| check | result |
|---|---|
| automated tests | 327 passed (258 existing + 69 Phase 4A) |
| two full real-data runs (`tennis-data prospective-run`) | byte-identical: 18 of 18 deterministic artifacts (`runtime.json` and `performance.json` excluded) |
| historical replay, 274 cases, 2002–2025 | max M4A error 1.1e-16, max surface-Elo error 0.0, max symmetry error 2.2e-16 (tolerance 1e-12) |
| state reconstruction | 193,634 Phase 2 Elo probabilities exact; Phase 3A feature table identical (193,634 × 142) |
| frozen model | hash `bc4e5c870c6e21b9`, configuration `5e72010100065889`, production artifact SHA-256 `24be84b7af161dc5` |
| Phase 1–3B outputs | 136 data files (validated, quarantine, Elo, features, modeling) byte-identical before and after |
| Phase 1–3B reports | no difference from `main` |
| Phase 1 `validate_frame` refactor | full Phase 1 rebuild byte-identical (validated and quarantine) |
| runtime of one full run | about 227 s (replay engine run about 219 s), peak RSS about 2.4 GB |

## Production inference fixture tests (`tests/test_prospective_production.py`)

Byte-exact copies of the frozen and production M4A artifacts, with their published checksums, are run through the
inference path. The tests cover:

- artifact loading and frozen and configuration hashes;
- the feature order of the frozen specification;
- preprocessing: frozen scaling, the 0 fallback and the missingness indicator;
- golden probabilities computed outside the package;
- player-order symmetry and independence from state column order;
- refusal of any artifact or manifest change, or a wrong expected hash;
- stale-state rejection for a 2026 match;
- the status gate, and that the configuration cannot enable predictions while the backfill is incomplete;
- timing guards.

All pass.

Related tests elsewhere:

- historical replay: `tests/test_prospective_pipeline.py::test_historical_replay_reproduces_published_predictions`;
- deterministic rebuild after late corrections, with recorded predictions left untouched:
  `test_late_corrections_rebuild_state_and_never_touch_recorded_predictions`.

## Remaining blockers

1. No authorized source for May 25 – Oct 25, 2026 results (the archive mirror ends at 2026-05-25).
2. Provider hosts blocked by the environment's network policy; no API key; licences not reviewed.
3. Sportradar (or other) response mapping and statistics convention unverified; no live schedule feed.
4. Freshness fails for any match after the prospective start (latest result 140 days old, statistics 154 days,
   21 empty weeks).
5. Status gate DISABLED until items 1–4 are resolved by a reviewed change.

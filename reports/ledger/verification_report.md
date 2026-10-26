# Phase 4B ledger verification report

Written by hand from measured runs on 2026-10-26 (branch `claude/phase4b-ledger`). All numbers are measured, not
estimated.

**Status:**
- Prospective predictions **DISABLED**; May–October 2026 backfill **INCOMPLETE**. Both gates are unchanged.
- **No external checkpoint has been published.** The ledger's integrity can only be anchored once checkpoints are
  stored outside this machine (`docs/ledger_threat_model.md`).
- **No genuine prospective prediction exists in any ledger.**

## 1. Automated tests

**386 passed:** 327 existing, plus 59 Phase 4B. They also pass with `data/` removed, as in CI.

| file | tests | covers |
|---|---|---|
| `test_ledger_core.py` | 16 | canonical JSON bytes, timestamp form, hash formulas, init, sequential commits, retrieval, identical-input determinism, UPDATE/DELETE blocked (events, metadata), out-of-chain inserts blocked, invalid probability blocked by CHECK, tampering detection, full-rewrite detection only via checkpoint, checkpoint export/verify, export/import |
| `test_ledger_predictions.py` | 18 | idempotent re-submission, conflict rejection and logging, forecast slots, orientation, invalid probabilities (NaN, out of range, sum ≠ 1), frozen/configuration hash checks, missing state, PROSPECTIVE refused (injected and system clock), mode namespaces, post-result rules, ledger-assigned commit time, every prospective timing rule |
| `test_ledger_lifecycle.py` | 13 | postponement, cancellation, voiding with and without evidence, no outcome-based void, player replacement, result correction chain, non-binary outcomes, timing statuses with late actual start |
| `test_ledger_concurrency.py` | 5 | 4 process writers, concurrent identical requests, failed commit rollback, crash inside a transaction, clock regression |
| `test_ledger_integration.py` | 5 | real frozen 2026 M4A artifacts → state rows → commit → receipt; PROSPECTIVE refused before inference; no receipt without commit; tampered receipt; full Phase 4A path on a synthetic world (SYNTHETIC + HISTORICAL_REPLAY) |
| `test_ledger_cli.py` | 2 | every CLI command end to end; read-only DuckDB views |

## 2. Tampering detection

To simulate a privileged attacker, the tests drop the protective triggers or edit exported event lists.

| tampering | detected by | result |
|---|---|---|
| prediction payload edited (probabilities changed) | `event_hash_mismatch`, `denormalized_column_mismatch` | detected |
| event deleted | `sequence_gap_or_reorder`, `previous_hash_mismatch` | detected |
| events reordered | `sequence_gap_or_reorder`, `previous_hash_mismatch` | detected |
| predecessor hash altered | `previous_hash_mismatch` | detected |
| event hash altered | `event_hash_mismatch` | detected |
| duplicate event id / duplicated event | `duplicate_event_id`, `duplicate_idempotency_key` | detected |
| export file edited | `import_jsonl` refuses ("does not verify") | detected |
| checkpoint body altered | checkpoint hash mismatch | detected |
| history truncated below a checkpoint | "history truncated" | detected |
| **entire chain rewritten and re-hashed** | local verifier: **not detected** (internally consistent); `ledger-verify-checkpoint` against the earlier checkpoint: "history rewritten" | detected **only with a trusted checkpoint** |
| ordinary `UPDATE` / `DELETE` / off-chain `INSERT` | SQLite triggers abort | prevented |

## 3. Concurrency and recovery

| test | result |
|---|---|
| 4 processes × 40 commits at once (test) | 160 events, sequences 1–160 with no gaps or duplicates, chain valid |
| 4 processes × 250 commits at once (benchmark) | 1,000 events in 2.19 s wall time, chain valid, all writers exit 0 |
| 8 threads submitting the same request | exactly 1 committed, 7 returned `existing`, 0 errors |
| COMMIT failure injected | transaction rolled back, head unchanged, next commit reuses the sequence, chain valid |
| process killed (`os._exit`) inside a transaction after INSERT | the event never became visible; ledger verified; the next commit is seq 4 |
| writer clock behind the chain head | commit refused ("clock regression"); verifier also reports regressions |

## 4. Inference → ledger integration

**Synthetic fixtures (CI)**

- The real frozen 2026 production M4A artifacts (`tests/fixtures/m4a_production`) priced 6 hand-made states and
  committed them as SYNTHETIC.
- Committed probabilities match the independently computed golden values to within 1e-12 under the ledger's player
  orientation. Every receipt verifies.
- Re-submission returns `existing`. PROSPECTIVE is refused before inference. No receipt is returned when the commit
  fails.

**Synthetic matches on the real archive state** (scratch ledger, not committed)

- Two synthetic upcoming matches went through the unchanged Phase 4A path: `load_state_snapshot` →
  `compute_state` → `assess` → `load_production` → `predict_state` → `commit_prediction`. The state snapshot was
  `3253426224f7ebc0`.

| check | result |
|---|---|
| receipts | 2 committed; both verified independently (`verify_receipt`) |
| identical re-submission | 2 × `existing` (no new events) |
| max \|P(A) + P(B) − 1\| | 0.0 |
| frozen hash in records | `bc4e5c870c6e21b9` |
| freshness recorded | `fail` (state ends 2026-05-25), recorded rather than hidden; SYNTHETIC records may carry it |
| PROSPECTIVE attempt | rejected: "PROSPECTIVE predictions are DISABLED (backfill INCOMPLETE)" |
| chain / checkpoint verification | valid / valid |
| inference + commit time | 223 s (one full engine pass over the archive) |

**Historical replay:** the synthetic-world test commits a HISTORICAL_REPLAY record in the `replay:` namespace.
Phase 4A already verified replay accuracy against the published Phase 3B predictions (max error 1.1e-16).

## 5. Checkpoints

- Checkpoints are deterministic: the same ledger gives the same checkpoint, and the export time goes in a separate
  `.meta.json`.
- A chained checkpoint (`previous_checkpoint_hash`) verifies after the ledger grows.
- An altered checkpoint, a truncated history and a rewritten history are each rejected.
- `LEDGER_CHECKPOINT_CREATED` events record checkpoints inside the chain.
- Publication to an external system is **not configured**.

## 6. Performance

The benchmark (`tennis-data ledger-benchmark --events 10000`) used a synthetic ledger of 8,000 predictions and 2,000
results. Settings: system clock, WAL, `synchronous=FULL`, local disk of the cloud container.

| metric | measured |
|---|---|
| single append latency (prediction) | median 2.06 ms, p95 2.66 ms, max 13.0 ms |
| sequential append throughput | 561 events/s |
| full verification of 10,000 events | 1.62 s |
| export of 10,000 events (JSONL + Parquet) | 0.74 s |
| storage | about 1,980 bytes per event (SQLite + WAL) |
| 4 concurrent writers, 1,000 events | 2.19 s wall time, chain valid |

The ledger favours integrity over throughput. Each append fsyncs and replays the match's own history to check its
lifecycle.

## 7. Regression and frozen-model checks

- Phase 1–3B data files: 136/136 byte-identical. Phase 4A data files: 13/13 byte-identical. No report changed.
- Code changes outside `src/tennis_data/ledger/` are additive only: new CLI commands in `cli.py`, an optional
  `ledger_views` in `storage/query.py`, and a README section.
- Frozen M4A: hash `bc4e5c870c6e21b9`, configuration `5e72010100065889`. The production artifact checksums are
  verified on every load.

## 8. Remaining limitations

1. No external checkpoint has been published and no trusted timestamping is configured. The local hash chain alone
   cannot prove when an event was created; it shows order and integrity relative to a trusted head only. A full local rewrite cannot
   be detected without one.
2. The commit time is the host's system clock. Strong evidence of time needs an external timestamp.
3. Prospective commitments are refused (gate DISABLED, backfill INCOMPLETE), so the PROSPECTIVE path has been tested
   only through its rule function (`check_prospective_timing`) and the refusal.
4. Schedule and actual-start sources are fixtures. No live provider is verified (Phase 4A).
5. A single local SQLite file. Multi-host deployment and replicated storage are out of scope.
6. Result ingestion records outcomes only. Performance evaluation is Phase 4C.

# Phase 4B: append-only prediction ledger

The ledger records pre-match forecasts so that anyone can later check:

- what was predicted, and with which model and state;
- which data sources the prediction used;
- when it was committed;
- whether the recorded history has changed since.

It is a local SQLite event log with a SHA-256 hash chain and exportable checkpoints. It makes ordinary records
immutable and unauthorized modification detectable. It does **not** make tampering impossible
(`docs/ledger_threat_model.md`).

**Status:** prospective predictions are **DISABLED** and the May–October 2026 backfill is **INCOMPLETE**
(`config/prospective.toml [status]`). The ledger refuses every PROSPECTIVE commitment until a reviewed change enables
the gate. **No external checkpoint has been published.**

```bash
uv run tennis-data ledger-init
uv run tennis-data ledger-commit-synthetic request.json        # SYNTHETIC only
uv run tennis-data ledger-record MATCH_RESULT_RECORDED synthetic:m1 result.json
uv run tennis-data ledger-verify
uv run tennis-data ledger-lookup synthetic:m1
uv run tennis-data ledger-events synthetic:m1
uv run tennis-data ledger-export                              # data/ledger/export/events.{jsonl,parquet}
uv run tennis-data ledger-checkpoint [--previous cp.json] [--record-event]
uv run tennis-data ledger-verify-checkpoint cp.json
uv run tennis-data ledger-conflicts
uv run tennis-data ledger-benchmark --events 10000
uv run tennis-data query "SELECT mode, count(*) FROM ledger_predictions GROUP BY ALL"   # read-only, from the export
```

## 1. Modules (`src/tennis_data/ledger/`)

| module | role |
|---|---|
| `hashing.py` | canonical JSON, timestamps, genesis/event/checkpoint hashes |
| `storage.py` | SQLite schema, constraints, append-only and chain triggers, durability settings |
| `events.py` | payload schemas and validation for the seven event types |
| `lifecycle.py` | deterministic per-match state machine and timing statuses |
| `writer.py` | trusted writer: validation, mode gate, timing, idempotency, atomic append |
| `service.py` | Phase 4A integration: state → freshness → frozen M4A → commit → receipt |
| `reader.py`, `verification.py`, `checkpoint.py`, `export.py` | read, verify, checkpoint, export and import |
| `cli.py`, `benchmark.py` | commands and measured performance |

## 2. Event schema (`ledger-1`)

Each event is one row of `events`:

| column | meaning |
|---|---|
| `seq` | ledger sequence 1, 2, 3, … (no gaps) |
| `event_id` | stable unique id: `pred-<first 32 hex of idempotency key>` for predictions; `<type prefix>-<hash of ledger id, sequence, type, match, payload>` for others |
| `event_type` | `PREDICTION_COMMITTED`, `MATCH_METADATA_CORRECTED`, `MATCH_CANCELLED`, `MATCH_POSTPONED`, `MATCH_RESULT_RECORDED`, `PREDICTION_VOIDED`, `LEDGER_CHECKPOINT_CREATED` |
| `schema_version` | `ledger-1` |
| `recorded_at` | ledger commit time from the writer's clock (UTC, `YYYY-MM-DDTHH:MM:SS.ffffffZ`) |
| `match_id` | canonical internal match id (null for checkpoints) |
| `mode`, `idempotency_key`, `probability_a/b` | predictions only (denormalized from the payload, checked by the verifier) |
| `ref_event_id` | referenced earlier event (voids) |
| `payload`, `refs` | canonical JSON |
| `previous_hash`, `event_hash` | the chain |

**`PREDICTION_COMMITTED` payload.**

- `match`: `match_id`, `provider_match_ids` (external ids, kept separately), `player_a_id`, `player_b_id`,
  `tournament_id`, `surface`, `round`, `best_of`, `scheduled_start_at`, `scheduled_start_verified`,
  `schedule_snapshot_id`, `schedule_retrieved_at`.
- `prediction`: `model_id`, `model_version`, `frozen_hash`, `configuration_hash`, `feature_schema_version`,
  `state_snapshot_id`, `feature_state_hash`, `source_snapshots`, `prediction_created_at`, `prediction_cutoff`,
  `probability_a`, `probability_b`, `baseline_elo_probability_a`, `feature_availability`, `freshness_status`,
  `freshness`.
- `audit`: `mode`, `forecast_slot`, `inference_engine_version`, `git_commit`, `ledger_schema_version`,
  `idempotency_key`, `content_hash`, `orientation`, `clock` (`system` or `injected`), `prospective_gate`, `timing`.

The ledger commit timestamp is the event's `recorded_at`, and the prediction event hash is its `event_hash`.

**Probability checks.** Both probabilities must be finite floats in [0, 1], with |P(A) + P(B) − 1| ≤ 1e-12.

**Player orientation** (`phase1-sha256-match-id`): player A is the lower player id when the first byte of
SHA-256(match id, without the `synthetic:` or `replay:` prefix) is even, and the higher id otherwise. This is the
Phase 1 rule (`normalization.ordering.assign_players`). It is independent of who wins, and the writer refuses any
other orientation. `orient()` swaps a request into this order.

## 3. Canonical hashing

- Canonical JSON: UTF-8 encoded; `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  allow_nan=False)`.
  - Allowed types: `null`, `bool`, `int`, finite `float`, `str`, `list`, and `dict` with string keys.
  - Floats are written as Python's shortest round-trip `repr`, and `-0.0` is normalized to `0.0`.
  - Every other type, and NaN or ∞, is rejected.
- Timestamps take exactly one form, `YYYY-MM-DDTHH:MM:SS.ffffffZ` (UTC). Naive datetimes are rejected.
- `genesis = SHA256("tennis-ledger/genesis/v1\0" ‖ ledger_id)`.
- `event_hash = SHA256("tennis-ledger/event/v1\0" ‖ canonical(body) ‖ "\0" ‖ previous_hash)`. `body` is every stored
  field except `event_hash`: seq, event id, type, schema, recorded time, match, mode, idempotency key, probabilities,
  ref, payload, refs, and previous hash.
- `checkpoint_hash = SHA256("tennis-ledger/checkpoint/v1\0" ‖ canonical({format, ledger_id, schema_version,
  sequence, head_hash, previous_checkpoint_hash}))`.

Tests pin these rules: the exact canonical bytes, the hash formulas, independence from key order, and identical
hashes for identical inputs under a fixed clock.

## 4. Prediction modes

| mode | match-id namespace | meaning | counts as prospective evidence |
|---|---|---|---|
| `SYNTHETIC` | `synthetic:` | fixtures and test states | never |
| `HISTORICAL_REPLAY` | `replay:` | already-played matches, for verifying inference | never |
| `PROSPECTIVE` | canonical ids only | before a real upcoming match | only this mode; **currently refused** |

The namespaces make the modes impossible to confuse. A `synthetic:` id cannot be committed as a replay and vice
versa, and neither can be committed as PROSPECTIVE.

The PROSPECTIVE refusal sits in `LedgerWriter._precheck` (the trusted service), and again in `service._gate`. There
is no bypass flag. A PROSPECTIVE commit additionally requires:

- the trusted `SystemClock` (an injected clock is refused);
- a verified scheduled start;
- a prediction cutoff no later than the start, with commit time < cutoff and commit time < start;
- a creation time no later than the commit time;
- a schedule snapshot at most `max_schedule_age_hours` old;
- freshness not `fail`.

`check_prospective_timing` implements these rules and is tested directly, because the gate keeps the full path closed.

## 5. Timing rules

The ledger never trusts a caller-supplied time as proof. The commit time (`recorded_at`) and the creation time used by
`service.py` both come from the writer's clock. The writer also refuses a commit time earlier than the chain head
(clock regression). The verifier reports commit-time regressions.

For each prediction, three statuses are derived and never rewritten:

1. `generated_before_scheduled_start`: creation time is before the scheduled start recorded in the prediction.
2. `committed_before_scheduled_start`: ledger commit time is before that scheduled start. This is also stored in
   `audit.timing` at commit time.
3. `verified_before_actual_start`: null until a sourced `MATCH_METADATA_CORRECTED(kind=actual_start)` arrives, then
   commit time < actual start.

Without a verified scheduled start, a record is never certified as pre-match.

## 6. Idempotency

The idempotency key is SHA-256 of {match id, mode, forecast slot, model id, frozen hash, configuration hash, schedule
snapshot id}. Forecast slots come from a fixed list in `config/ledger.toml`: `pre_match`, `t_minus_24h`,
`t_minus_1h`.

- **Same key, same content** (all fields except creation time and git commit): the existing record is returned, with
  status `existing`.
- **Same key, different content:** `IdempotencyConflict`. The original is kept, and the attempt is appended to
  `idempotency_conflicts` (append-only), which is listed by `ledger-conflicts`.
- **Concurrent identical requests** commit exactly once. The database UNIQUE constraint on the key settles races.

Re-submitting cannot be used to pick a more favourable number after results are known. Each slot holds one
forecast, conflicting attempts are recorded, and SYNTHETIC predictions after a result are refused.

## 7. Lifecycle, corrections and voiding (`lifecycle.py`)

**Rules per match, applied in sequence order:**

- A prediction cannot be committed for a cancelled or superseded match. Once a result is recorded, only
  `HISTORICAL_REPLAY` predictions are accepted.
- **Postponement** adds a new scheduled start. The original start and the forecast stay unchanged.
- **Cancellation** is terminal: no result and no new prediction can follow it.
- A **players** correction supersedes the match. The replacement pairing gets its own match id and its own forecasts,
  and the old forecast stays with the old match.
- **Result corrections** must reference the current result (`supersedes_event_id`) and give a reason. The original
  result stays in the history.
- Result statuses are completed, retired, walkover, default, abandoned and unknown. Abandoned and unknown have no
  winner; nothing is forced into win/loss. A result never changes a prediction's probability, hashes or commit time.

**Voiding.** A void needs a predefined reason code, the actor or process, and evidence events from the same match:

| reason code | required evidence event |
|---|---|
| `match_cancelled` | `MATCH_CANCELLED` |
| `players_changed` | `MATCH_METADATA_CORRECTED(players)` |
| `committed_after_actual_start` | an actual start at or before the commit time |
| `walkover_no_play` | `MATCH_RESULT_RECORDED(walkover)` |

There is no outcome-based reason. Voided predictions remain in every query and export.

## 8. Concurrency and crash recovery

- **One writer at a time.** Each append is a single `BEGIN IMMEDIATE` transaction: validate, check idempotency, read
  the head under the write lock, assign seq, hash, insert, commit. Other writers wait up to `busy_timeout_ms`.
- **Database-level chain check.** The `BEFORE INSERT` trigger also refuses any event that isn't next in sequence on
  the current head.
- **Durability.** WAL journal with `synchronous=FULL`: every commit is fsynced. An exception rolls the whole
  transaction back.
- **Recovery after a crash or power loss.** Reopen the database: SQLite recovers or rolls back the WAL. Then run
  `ledger-verify`, and `ledger-verify-checkpoint` against the latest external checkpoint. An uncommitted event never
  becomes visible; this is tested by killing a process inside a transaction.
- **Corrupted file.** Restore from the latest export (`import_jsonl` re-verifies the chain) and compare the head with
  the latest checkpoint.

## 9. Checkpoints

A checkpoint is a deterministic JSON file `{format, ledger_id, schema_version, sequence, head_hash,
previous_checkpoint_hash, checkpoint_hash}`, plus a separate `.meta.json` holding the operational export time.
`ledger-verify-checkpoint` checks:

- the checkpoint's own hash;
- that the ledger still contains exactly the committed history up to `sequence` (detects truncation and rewriting);
- that everything after it is a valid continuation.

`--record-event` also writes `LEDGER_CHECKPOINT_CREATED` into the chain.

A hash chain is tamper-evident only relative to a head recorded where the local writer cannot change it. Copying
checkpoints to an independent system makes later rewriting detectable, for example:

- a separate repository with protected history;
- an RFC 3161 timestamping service;
- a signed message to a third party.

Phase 4B implements export and verification only. **No external publication is configured.**

## 10. Integration with Phase 4A

```
validated upcoming match (UpcomingMatch, schedule snapshot)
  → prospective.backfill.load_state_snapshot / prospective.state.compute_state   (unchanged engines)
  → prospective.freshness.assess                                                   (freshness checks)
  → prospective.inference.load_production + predict_state                         (frozen M4A, integrity-checked)
  → ledger.writer.commit_prediction                                                (validation, gate, transaction)
  → Receipt {ledger_id, seq, event_id, event_hash, previous_hash, recorded_at}
```

`ledger.service` performs this path without copying any feature or model code. A prediction counts as recorded only
when a receipt exists: a failed transaction raises, and no receipt is returned. `verify_receipt` re-checks a receipt
independently.

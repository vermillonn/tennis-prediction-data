# Phase 4C: prospective evaluation, external checkpointing and independent audit

Phase 4C is an independent, **read-only** auditor of the Phase 4B prediction ledger. It does not generate, modify,
retrain, recalibrate or select predictions. It answers, from committed evidence only:

1. Which predictions were recorded before their matches?
2. Can their commitment history be independently verified?
3. Which matches have trustworthy final outcomes?
4. Which predictions are eligible for scoring?
5. How does the frozen M4A compare with its committed surface-Elo baseline on the same matches?
6. How well calibrated are the probabilities?
7. What coverage, exclusions and failures affect interpretation?
8. How much statistical uncertainty remains?

**Status (2026-10-26):**

- Prospective predictions are **DISABLED** and the May–October 2026 backfill is **INCOMPLETE**. Both gates are
  unchanged.
- Real provider connectivity is **UNVERIFIED**.
- **External checkpoint publication is NOT CONFIGURED.** The GitHub and RFC 3161 backends are implemented and tested
  against local fakes only; **no real publication has occurred**.
- No operational ledger exists in this checkout. The live prospective report therefore shows **0 eligible genuine
  prospective forecasts** and reports no performance metric.
- A clearly labelled SYNTHETIC demonstration lives in `reports/evaluation/synthetic_demonstration/`.

Frozen model: M4A, raw (uncalibrated), frozen hash `bc4e5c870c6e21b9`, configuration hash `5e72010100065889`. Both
are read from `config/prospective.toml` and the frozen modeling configuration and are never changed.

## 1. Modules (`src/tennis_data/evaluation_prospective/`)

| module | role |
|---|---|
| `config.py` | `config/evaluation.toml`; the frozen protocol (`pe-1`) and its hash |
| `safeio.py` | untrusted-input handling: size limits, strict JSON (no duplicate keys, NaN or Infinity), safe relative paths, no-overwrite writes |
| `snapshot.py` | content-addressed ledger snapshots: export through a read-only connection; strict canonical loader; change detection |
| `integrity.py` | Phase 4B `verify_events`, in-chain checkpoint events, and checkpoint-vs-snapshot consistency (rewrite, fork, truncation) |
| `anchoring.py` | anchor records; publishers (local fixture, GitHub Contents, RFC 3161); receipt verification; evidence-based trust levels |
| `timestamping.py` | RFC 3161 request building and offline token verification with the system `openssl` |
| `outcomes.py` | outcome snapshots (`tennis-outcome-snapshot-1`) and reconciliation with the ledger lifecycle |
| `eligibility.py` | one exclusive category per prediction, overlapping flags, forecast selection, evidence level, subgroup labels |
| `metrics.py` | paired M4A / Elo scoring through the shared Phase 2/3B scorer |
| `calibration.py` | reliability tables and pre-declared subgroups (measured, never fitted) |
| `uncertainty.py` | week and tournament block bootstrap (shared Phase 3B implementation) with sample-size gates |
| `coverage.py` | flow, categories, flags, evidence levels, exclusions, voids, unresolved outcomes, each with its denominator |
| `report.py` | deterministic Markdown reports rendered from `results.json` |
| `pipeline.py` | `evaluate` (pure function of its inputs), audit packages, provenance |
| `verify.py` | independent verification of an audit package (recomputes everything) |
| `demo.py` | the SYNTHETIC demonstration world |
| `benchmark.py` | stage timings and memory on synthetic ledgers of increasing size |
| `cli.py` | `tennis-data eval-*` commands |

Reused and not redefined:

- Hash chain, canonical JSON, checkpoints and lifecycle rules come from `ledger.*`.
- Log loss, Brier, accuracy, calibration bins, ECE, per-row losses and the paired cluster bootstrap come from
  `elo.evaluation` and `modeling.evaluation`, with the same 1e-15 clip. `metrics.check_clip` refuses a protocol
  whose clip differs.
- Score-status markers come from Phase 1 `normalization.score.match_status`.
- The outcome policy follows Phase 4A `rec-1` (latest version per provider; disagreement is never resolved by
  preference). Phase 4A reconciles *provider* records into the archive. Phase 4C applies the same principles to final
  outcomes of predicted matches, with an extra identity check against the predicted pair.

No Phase 1–4B code, configuration or report was modified.

## 2. Architecture

```
ledger.sqlite ──(read-only export, ledger.export)──▶ snapshots/<sha16>/events.jsonl      content-addressed
                                                         │
         anchors/records/*.json, tokens/*.tsr ──▶ integrity + anchors + retained checkpoints ──▶ trust levels
                                                         │
         outcomes.jsonl (provenance) ────────────▶ lifecycle replay + outcome reconciliation
                                                         │
                                         classification (protocol pe-1, selection before outcomes)
                                                         │
                         paired metrics ─ calibration ─ subgroups ─ block bootstrap ─ coverage
                                                         │
          package/{inputs/, results.json, reports/, tables/, manifest.json}  ──▶  eval-verify (recompute)
```

`evaluate()` takes only the following inputs:

- snapshot bytes;
- the outcome snapshot;
- the protocol;
- the anchor evidence;
- the verifier's TSA trust anchors;
- the online/offline choice.

No wall-clock time, absolute path or host detail enters `results.json` or the reports, so identical inputs give
byte-identical outputs (§13). Operational provenance (git commit, status gates) is stored only in `manifest.json`,
and reports show it as a claim.

## 3. Deficiencies of the ledger-1 schema (found during inspection, handled here, not changed in Phase 4B)

| # | deficiency | consequence | Phase 4C handling |
|---|---|---|---|
| 1 | `MATCH_RESULT_RECORDED` does not check that `winner_id` is one of the predicted players | a wrong-match result could be recorded | outcome `IDENTITY_MISMATCH` (`winner_not_in_match`), never scored |
| 2 | the prediction payload has no `tournament_level` | the tournament-level subgroup needs another source | taken from the outcome snapshot; `(unavailable)` otherwise |
| 3 | `LEDGER_CHECKPOINT_CREATED` carries no external receipt | in-chain checkpoints cannot prove anything externally | anchor records and receipts are kept outside the ledger (`tennis-ledger-anchor-1`); in-chain checkpoint events are still checked for consistency |
| 4 | a result event names a single source | cross-provider agreement cannot be expressed in the chain | an outcome snapshot with per-provider records and versions; unanimity required |
| 5 | `recorded_at` is the host clock | the chain fixes order and content, not time | timing is classified as host-clock evidence unless an RFC 3161 token attests the covering checkpoint before the actual start |
| 6 | the rejected-idempotency-conflict log is not part of the export | conflicting re-submissions are invisible to an exporter-only auditor | documented; accepted forecasts are all in the export, and conflicts never replace an accepted forecast |
| 7 | the schedule snapshot id is part of the idempotency key | a re-issued schedule allows a second forecast in the same slot | the frozen selection rule picks the first admissible forecast (`DUPLICATE_SUPERSEDED` for the rest) |
| 8 | the actual start is only a `MATCH_METADATA_CORRECTED` (`actual_start`) event, and the lifecycle keeps the latest one | a later source could hide an earlier start | every sourced start (ledger events and outcome snapshot) is collected and the **earliest** is used |

None of these is a defect that invalidates an existing Phase 4B guarantee, so no earlier component was modified.

## 4. Read-only snapshot isolation

- `snapshot_from_ledger` uses Phase 4B `ledger.export` through a read-only SQLite connection (`mode=ro`). It writes to
  a temporary directory and renames it to `<snapshot_dir>/<sha256[:16]>/`. An existing directory must hold identical
  bytes.
- `load_export` accepts only canonical JSON lines, which is exactly what the exporter writes. The meta keys and event
  columns must be exactly the ledger-1 ones; duplicate keys and non-finite numbers are rejected; size limits apply.
- `assert_unchanged` re-hashes the snapshot and outcome files after evaluation. A change raises `SnapshotChanged`.
- Tests show the evaluation leaves unchanged:
  - the SQLite file;
  - the frozen M4A fixtures (`tests/fixtures/m4a_production`);
  - the anchor evidence directory.
- Every report identifies:
  - the ledger id, head sequence and head hash;
  - the snapshot export SHA-256 and the outcome-snapshot SHA-256;
  - the evaluation cutoff;
  - the protocol version and hash;
  - the frozen model and configuration hashes;
  - the anchor verification mode and the TSA trust anchors.

  The package manifest adds the git commit (`+uncommitted-changes` when the tree is dirty) and the status gates.

## 5. External checkpoint anchoring

### 5.1 Anchor record (`tennis-ledger-anchor-1`)

```
{format, ledger_id, checkpoint_sequence, chain_head_hash, checkpoint_schema_version, previous_checkpoint_hash,
 checkpoint_hash, hash_algorithm: "sha256", checkpoint: <Phase 4B checkpoint>, publisher: {backend, identity},
 published_at: <publishing host clock, a claim>, receipt: <backend evidence>}
```

- Records are created with `O_EXCL` and are never overwritten. A second publication of the same checkpoint becomes a
  new record beside the first.
- A failed publication writes a failure note under `failures/` and **no record**. The ledger is never touched.
- The receipt *verification status* is not stored as a trusted field: it is recomputed at every verification. A
  label added to a record (for example `"trust_level": "INDEPENDENTLY_TIMESTAMPED"`) is ignored (tested).

### 5.2 Backends

| backend | publishes | receipt | verification | best level |
|---|---|---|---|---|
| `local_fixture` | a copy in a local directory | path + SHA-256 | file must match | **LOCAL_ONLY**, always (deterministic tests and demos only) |
| `github_contents` | the checkpoint file, committed to an explicitly configured repository through the Contents API | repository, branch, path, commit SHA, blob SHA, content SHA-256, API commit date (informational) | online: the file **at the commit SHA** must equal the checkpoint bytes; also reports `removed_from_branch` if the branch no longer holds it | **EXTERNALLY_PUBLISHED** |
| `rfc3161` | a time-stamp request for the checkpoint hash to an explicitly configured TSA | token file + SHA-256 + imprint | offline `openssl ts -verify` against CA certificates **the verifier** passes (`--tsa-ca`); the imprint must equal the checkpoint hash | **INDEPENDENTLY_TIMESTAMPED** |

The backends fail closed:

- **GitHub:** a missing token or a non-https URL is an error. An existing path with different content is refused
  (never overwritten). After the commit, the file is read back *at the returned commit* and compared byte for byte;
  any mismatch raises and nothing is recorded.
- **RFC 3161:** the token is verified before it is stored. A token that does not verify against the configured
  trusted CAs is not recorded.
- Tokens come from the environment only, are never logged or written, and must be least-privilege: contents:write on
  the one checkpoint repository.

Both remote backends are `enabled = false` in `config/evaluation.toml`, and tests never publish. **No real publication
has been made**: there is no authorized repository, token or TSA configuration.

### 5.3 What a GitHub commit does and does not prove

A commit proves only that the checkpoint bytes were in that repository at that commit. It does **not** prove time:

- The commit and author dates are set by the client.
- Repository owners and administrators, or anyone holding a write token, can force-push, rewrite or delete history,
  and can delete or replace the repository.
- GitHub may continue to serve an orphaned commit by SHA for some time, which is why the verifier separately reports
  `removed_from_branch`.

A GitHub publication is therefore **EXTERNALLY_PUBLISHED**, never INDEPENDENTLY_TIMESTAMPED. Its value is that
rewriting the local ledger after publication becomes detectable by anyone who fetched, or can still fetch, the
published checkpoint. This value grows when:

- the repository is controlled by a different party than the ledger operator;
- branch protection forbids force pushes;
- third parties mirror it.

### 5.4 Trust levels (computed from evidence only)

| level | meaning |
|---|---|
| INVALID | integrity verification failed, or an anchor or retained checkpoint contradicts the snapshot (rewrite, fork, truncation, altered evidence) |
| LOCAL_ONLY | the chain verifies internally; no external evidence verifies (no anchors, fixture anchors, offline or unreachable remotes, untrusted TSA). Proves nothing about *when* |
| EXTERNALLY_PUBLISHED | the exact checkpoint bytes were read back from an independently hosted commit; time and administrative independence not attested |
| INDEPENDENTLY_TIMESTAMPED | an RFC 3161 token, verified against a TSA the verifier trusts, proves the checkpoint existed no later than the attested time |

How an event gets its level:

- An event at sequence `s` takes the strongest verified evidence among anchors with `checkpoint_sequence ≥ s`. Its
  attested time is the earliest attested time among the covering timestamp anchors.
- A timestamp only counts for pre-match evidence if the attested time is **before the actual start**. Otherwise the
  forecast falls back to EXTERNALLY_PUBLISHED or LOCAL_ONLY, flagged `timestamp_not_before_actual_start`.
- A TSA token is not a publication of the bytes, so it does not count as EXTERNALLY_PUBLISHED.
- Any INVALID anchor, forked sequence (two anchors of one sequence with different hashes) or inconsistent retained
  checkpoint makes the whole snapshot `INVALID_EVIDENCE`, and nothing is scored.

### 5.5 Threat model and assumptions

| threat | effect | detection / mitigation | residual assumption |
|---|---|---|---|
| Host clock manipulation | `recorded_at` backdated | regressions are refused and reported (4B). The primary cohort needs a TSA time before the actual start; host-clock timing is only "operational" | without a TSA token, commit times are as trustworthy as the host |
| Remote administrators (GitHub) | rewrite, force-push or delete checkpoint history | the verifier reads the file at the recorded commit, reports branch removal, and compares with retained checkpoints | the GitHub repository is not a timestamp authority; independence must come from who controls it |
| Credential compromise | an attacker publishes forged checkpoints, or deletes real ones | records are never overwritten; two anchors of one sequence with different hashes are a fork (INVALID); retained checkpoints held by auditors contradict a forged history | a compromised token can still add a *consistent* forged checkpoint for a forged chain if no earlier anchor or retained checkpoint exists |
| Remote history rewriting | the published checkpoint disappears | `removed_from_branch`; mirrors and retained checkpoints keep the evidence | auditors should retain checkpoints (`trusted_checkpoint_dir`) |
| Checkpoint deletion (local) | anchors removed to hide a later rewrite | the evaluation can only use evidence that is present; evidence-level counts and the integrity report show what was supplied | external copies (GitHub, TSA receipts held by others) are the defence |
| Reconstructed hash chain | the full chain re-hashed with different content | internal verification passes, but any retained checkpoint or verified anchor at an earlier head fails ("history rewritten or forked") | undetectable if no checkpoint was retained or published before the rewrite |
| Loss of network connectivity | remote receipts cannot be checked | GitHub receipts stay LOCAL_ONLY (`not_checked_offline` or `unreachable`), never assumed valid. RFC 3161 verification is offline and unaffected | none: failure is conservative |
| TSA compromise, mis-trust or certificate revocation | false times | the trust anchor is chosen by the verifier, never by the evidence; a package verified with other trust anchors reports `trust_anchors` and differing results. Revocation is applied by the verifier removing the TSA's CA from `--tsa-ca`, after which its tokens are LOCAL_ONLY (tested) | the chosen TSA is honest. `openssl ts -verify` does not consult CRL or OCSP, so revocation is not discovered automatically |
| Malicious input files | parser exploits, path traversal, oversized files | size limits, strict JSON, canonical lines, exact key sets, safe relative paths (no `..`, absolute paths, drives or symlink escapes), regular files only, no deserialization of code | the `openssl` binary is trusted |

## 6. Prediction eligibility (protocol `pe-1`)

Each `PREDICTION_COMMITTED` event gets exactly one category: the first rule that applies.

| order | category | rule |
|---|---|---|
| 1 | INVALID_CHAIN | the snapshot fails chain or lifecycle verification |
| 2 | INVALID_EVIDENCE | an anchor or retained checkpoint contradicts the snapshot |
| 3 | SYNTHETIC / HISTORICAL_REPLAY / PROSPECTIVE | the mode is not the protocol's cohort mode. Never upgraded |
| 4 | MODEL_NOT_APPROVED | model id ≠ `M4A+raw`, frozen hash not `bc4e5c870c6e21b9…`, or configuration hash ≠ `5e72010100065889` |
| 5 | MISSING_BASELINE | no committed finite Elo baseline probability |
| 6 | NOT_OPERATIONALLY_READY | recorded readiness fails: freshness `fail` or an unverified schedule. PROSPECTIVE also needs the trusted system clock, the status gate `[COMPLETE, ENABLED]` at commit time, a prediction cutoff, and a commit before the scheduled start |
| 7 | NOT_PRIMARY_SLOT | slot ≠ `pre_match` |
| 8 | DUPLICATE_SUPERSEDED | an earlier admissible forecast of the match exists in the primary slot |
| 9 | VOIDED | voided under an objective, evidence-backed reason (Phase 4B `VOID_REASONS`) |
| 10 | MATCH_CANCELLED | the match was cancelled |
| 11 | MATCH_SUPERSEDED | the players changed (the match id was superseded) |
| 12 | LATE_COMMITMENT | committed at or after the earliest sourced actual start |
| 13 | OUTCOME_PENDING | admissible, but no outcome yet. **Stays in the cohort**, reported as pending |
| 14 | TIMING_UNRESOLVED | outcome known, no actual-start evidence: pre-match commitment unverified |
| 15 | OUTCOME_IDENTITY_MISMATCH | the outcome's players or winner do not match the forecast |
| 16 | OUTCOME_DISPUTED | sources disagree, or a version was reissued |
| 17 | OUTCOME_RETIRED | retirement: secondary analysis only (`retirement_policy = "separate"`) |
| 18 | OUTCOME_WALKOVER / DEFAULT / ABANDONED / UNKNOWN | not a played, completed match |
| 19 | SCORABLE | all of the above passed, and the outcome is COMPLETED with a binary winner |

Forecast admissibility (rules 3–12, plus pending) is kept separate from outcome scorability (rules 13–19).

Cohorts:

- **Primary:** SCORABLE and INDEPENDENTLY_TIMESTAMPED before the actual start.
- **Operational:** every SCORABLE forecast, with its weaker evidence level shown. Never described as independently
  timestamp-verified.
- **Retirement (secondary):** OUTCOME_RETIRED.

Overlapping flags are counted separately, for example:

- `readiness:*`, `voided:*`, `model_not_approved`;
- `committed_at_or_after_actual_start`, `timestamp_not_before_actual_start`;
- `mode_*`.

## 7. Multiple forecasts, postponements, corrections and voids

The selection policy is `first_admissible_in_slot`, frozen in `pe-1`:

- **Selection happens before outcomes are used.** Per match, the selected forecast is the lowest-sequence one in the
  primary slot that passes rules 3–6.
- If the selected forecast is voided, late or cancelled, the match drops out. **A later forecast never takes its
  place**, so a losing forecast cannot be replaced by a better one (tested).
- An identical re-submission never creates a second event (4B idempotency).
- A re-issued schedule creates a new forecast, which is `DUPLICATE_SUPERSEDED`.
- Forecasts in other slots are `NOT_PRIMARY_SLOT`.

Postponements, corrections and voids:

- **Postponement or schedule change:** the forecast keeps the scheduled start it committed (never rewritten). Timing is
  judged against the earliest sourced *actual* start.
- **Cancellation:** MATCH_CANCELLED (or VOIDED, when an evidence-backed void exists). A cancelled match with a
  reported result is DISPUTED.
- **Player substitution or corrected identity:** MATCH_SUPERSEDED, and the new match id is a separate match.
- **Voids:**
  - Only Phase 4B's objective reasons are possible: `match_cancelled`, `players_changed`,
    `committed_after_actual_start` and `walkover_no_play`. Each needs evidence events of the same match. There is no
    discretionary or outcome-based void.
  - Every void is listed in the coverage report with reason, actor and evidence.
  - A void of a non-selected forecast changes nothing.
- **Missing actual-start evidence:** TIMING_UNRESOLVED once the outcome is known; OUTCOME_PENDING before that.

## 8. Match-outcome reconciliation and scoring policy

The outcome snapshot (`tennis-outcome-snapshot-1`) is JSONL. Its header is
`{format, snapshot_id, created_at, description}`, followed by records with exactly these fields:

`match_id`, `provider`, `provider_record_id`, `source_version`, `status`, `winner_id`, `player_ids`, `score`,
`actual_start_at`, `tournament_id`, `tournament_level`, `retrieved_at`.

Policy `unanimous_latest_version`:

1. Per (provider, record), the highest `source_version` is current. Different contents under one version are
   `version_reissued`, which makes the match DISPUTED.
2. The ledger's current result (after corrections) and every provider's current record must agree on status and
   winner. Otherwise the match is DISPUTED, quarantined from scoring, with no preferred source.
3. A score whose own marker (RET, W/O, DEF, ABD via Phase 1 `match_status`) contradicts its status is DISPUTED.
4. Players must equal the predicted pair and a winner must be one of them; otherwise IDENTITY_MISMATCH.
5. Statuses stay distinct: COMPLETED, RETIRED, WALKOVER, DEFAULT, ABANDONED, UNKNOWN, CANCELLED, SUPERSEDED, PENDING.
   Walkovers, defaults and abandonments are never converted into ordinary wins.
6. A dispute is resolved only by a newer source version or a ledger correction.
7. Records retrieved, and ledger events recorded, after `--cutoff` are ignored.

A correction produces a new snapshot hash and therefore a new package. Earlier packages stay as they are, with their
own provenance.

Primary match-winner scoring uses COMPLETED outcomes only. Retirements are analysed separately (or excluded, if the
protocol says `exclude`) and never enter the primary score.

## 9. Metrics, calibration and subgroups

- M4A (`probability_a`, raw) and the committed Elo baseline (`baseline_elo_probability_a`) are scored on exactly the
  same matches. Scoring uses the shared scorer: log loss as the primary metric, plus Brier, accuracy (half credit at
  0.5), ECE and reliability bins. Both models use the same 1e-15 clip.
- Paired per-match differences are `d_i = LL_M4A,i − LL_Elo,i`; negative favours M4A. They are listed per match in
  `results.json`.
- Calibration is measured, never fitted, and nothing is learned from prospective outcomes.
- Pre-declared subgroups:
  - surface;
  - tournament level;
  - best-of;
  - prior service points (min over both players; Phase 3B edges 1 / 1000 / 5000);
  - Elo experience (min prior rated matches; 1 / 20 / 100);
  - favourite probability range;
  - freshness status;
  - month.
- All subgroups are always listed. Groups under 50 matches show counts only and are marked inconclusive, so no
  subgroup can be picked after looking.

## 10. Statistical uncertainty

- **Paired block bootstrap** of ΔLL and ΔBrier (`modeling.evaluation.paired_bootstrap`): 2,000 resamples, seed
  20261025.
- **Clusters:**
  - ISO week of the actual start;
  - tournament id, only when every scored match has one.
- **Sample-size gate:** no interval is reported below **200 matches** or **10 blocks**. The result is then
  "insufficient evidence (descriptive only)".
- **Season-level figures** are descriptive only.
- Week and tournament resampling ignore dependence through recurring players, so intervals are, if anything, too
  narrow.

## 11. Coverage and exclusion accounting

`coverage_report.md` lists:

- **the flow:**
  - all commitments;
  - cohort-mode records;
  - approved, ready records;
  - selected forecasts;
  - not voided, cancelled or superseded;
  - committed before a verified start;
  - scorable;
  - the primary cohort;
- **every category, mode, flag and evidence level, each with its denominator;**
- **voids, unresolved outcomes and every exclusion**, with the full list in `tables/exclusions.csv`.

`tables/classification.csv` gives every prediction's final category, flags, timing, evidence level, cohort and
outcome.

## 12. Audit packages and independent verification

```
package/
  inputs/ledger/events.jsonl          the evaluated snapshot (exact bytes)
  inputs/outcomes.jsonl               the outcome snapshot (if any)
  inputs/evidence/records/…, tokens/… anchor records and RFC 3161 tokens of this ledger
  inputs/fixture_publications/…       fixture copies referenced by receipts
  inputs/trusted_checkpoints/…        retained checkpoints
  inputs/protocol.json                the frozen protocol
  results.json                        canonical results
  reports/*.md   tables/*.csv         renderings
  manifest.json                       SHA-256 of every file, identity, provenance
```

`tennis-data eval-verify PACKAGE [--tsa-ca CA] [--online] [--manifest-sha256 H] [--check-protocol]` needs only the
package, the verifier's own trust anchors and this code: no SQLite ledger. It checks:

1. **manifest:** every listed file is present with its hash, all paths are safe, and there are no unlisted files.
   Optionally, the manifest hash equals an independently obtained reference.
2. **inputs:** everything loads under the strict parsers.
3. **protocol:** optionally, the protocol equals the verifier's configuration.
4. **recompute:**
   - chain integrity;
   - checkpoints and anchors (under the *verifier's* TSA trust anchors and online/offline choice);
   - outcome reconciliation and eligibility;
   - metrics, calibration and uncertainty.

   `results.json` must be byte-identical.
5. **reports:** every report re-renders byte-identically.

Published numbers are never accepted from a report or from `results.json`: they are recomputed. An attacker who
edits an input, result or report *and* updates the manifest consistently is still caught by recomputation. Swapping
the whole package for a self-consistent forgery is caught only by comparing the manifest hash with a reference
obtained independently, for example one published with a checkpoint.

The result is machine-readable JSON: `{ok, checks: [{check, ok, detail}], recomputed: {…}}`.

## 13. Commands

| command | purpose |
|---|---|
| `tennis-data eval-snapshot` | read-only, content-addressed export of the configured ledger |
| `tennis-data eval-checkpoint [--export F] [--previous CP] [--out F]` | the checkpoint of a snapshot head (no database access) |
| `tennis-data eval-checkpoint-publish CP --backend {local_fixture,github_contents,rfc3161} --identity NAME` | publish with an **explicitly enabled** backend and record the anchor |
| `tennis-data eval-checkpoint-verify [--export F] [--tsa-ca CA] [--online]` | anchors and retained checkpoints vs a snapshot |
| `tennis-data eval-run [--export F] [--outcomes F] [--cutoff TS] [--tsa-ca CA] [--online] [--package DIR] [--write-reports]` | evaluate and write an audit package (and the repository reports) |
| `tennis-data eval-verify PACKAGE …` | independent verification |
| `tennis-data eval-calibration PACKAGE [--cohort] [--model]` | calibration table of a *verified* package as CSV |
| `tennis-data eval-exclusions PACKAGE [--unresolved]` | exclusions or unresolved outcomes of a verified package |
| `tennis-data eval-demo --out DIR [--write-reports]` | the SYNTHETIC demonstration |
| `tennis-data eval-benchmark [--sizes 100,1000,10000]` | stage timings and memory |

The DuckDB `ledger_*` views of Phase 4B remain the analytical interface to exports. Phase 4C adds no views, so the
standard CLI tests need no `data/` directory.

## 14. Reproducibility and performance

- The same inputs give byte-identical `results.json`, reports, tables and manifest (tested by building two packages).
  Two full runs of the live evaluation and of the demonstration produced identical package hashes; see
  `reports/evaluation/verification_record.md`.
- Benchmark (`eval-benchmark`, synthetic demo ledgers, measured 2026-10-26):

| predictions | events | scorable | integrity | outcome reconciliation | classification (incl. reconciliation) | metrics + calibration + bootstrap | reports | full evaluate | peak Python allocation |
|---|---|---|---|---|---|---|---|---|---|
| 103 | 268 | 69 | 0.04 s | 0.008 s | 0.012 s | 0.023 s | 0.001 s | 0.08 s | 1.4 MB |
| 1,042 | 2,702 | 690 | 0.30 s | 0.04 s | 0.12 s | 0.15 s | 0.001 s | 0.65 s | 13.7 MB |
| 10,235 | 26,769 | 7,099 | 3.34 s | 0.63 s | 2.20 s | 0.61 s | 0.004 s | 5.86 s | 136.6 MB |

  Process peak RSS over the whole benchmark was 705 MB (`reports/evaluation/benchmark.json`). Integrity verification, the Phase 4B verifier, dominates and
  scales linearly.

## 15. Remaining operational blockers

1. **No genuine prospective forecast exists.** The status gate is DISABLED, the backfill is INCOMPLETE and real
   provider connectivity is UNVERIFIED (Phase 4A). The live report is correctly empty.
2. **No external publication is configured.** Before any prospective use, the operator must provide:
   - an independently controlled checkpoint repository with branch protection, and a least-privilege token; and/or
   - an RFC 3161 TSA URL, with its CA certificates distributed to auditors.
3. **No outcome feed for prospective matches.** Outcome snapshots must come from validated provider data (Phase 4A
   reconciliation). None exists for post-May-2026 matches.
4. **No actual-start feed.** Without sourced actual starts every forecast is TIMING_UNRESOLVED; the providers must
   supply them.
5. **Sample size.** Even after enabling, at least 200 scored matches in at least 10 weeks are needed before any
   interval is reported.

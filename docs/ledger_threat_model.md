# Ledger threat model (Phase 4B)

The ledger aims to make ordinary records immutable, unauthorized modification detectable, and checkpointed history
verifiable. It does **not** claim to stop a privileged attacker. A local hash chain alone is **not** independently
immutable.

| actor or situation | what can happen | protection | remaining risk |
|---|---|---|---|
| Trusted application process (`LedgerWriter`) | Bugs; attempts to commit invalid, late or prospective records | Validation, mode namespaces, the status gate (PROSPECTIVE refused), the writer's own clock, idempotency, transactions; tests | A bug in the writer itself. Mitigated by the independent verifier and tests |
| Accidental modification (scripts, ad-hoc SQL) | UPDATE or DELETE of events; out-of-order insert | SQLite triggers abort UPDATE and DELETE and refuse inserts off the chain head; CHECK and UNIQUE constraints | None for ordinary SQL through SQLite |
| Process crash or power loss | Half-written event | Single transaction per event, WAL + `synchronous=FULL`, rollback on reopen (tested by killing a process mid-transaction) | Storage-level corruption: restore from an export and compare with a checkpoint |
| Compromised local machine or user with file access | Drop the triggers, edit rows, delete events, or rewrite and re-hash the entire chain | `ledger-verify` detects partial edits, deletions, reordering and broken links. A **full rewrite** is only detected by comparison with a checkpoint kept outside the machine | Without external checkpoints, a consistent forged chain is undetectable |
| Privileged database administrator | Same as above, plus replacing the file and its WAL | Same: external checkpoints are the only defence | Same |
| Missing external checkpoints (current state) | History between local creation and the first external publication is unanchored | Checkpoint export and verification are implemented and chained (`previous_checkpoint_hash`) | **No checkpoint has been published.** Until one is, the ledger proves internal consistency, not age or authenticity |
| Clock manipulation | Backdating commits | Commit time comes from the writer's clock; regressions are refused and reported; only `SystemClock` is accepted for PROSPECTIVE | A host whose system clock is wrong. An external trusted timestamp (Phase 4C/deployment) is needed for strong time evidence |
| Selective reporting | Dropping losing forecasts, re-submitting better numbers | No delete; voiding needs evidence-backed objective reasons; conflicts are logged; exports include every event | Someone omitting the ledger file entirely. External checkpoints and publishing the full export address this |
| Credential leakage | API keys in logs or records | The ledger stores no credentials. Provider keys stay in environment variables, and raw-store parameters are redacted (Phase 4A) | n/a |

**Recommendations before any prospective use:**

1. Publish a checkpoint to an independent, append-only location after every batch of commitments. Ideally also
   obtain a trusted timestamp (RFC 3161) for each checkpoint.
2. Keep regular exports outside the machine.
3. Restrict write access to the ledger file to the service account.

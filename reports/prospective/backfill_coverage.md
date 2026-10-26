# Phase 4A backfill coverage: the historical gap to the prospective start

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

## 1. Verdict

Status: historical backfill **INCOMPLETE** · prospective predictions **DISABLED** · real-provider integrations **UNVERIFIED** (gate: `config/prospective.toml` `[status]`).

**The gap is not filled.** The pinned archive (commit `83733587353d`) ends with events starting on 2026-05-25. No result after that is available to this system: 22 weekly batches (2026-05-30 to 2026-10-24) contain no newly starting events (the first of them is only the second week of Roland Garros 2026), and the backfill accepted 0 records for that period. No authorized source was reachable (`source_feasibility.md`). The player state therefore stops at the archive end.

## 2. Final valid coverage of the archive (verified from the validated table)

- Last tournament start date: **2026-05-25** (configured snapshot end 2026-05-25; matches). The end date differs between levels:

| level | last event start |
|---|---|
| A | 2026-05-17 |
| D | 2026-02-07 |
| F | 2025-12-17 |
| G | 2026-05-25 |
| M | 2026-05-06 |
| O | 2024-07-29 |

- Last batch whose close updates Elo: 2026-05-30 (the second week of the last two-week event). Last batch releasing usable point statistics: **2026-05-16**.

Events in the final five weeks (knockout completeness = rows against draw size − 1):

| event | level | start | draw | rows | expected | completeness | usable stats | update Elo | weeks | release batch |
|---|---|---|---|---|---|---|---|---|---|---|
| Madrid Masters | M | 2026-04-22 | 96 | 95 | 95 | complete | 91 | 91 | 2 | 2026-04-25 |
| Rome Masters | M | 2026-05-06 | 96 | 95 | 95 | complete | 93 | 93 | 2 | 2026-05-09 |
| Geneva | A | 2026-05-17 | 28 | 27 | 27 | complete | 27 | 27 | 1 | 2026-05-16 |
| Hamburg | A | 2026-05-17 | 32 | 31 | 31 | complete | 31 | 31 | 1 | 2026-05-16 |
| Roland Garros | G | 2026-05-25 | 128 | 127 | 127 | complete | 0 | 118 | 2 | 2026-05-30 |

The last event (Roland Garros 2026) is complete as a results record but has **no serve statistics** in the archive, so it updates Elo but not the serve/return state.

## 3. 2026 rows in the archive by evidence category

| category | rows |
|---|---|
| 1. complete observation (target + usable statistics) | 1160 |
| 2. usable outcome, missing point statistics | 240 |
| 3. incomplete or ambiguous record (quarantined) | 0 |
| 4. excluded by the frozen eligibility rules (walkovers, retirements, unknown status) | 49 |

| match status | rows |
|---|---|
| completed | 1400 |
| retired | 32 |
| unknown | 5 |
| walkover | 12 |

| statistics exclusion | rows |
|---|---|
| stats_missing | 240 |
| status_retired | 32 |
| status_unknown | 5 |
| status_walkover | 12 |

- Multi-week events in the year: 6. Players not in the player master: 0. Duplicate or conflicting rows quarantined: 0.
- Timestamps: every row has `time_precision = tournament_week` (event start date only). No match date or time exists, so all timing is the conservative weekly-batch inference.

## 4. What is missing (estimate)

No 2026 calendar for the missing weeks is available, so the volume is estimated from the same 22 weekly batches one year earlier (2025-05-31 to 2025-10-25). This is an estimate, not a list of 2026 events:

| level | events | matches |
|---|---|---|
| A | 23 | 667 |
| D | 33 | 109 |
| G | 2 | 254 |
| M | 4 | 340 |

- Estimated missing tour-level matches: **~1370** in ~62 events, including the Grand Slams Us Open, Wimbledon and multi-week events Canada Masters, Cincinnati Masters, Shanghai Masters, Us Open, Wimbledon.
- In the comparison year, 1202 were complete observations, 105 outcome-only and 63 excluded by eligibility; a backfill of similar quality would yield usable statistics for ~88% of matches.

## 5. Backfill run on the current raw store

| item | value |
|---|---|
| raw snapshot id | e3b0c44298fc1c14 |
| raw snapshots | 0 |
| provider match records | 0 |
| accepted after reconciliation | 0 |
| Phase 1 validated | 0 |
| Phase 1 quarantined | 0 |
| with complete statistics | 0 |
| state-eligible Elo updates | 0 |
| state-eligible statistics | 0 |
| state snapshot id | 3253426224f7ebc0 |

Reconciliation outcomes: none.

## 6. Chronology of backfilled results

Each accepted result gets a ledger row (`data/prospective/backfill/ledger.parquet`) with `event_start_date`, `event_start_at` and `event_completed_at` (only if the provider states them), `source_published_at`, `ingested_at`, `release_batch` and `state_eligible_at` = 00:00 UTC on the day after the release batch closes (`eligibility_basis = inferred_weekly_batch`). Multi-week events release one week later, exactly as in Phase 2. Every backfilled row has `prediction_mode = historical_reconstruction`: a result discovered now is never presented as information available, or forecast, on its event date.

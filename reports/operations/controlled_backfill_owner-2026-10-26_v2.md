# Controlled historical backfill, version 2 (owner-2026-10-26)

Authorization: Phase 5A v2 identity approval and isolated v2 backfill authorization, 2026-10-27 (identity review 534f65c). Run started 2026-10-27T06:44:18+00:00. Results only, ATP main tour, frozen weekly batching. Production `data/prospective/` (state `3253426224f7ebc0`) is unchanged; the backfilled state is a new versioned local snapshot. Gates: backfill INCOMPLETE, prospective predictions DISABLED.

## 1. Identity decisions

| evidence class | proposed | approved | rejected by an invariant | pending |
|---|---|---|---|---|
| overlap-corroborated | 205 | 200 | 0 | 0 |
| name and birth date | 103 | 99 | 0 | 1 (the `name_country` mapping) |
| ambiguous | 26 | 0 | 0 | 19 |
| new player | 1 | 0 | 0 | 1 |
| **individual overrides** (owner decision after review) | - | **15** (8 previously rejected by the automatic rule, 7 previously ambiguous) | - | - |

Approval manifest: local and git-ignored (`data/operations/exports/<label>/private/approvals_v2.toml`, SHA-256 `1e3fbab5b2368921…`), each entry with reviewer, approval class, authorization reference and evidence; it holds source-derived player codes and is not published. The proposed mappings (`identity_overrides.proposed.toml`, `1b550309b7d26b24…`) and `config/identity_overrides.toml` are unchanged. Invariants: `proposal_provenance`, `provider_in_verified_source`, `phase4a_decision`, `blocking_flags_clear`, `names_exact`, `birth_date_exact`, `one_to_one`, `no_indistinguishable_namesake`, `not_ambiguous_candidate`, `chronology`, `overlap_corroboration`. Names are compared with `identity.norm` (NFKD accents removed, case-folded, punctuation-insensitive, listed letter folds) on `first last`; birth dates must be equal full dates on both sides. Individual overrides passed `provider_in_verified_source`, `canonical_in_archive`, `birth_date_exact`, `one_to_one`, `not_candidate_of_pending`, `holds_archive_history`, `chronology`, `evidence_recorded`.

Rejected by the automatic rule and not individually approved: 0. Per-mapping evidence is kept in the local run record.

## 2. Matches

| item | count |
|---|---|
| backfill-set records (events after the archive) | 1,252 |
| **accepted (newly imported)** | **1,210** |
| **Elo updates (frozen eligibility)** | **1,176** |
| not Elo-eligible | status_default 1, status_retired 27, status_unknown 1, status_walkover 5 |
| quarantined (identity pending) | 42 (player slots: ambiguous 41, name_and_birth_date_only 1, new_player 1) |
| Phase 4A quarantine | 0 |
| deduplicated against the archive | 0 in the backfill set (Roland Garros 2026 starts before the gap: 0 records; on the full-file validation run 1,233 duplicates were recognised, 117 of them Roland Garros) |
| statistics removed (incompatible definition) | 1182 (state has 0 rows with statistics) |
| excluded | all January–May records (archive period, validation only); `2026_challenger.csv` (never parsed) |

Per-match status is record-level and source-derived: it is kept locally (`data/operations/exports/<label>/private/`), with its SHA-256 below.

## 3. Coverage by batch

| batch | records | accepted | Elo updates | quarantined |
|---|---|---|---|---|
| 2026-06-06 | 54 | 52 | 52 | 2 |
| 2026-06-13 | 62 | 61 | 61 | 1 |
| 2026-06-20 | 54 | 51 | 50 | 3 |
| 2026-06-27 | 127 | 126 | 121 | 1 |
| 2026-07-11 | 81 | 79 | 77 | 2 |
| 2026-07-18 | 54 | 54 | 53 | 0 |
| 2026-07-25 | 58 | 56 | 53 | 2 |
| 2026-08-01 | 95 | 93 | 89 | 2 |
| 2026-08-08 | 95 | 88 | 87 | 7 |
| 2026-08-22 | 47 | 45 | 42 | 2 |
| 2026-08-29 | 127 | 125 | 123 | 2 |
| 2026-09-12 | 57 | 49 | 48 | 8 |
| 2026-09-19 | 103 | 95 | 95 | 8 |
| 2026-09-26 | 62 | 62 | 58 | 0 |
| 2026-10-03 | 95 | 94 | 90 | 1 |
| 2026-10-17 | 81 | 80 | 77 | 1 |

## 4. Coverage by tournament

| start | tournament | level | records | accepted | Elo updates | quarantined |
|---|---|---|---|---|---|---|
| 2026-06-08 | 's-Hertogenbosch | A | 27 | 27 | 27 | 0 |
| 2026-06-08 | Stuttgart | A | 27 | 25 | 25 | 2 |
| 2026-06-15 | Halle | A | 31 | 30 | 30 | 1 |
| 2026-06-15 | Queen's Club | A | 31 | 31 | 31 | 0 |
| 2026-06-22 | Eastbourne | A | 27 | 27 | 26 | 0 |
| 2026-06-22 | Mallorca | A | 27 | 24 | 24 | 3 |
| 2026-06-29 | Wimbledon | G | 127 | 126 | 121 | 1 |
| 2026-07-13 | Bastad | A | 27 | 25 | 24 | 2 |
| 2026-07-13 | Gstaad | A | 27 | 27 | 27 | 0 |
| 2026-07-13 | Umag | A | 27 | 27 | 26 | 0 |
| 2026-07-20 | Estoril | A | 27 | 27 | 26 | 0 |
| 2026-07-20 | Kitzbuhel | A | 27 | 27 | 27 | 0 |
| 2026-07-27 | Washington | A | 31 | 31 | 30 | 0 |
| 2026-07-28 | Los Cabos | A | 27 | 25 | 23 | 2 |
| 2026-08-02 | Canada Masters | M | 95 | 93 | 89 | 2 |
| 2026-08-14 | Cincinnati Masters | M | 95 | 88 | 87 | 7 |
| 2026-08-24 | Winston-Salem | A | 47 | 45 | 42 | 2 |
| 2026-08-30 | US Open | G | 127 | 125 | 123 | 2 |
| 2026-09-18 | Davis Cup QLS R2: AUT vs BEL | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup QLS R2: CAN vs FRA | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup QLS R2: CZE vs USA | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup QLS R2: KOR vs IND | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: BRA vs SUI | D | 3 | 3 | 2 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: DEN vs BUL | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: FIN vs MON | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: LUX vs HUN | D | 3 | 2 | 2 | 1 |
| 2026-09-18 | Davis Cup WG1 R1: NED vs COL | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: NOR vs CHN | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: PER vs PAR | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: BOL vs HKG | D | 3 | 2 | 2 | 1 |
| 2026-09-18 | Davis Cup WG2 R1: DOM vs EGY | D | 3 | 1 | 1 | 2 |
| 2026-09-18 | Davis Cup WG2 R1: INA vs ROU | D | 3 | 1 | 1 | 2 |
| 2026-09-18 | Davis Cup WG2 R1: KAZ vs THA | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: SLO vs ISR | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: TUN vs PAK | D | 2 | 2 | 2 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: URU vs EST | D | 3 | 1 | 1 | 2 |
| 2026-09-19 | Davis Cup QLS R2: CHI vs ESP | D | 3 | 2 | 2 | 1 |
| 2026-09-19 | Davis Cup QLS R2: GBR vs ECU | D | 3 | 2 | 2 | 1 |
| 2026-09-19 | Davis Cup QLS R2: GER vs CRO | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: ARG vs TUR | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: AUS vs POL | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: NZL vs JPN | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: SRB vs LTU | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: SVK vs GRE | D | 4 | 4 | 4 | 0 |
| 2026-09-19 | Davis Cup WG1 R1: TPE vs SWE | D | 4 | 4 | 4 | 0 |
| 2026-09-19 | Davis Cup WG2 R1: BER vs MEX | D | 3 | 1 | 1 | 2 |
| 2026-09-19 | Davis Cup WG2 R1: CYP vs UKR | D | 3 | 3 | 3 | 0 |
| 2026-09-19 | Davis Cup WG2 R1: ESA vs POR | D | 2 | 1 | 1 | 1 |
| 2026-09-19 | Davis Cup WG2 R1: MAR vs NGR | D | 3 | 1 | 1 | 2 |
| 2026-09-19 | Davis Cup WG2 R1: RSA vs BIH | D | 3 | 3 | 3 | 0 |
| 2026-09-23 | Chengdu | A | 27 | 26 | 26 | 1 |
| 2026-09-23 | Hangzhou | A | 27 | 27 | 27 | 0 |
| 2026-09-25 | Laver Cup | A | 6 | 6 | 6 | 0 |
| 2026-09-30 | Beijing | A | 31 | 31 | 28 | 0 |
| 2026-09-30 | Tokyo | A | 31 | 31 | 30 | 0 |
| 2026-10-07 | Shanghai | M | 95 | 94 | 90 | 1 |
| 2026-10-19 | Almaty | A | 27 | 27 | 25 | 0 |
| 2026-10-19 | Brussels | A | 27 | 27 | 26 | 0 |
| 2026-10-19 | Lyon | A | 27 | 26 | 26 | 1 |

## 5. Quarantined players (identity pending)

| class | players | match slots |
|---|---|---|
| ambiguous | 19 | 41 |
| name_and_birth_date_only | 1 | 1 |
| new_player | 1 | 1 |

## 6. State, reproducibility and integrity

| item | value |
|---|---|
| baseline (content-addressed, verified by restore) | `prospective-baseline-3253426224f7ebc0`, tree `5368ff8752320a76…`, 15 files, restored state `3253426224f7ebc0` |
| new state snapshot | `8d898b4c4d0bdab9` |
| reconstruction (frozen engines) | `f5eac02e653b6ae7` (194,810 matches, 227.4 s), `f5eac02e653b6ae7` (194,810 matches, 228.8 s); deterministic **True** |
| replay (published Phase 2/3A/M4A) | passed **True**, 274 cases, max abs error 1.1e-16, max Elo error 0.0 |
| M4A frozen / configuration hash | `bc4e5c870c6e21b9` / `5e72010100065889` (match True) |
| Elo / feature configuration hash | `1c330d4db8801ff6` / `3d889b94456995d1` |
| promoted | **True**: `historical-v2-8d898b4c4d0bdab9` (tree `bd651c3fa679b5a3…`, restored state `8d898b4c4d0bdab9`), registry `data/operations/snapshots/registry.json` |

Verification checks:

- `accepted_equals_imported_minus_quarantine`: pass
- `all_players_manually_approved`: pass
- `elo_feature_configuration_unchanged`: pass
- `frozen_hashes_match`: pass
- `frozen_weekly_batching`: pass
- `historical_reconstruction_only`: pass
- `integrity_passed`: pass
- `main_tour_levels_only`: pass
- `no_archive_duplicates`: pass
- `no_new_player_identities`: pass
- `no_prediction_ledger`: pass
- `no_roland_garros_2026`: pass
- `no_statistics_in_state`: pass
- `nothing_after_prospective_start`: pass
- `phase1_4c_data_unchanged`: pass
- `phase4a_quarantine_empty`: pass
- `proposed_identity_file_unchanged`: pass
- `prospective_gate_closed`: pass
- `reconstruction_deterministic`: pass
- `replay_passed`: pass
- `state_snapshot_reproducible`: pass

## 7. Operational readiness after the backfill (staged state)

- feature freshness fails for a match after the prospective start: last_point_statistics
- no verified live provider: schedules for upcoming matches cannot be ingested automatically
- status gate: backfill INCOMPLETE, prospective predictions DISABLED (config/prospective.toml [status])

Freshness probe: `last_eligible_result` ok, `last_point_statistics` fail, `missing_backfill_weeks` ok, `opponent_fit_age_weeks` ok, `player_recent_history` warning, `source_age_days` ok, `unresolved_identities` ok.

Local per-match status SHA-256: `fe2fdc7f69ba146003984ea1ed136dbda77ec9c8bab23e4e27a1906819979f6b`.


## 8. Expectations and independent post-run checks

- Expected from the isolated simulation: about 1,210 accepted and 1,176 Elo updates (expectations, not success
  criteria). Actual: **1,210 accepted, 1,176 Elo updates, 42 quarantined**, identical to the simulation. The state
  snapshot ID differs from the simulation's (`fe5511fbb590fd5a`) only because it includes the raw-snapshot component,
  whose ingestion time is the real import time of this run.
- Compared with version 1: +117 accepted, +113 Elo updates, quarantine 159 → 42; all 299 version 1 approvals carried
  over unchanged; 15 individual overrides added; no canonical ID merged, deleted or rewritten; no new player identity.
- Full test suite **486 passed**; ruff check and format clean. All 273 files under `data/` outside
  `data/operations/` (Phase 1–4C artifacts and the active `data/prospective/`) are byte-identical to the list taken
  before the version 1 run.
- Versions 0, 1 and 2 restore independently from their content-addressed objects to `3253426224f7ebc0`,
  `9c9d537c9c855dbe` and `8d898b4c4d0bdab9`; the version 1 manifest is byte-identical after this run. Version 0 remains
  the active Phase 4A directory; version 2 is not activated.
- Gates unchanged: backfill INCOMPLETE, prospective predictions DISABLED, `state_availability_v1` disabled, GitHub
  and RFC 3161 checkpoint publishers disabled.

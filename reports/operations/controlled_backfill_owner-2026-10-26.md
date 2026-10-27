# Controlled historical backfill, version 1 (owner-2026-10-26)

Authorization: Phase 5A controlled historical backfill authorization, 2026-10-26 (reviewed package 40519b3). Run started 2026-10-26T21:33:46+00:00. Results only, ATP main tour, frozen weekly batching. Production `data/prospective/` (state `3253426224f7ebc0`) is unchanged; the backfilled state is a new versioned local snapshot. Gates: backfill INCOMPLETE, prospective predictions DISABLED.

## 1. Identity decisions

| evidence class | proposed | approved | rejected by an invariant | pending |
|---|---|---|---|---|
| overlap-corroborated | 205 | 200 | 5 | 0 |
| name and birth date | 103 | 99 | 3 | 1 (the `name_country` mapping) |
| ambiguous | 26 | 0 | 0 | 26 |
| new player | 1 | 0 | 0 | 1 |

Approval manifest: local and git-ignored (`data/operations/exports/<label>/private/approvals_v1.toml`, SHA-256 `f2d9633200f63ca8…`), each entry with reviewer, approval class, authorization reference and evidence; it holds source-derived player codes and is not published. The proposed mappings (`identity_overrides.proposed.toml`, `1b550309b7d26b24…`) and `config/identity_overrides.toml` are unchanged. Invariants: `proposal_provenance`, `provider_in_verified_source`, `phase4a_decision`, `blocking_flags_clear`, `names_exact`, `birth_date_exact`, `one_to_one`, `no_indistinguishable_namesake`, `not_ambiguous_candidate`, `chronology`, `overlap_corroboration`. Names are compared with `identity.norm` (NFKD accents removed, case-folded, punctuation-insensitive, listed letter folds) on `first last`; birth dates must be equal full dates on both sides.

Rejected by the automatic rule and not individually approved: 8 (`phase4a_decision` 8). Per-mapping evidence is kept in the local run record.

## 2. Matches

| item | count |
|---|---|
| backfill-set records (events after the archive) | 1,252 |
| **accepted (newly imported)** | **1,093** |
| **Elo updates (frozen eligibility)** | **1,063** |
| not Elo-eligible | status_default 1, status_retired 25, status_unknown 1, status_walkover 3 |
| quarantined (identity pending) | 159 (player slots: ambiguous 102, name_and_birth_date_only 1, new_player 1, rejected_by_invariant 62) |
| Phase 4A quarantine | 0 |
| deduplicated against the archive | 0 in the backfill set (Roland Garros 2026 starts before the gap: 0 records; on the full-file validation run 1,233 duplicates were recognised, 117 of them Roland Garros) |
| statistics removed (incompatible definition) | 1069 (state has 0 rows with statistics) |
| excluded | all January–May records (archive period, validation only); `2026_challenger.csv` (never parsed) |

Per-match status is record-level and source-derived: it is kept locally (`data/operations/exports/<label>/private/`), with its SHA-256 below.

## 3. Coverage by batch

| batch | records | accepted | Elo updates | quarantined |
|---|---|---|---|---|
| 2026-06-06 | 54 | 50 | 50 | 4 |
| 2026-06-13 | 62 | 58 | 58 | 4 |
| 2026-06-20 | 54 | 44 | 43 | 10 |
| 2026-06-27 | 127 | 116 | 112 | 11 |
| 2026-07-11 | 81 | 73 | 71 | 8 |
| 2026-07-18 | 54 | 49 | 48 | 5 |
| 2026-07-25 | 58 | 51 | 48 | 7 |
| 2026-08-01 | 95 | 80 | 77 | 15 |
| 2026-08-08 | 95 | 81 | 80 | 14 |
| 2026-08-22 | 47 | 45 | 42 | 2 |
| 2026-08-29 | 127 | 106 | 104 | 21 |
| 2026-09-12 | 57 | 42 | 41 | 15 |
| 2026-09-19 | 103 | 88 | 88 | 15 |
| 2026-09-26 | 62 | 49 | 46 | 13 |
| 2026-10-03 | 95 | 84 | 80 | 11 |
| 2026-10-17 | 81 | 77 | 75 | 4 |

## 4. Coverage by tournament

| start | tournament | level | records | accepted | Elo updates | quarantined |
|---|---|---|---|---|---|---|
| 2026-06-08 | 's-Hertogenbosch | A | 27 | 26 | 26 | 1 |
| 2026-06-08 | Stuttgart | A | 27 | 24 | 24 | 3 |
| 2026-06-15 | Halle | A | 31 | 29 | 29 | 2 |
| 2026-06-15 | Queen's Club | A | 31 | 29 | 29 | 2 |
| 2026-06-22 | Eastbourne | A | 27 | 20 | 19 | 7 |
| 2026-06-22 | Mallorca | A | 27 | 24 | 24 | 3 |
| 2026-06-29 | Wimbledon | G | 127 | 116 | 112 | 11 |
| 2026-07-13 | Bastad | A | 27 | 25 | 24 | 2 |
| 2026-07-13 | Gstaad | A | 27 | 26 | 26 | 1 |
| 2026-07-13 | Umag | A | 27 | 22 | 21 | 5 |
| 2026-07-20 | Estoril | A | 27 | 25 | 24 | 2 |
| 2026-07-20 | Kitzbuhel | A | 27 | 24 | 24 | 3 |
| 2026-07-27 | Washington | A | 31 | 27 | 26 | 4 |
| 2026-07-28 | Los Cabos | A | 27 | 24 | 22 | 3 |
| 2026-08-02 | Canada Masters | M | 95 | 80 | 77 | 15 |
| 2026-08-14 | Cincinnati Masters | M | 95 | 81 | 80 | 14 |
| 2026-08-24 | Winston-Salem | A | 47 | 45 | 42 | 2 |
| 2026-08-30 | US Open | G | 127 | 106 | 104 | 21 |
| 2026-09-18 | Davis Cup QLS R2: AUT vs BEL | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup QLS R2: CAN vs FRA | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup QLS R2: CZE vs USA | D | 4 | 2 | 2 | 2 |
| 2026-09-18 | Davis Cup QLS R2: KOR vs IND | D | 3 | 1 | 1 | 2 |
| 2026-09-18 | Davis Cup WG1 R1: BRA vs SUI | D | 3 | 3 | 2 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: DEN vs BUL | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: FIN vs MON | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: LUX vs HUN | D | 3 | 2 | 2 | 1 |
| 2026-09-18 | Davis Cup WG1 R1: NED vs COL | D | 3 | 2 | 2 | 1 |
| 2026-09-18 | Davis Cup WG1 R1: NOR vs CHN | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG1 R1: PER vs PAR | D | 4 | 4 | 4 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: BOL vs HKG | D | 3 | 2 | 2 | 1 |
| 2026-09-18 | Davis Cup WG2 R1: DOM vs EGY | D | 3 | 1 | 1 | 2 |
| 2026-09-18 | Davis Cup WG2 R1: INA vs ROU | D | 3 | 1 | 1 | 2 |
| 2026-09-18 | Davis Cup WG2 R1: KAZ vs THA | D | 4 | 2 | 2 | 2 |
| 2026-09-18 | Davis Cup WG2 R1: SLO vs ISR | D | 3 | 3 | 3 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: TUN vs PAK | D | 2 | 2 | 2 | 0 |
| 2026-09-18 | Davis Cup WG2 R1: URU vs EST | D | 3 | 1 | 1 | 2 |
| 2026-09-19 | Davis Cup QLS R2: CHI vs ESP | D | 3 | 1 | 1 | 2 |
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
| 2026-09-23 | Hangzhou | A | 27 | 23 | 23 | 4 |
| 2026-09-25 | Laver Cup | A | 6 | 4 | 4 | 2 |
| 2026-09-30 | Beijing | A | 31 | 27 | 24 | 4 |
| 2026-09-30 | Tokyo | A | 31 | 22 | 22 | 9 |
| 2026-10-07 | Shanghai | M | 95 | 84 | 80 | 11 |
| 2026-10-19 | Almaty | A | 27 | 26 | 24 | 1 |
| 2026-10-19 | Brussels | A | 27 | 25 | 25 | 2 |
| 2026-10-19 | Lyon | A | 27 | 26 | 26 | 1 |

## 5. Quarantined players (identity pending)

| class | players | match slots |
|---|---|---|
| ambiguous | 26 | 102 |
| name_and_birth_date_only | 1 | 1 |
| new_player | 1 | 1 |
| rejected_by_invariant | 8 | 62 |

## 6. State, reproducibility and integrity

| item | value |
|---|---|
| baseline (content-addressed, verified by restore) | `prospective-baseline-3253426224f7ebc0`, tree `5368ff8752320a76…`, 15 files, restored state `3253426224f7ebc0` |
| new state snapshot | `9c9d537c9c855dbe` |
| reconstruction (frozen engines) | `5ef8dbd26025a955` (194,697 matches, 218.9 s), `5ef8dbd26025a955` (194,697 matches, 213.0 s); deterministic **True** |
| replay (published Phase 2/3A/M4A) | passed **True**, 274 cases, max abs error 1.1e-16, max Elo error 0.0 |
| M4A frozen / configuration hash | `bc4e5c870c6e21b9` / `5e72010100065889` (match True) |
| Elo / feature configuration hash | `1c330d4db8801ff6` / `3d889b94456995d1` |
| promoted | **True**: `historical-v1-9c9d537c9c855dbe` (tree `6658abc655ca6743…`, restored state `9c9d537c9c855dbe`), registry `data/operations/snapshots/registry.json` |

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

Local per-match status SHA-256: `499b6e6b2574f650df73395e168b6397ced759bc6a22d11bb58c32651cad469a`.

## 8. Independent post-run checks (measured after promotion)

- Full test suite: **483 passed** (`uv run pytest`); ruff check and format clean.
- All 273 files under `data/` outside `data/operations/` (Phase 1–4C artifacts and the active `data/prospective/`)
  are byte-identical to the SHA-256 list taken before the run; the 99 git-tracked report files are unchanged.
- The baseline was restored a second time into a scratch directory, independently of the staging run: every object
  matched its manifest, the tree is byte-identical to `data/prospective/`, and its state ID is `3253426224f7ebc0`.
- Registry `data/operations/snapshots/registry.json`: version 0 `prospective-baseline-3253426224f7ebc0` (active
  Phase 4A directory) and version 1 `historical-v1-9c9d537c9c855dbe` (parent: version 0; not active).
- Gates unchanged: backfill INCOMPLETE, prospective predictions DISABLED, `state_availability_v1` disabled, external
  checkpoint publishers (GitHub, RFC 3161) disabled.

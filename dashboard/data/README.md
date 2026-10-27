# Dashboard dataset (data card)

Public, derived dataset used by the research dashboard. Built by `python -m tennis_data.dashboard.build` from this
repository's own Phase 1–3B outputs; `manifest.json` records the sources, row counts and SHA-256 of every file.

## Source and licence

- **Source:** Jeff Sackmann's ATP match data (https://github.com/JeffSackmann/tennis_atp), pinned mirror commit
  `83733587353df8a41f2fd4f516147d5aa83f5a8d`, licensed **CC BY-NC-SA 4.0**.
- **Changes:** records validated by the Phase 1 pipeline; Elo ratings, serve/return features and model predictions
  computed by this project.
- **Licence of this dataset:** CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). Research and education
  only; not for betting or commercial use.
- **Not included:** owner-supplied or other licensed sources (TennisMyLife exports, the Phase 5A 2026 backfill),
  identity approvals, local state snapshots or backups. The status file carries aggregate counts only.

## Files

| file | rows | content |
|---|---|---|
| `predictions.parquet` | 69,813 | every historical M4A prediction (walk-forward annual refits of the frozen specification, model id `M4A+raw`), 2002–2026, with the paired surface-Elo probability, the result and match context |
| `snapshots.parquet` | 218,054 | one pre-match row per player per tour-level match since 1991: global/surface Elo and Phase 3A serve/return indicators, point-in-time (state entering the match's weekly batch); ratings rounded to 0.1, rates to 1e-4 |
| `players.parquet` | 3,689 | player names, country, hand and career span within the snapshots |
| `status.json` | – | aggregate operational facts: archive commit, freshness probe of the active state, snapshot versions, backfill counts |
| `manifest.json` | – | sources, model and configuration hashes, coverage, row counts, SHA-256 |

Dates are **event start dates**: the archive has no match dates, and the frozen pipeline works in weekly batches.

## Provenance audit (2026-10-27)

The builder records the SHA-256 of its exact inputs in `manifest.json` (`inputs`): the Phase 3B predictions
(`selected.parquet`), the Phase 2 surface-Elo predictions, the Phase 3A serve/return feature tables and the Phase 1
validated archive tables. Rebuilding from those inputs reproduces the four data files byte for byte.

| check | result |
|---|---|
| predictions vs Phase 3B `M4A+raw` rows (configuration `5e72010100065889`) | 69,813 of 69,813; M4A probability, Elo probability and result identical |
| match IDs, player IDs, names and scores vs the pinned archive (commit `83733587`) | identical for every row |
| snapshot ratings vs Phase 2 output; snapshot rates vs Phase 3A features | within the stated rounding (0.1 rating points, 1e-4) |
| rows after the archive's last event (2026-05-25) | none: no Phase 5A backfill data |
| TennisMyLife player codes or match records, approval manifests, private snapshots | none |

**Transformations:** column selection and renaming; joins of the model, Elo, feature and archive tables on the
archive match ID; one row per player per match for the snapshots (A and B sides stacked, the result flipped for B);
rounding as above; aggregation of player names, country and hand from the archive. No values are imputed or
synthesised; missing indicators stay null.

**Personal data:** public professional sports records only: player names, country (IOC code), playing hand and
match results as published in the source archive. No birth dates, ages, heights, rankings or contact details.

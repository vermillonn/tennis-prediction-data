# Identity resolution review (owner-2026-10-26)

Phase 5A identity review of the mappings left unresolved by the version 1 backfill. Aggregate-only public report:
player-level evidence (provider codes, names, birth dates, candidates, per-player counts) is source-derived and kept
locally in `data/operations/exports/owner-2026-10-26/private/` and
`data/operations/identity_resolution/owner-2026-10-26/evidence.json` (git-ignored). No identity is approved by this
review; the owner's decisions are recorded in the local version 2 approval manifest.

## Method

1. **Archive assignment in the same matches** (January–May 2026 overlap): each provider record is paired with an
   archive match through the *opponent's* mapping, the round and the event week; the archive ID on the player's side
   is the archive's own identification (winners compared too). The disputed mapping is never used for the pairing
   (`operations/identity_resolution.overlap_assignment`, tested in `tests/test_identity_resolution.py`).
2. **Archive record attributes** (Jeff Sackmann's ATP archive, CC BY-NC-SA 4.0): full name, birth date, hand,
   height, country, Wikidata ID and archive match history; compared with the provider record (yes/no only).
3. Official ATP player IDs are not in the archive, so the match assignment and Wikidata IDs are the independent
   links. Famous names were never used as evidence.

## Findings

| group | mappings | finding | owner decision |
|---|---|---|---|
| rejected by the v1 automatic rule (`phase4a_decision`) | 8 | every competing candidate has **0 archive matches**; every paired overlap match uses the proposed ID with the same winner. 7: an undated, initials-only or different-country record surfaced by token or initial matching (overly broad candidate generation); 1: a duplicate archive record of the same person (same birth date, no history) | approved individually for v2 |
| defensible ambiguous, sufficient independent evidence | 7 | overlap assignment, or exact birth date with matching hand and height and no other candidate | approved individually for v2 |
| defensible ambiguous, manual judgment | 3 | only the birth date links the records, or no history on either record, or two archive IDs for one person with history and attributes on different IDs | kept quarantined |
| other ambiguous, new, `name_country` | 18 | not proposed | kept quarantined |

The automatic conservative rule is unchanged; the 15 approvals are individual, evidence-linked overrides, each checked
separately (`controlled_backfill.INDIVIDUAL_CHECKS`). No canonical ID is merged, deleted or rewritten.

## Recoverable matches (isolated simulations restored from version 0)

| scenario | accepted | Elo updates | held |
|---|---|---|---|
| version 1 (verified) | 1,093 | 1,063 | 159 |
| the 8 rejected only | 1,151 | 1,118 | 101 |
| the 15 approved for v2 | 1,210 | 1,176 | 42 |
| all 18 reviewed | 1,214 | 1,180 | 38 |

No scenario creates a new player identity or a Phase 4A quarantine. The recovered matches are concentrated on a few
active players (largest single-player recovery: 23 matches).

## Canonical duplication risks

- The archive contains 170 groups of records sharing a normalized full name and birth date (57 for players born
  1990 or later); none contains a version 1 approved canonical ID.
- 8 known duplicate-record situations touch this export (2 with archive matches on both records, which stay
  unresolved).
- One version 1 mapping (overlap-corroborated, holding all of the player's archive matches) has a likely duplicate
  archive record with a placeholder-like 1 January birth date and no matches; the v1 rule treats a different birth
  date as distinguishing. No history is split today; a later check should flag same-name records with 1 January
  birth dates.

## Public artifact policy

Possessing the owner's files does not establish redistribution rights; TennisMyLife raw redistribution is recorded
as prohibited without authorization, and publication of record-level derived data is unclear. Public reports on this
branch therefore carry aggregates, hashes, provenance and methods only; record-level outputs (identity packages,
approval manifests, per-match status, player-level review tables, score strings) are written to
`data/operations/exports/<label>/private/` (git-ignored). See `docs/operational_data.md` §16.

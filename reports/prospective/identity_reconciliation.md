# Phase 4A identity reconciliation

Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data.

## 1. Policy (`id-1`)

Provider player IDs are mapped to canonical (archive) player IDs by `prospective/identity.py`:

| evidence | category | usable in state |
|---|---|---|
| reviewed entry in `config/identity_overrides.toml` | verified (manual_override) | yes |
| provider ID already registered as a new player | unmatched (new_player, same ID) | yes |
| provider-declared canonical ID that exists and whose name agrees (no DOB conflict) | verified | yes |
| unique exact name, confirmed by birth date (or nationality when no DOB and no other candidate) | high_confidence | only after approval |
| several candidates, abbreviated/partial name only, declared ID contradicted, DOB conflict | ambiguous | never (held for review) |
| no canonical player shares the surname | unmatched -> new canonical ID | yes |

- Names are compared accent-, case- and punctuation-insensitively; compound surnames match on full or contained token sets (`Carreno Busta` / `Carreño-Busta`).
- A differing birth date eliminates a candidate. Nationality never does (players change representation); it is recorded as evidence.
- New players receive IDs from 9,000,000 upward in sorted provider-ID order; the registry (`data/prospective/identity/new_players.parquet`) is append-only and archive IDs are never changed.
- Every decision is persisted with method, candidates and evidence (`identity/decisions.parquet`).

## 2. Decisions in the current backfill

No provider player records were ingested (no authorized source accessible), so there are no live decisions and **0 unresolved identities**. That is an absence of data, not evidence that matching works on a real provider.

## 3. Behaviour on deliberately hard cases against the real player master

Hypothetical provider records (not ingested data) classified against the archive's 66,912 players:

| case | provider name | category | method | canonical id | candidates | usable |
|---|---|---|---|---|---|---|
| accent + hyphen | Félix Auger-Aliassime | high_confidence | name_country | 200000 | 1 | no |
| compound surname | Pablo Carreno-Busta | high_confidence | name_country | 105807 | 1 | no |
| particle surname | Juan Martin Del Potro | high_confidence | name_country | 105223 | 1 | no |
| abbreviated first name | A. Zverev | ambiguous | multiple_candidates | – | 2 | no |
| brothers, full name + DOB | Alexander Zverev | high_confidence | name_dob | 100644 | 1 | no |
| same name, no DOB | Carlos Alcaraz | ambiguous | unconfirmed_name_match | – | 2 | no |
| same name + DOB | Carlos Alcaraz | high_confidence | name_dob | 207989 | 2 | no |
| declared ID contradicts name | Jannik Sinner | ambiguous | declared_id_contradicted | – | 1 | no |
| declared ID consistent | Jannik Sinner | verified | declared_id | 206173 | 1 | yes |
| changed nationality, DOB match | Daniil Medvedev | high_confidence | name_dob | 106421 | 1 | no |
| DOB conflict | Daniil Medvedev | ambiguous | surname_only | – | 3 | no |
| unseen player | Qwerty Zyxwvutsrq | unmatched | new_player | – | 0 | yes |

The same behaviours are asserted in `tests/test_prospective_units.py` on a controlled master that contains identical names, brothers, compound and particle surnames and players without metadata.

## 4. Uncertainty

- Real providers may spell names in ways not covered here (transliteration of Cyrillic and Chinese names varies). Such cases become ambiguous, not wrong: they block the match until reviewed.
- Birth dates are the strongest secondary evidence, but the archive lacks them for some players; then only nationality and the absence of other candidates support a high-confidence match, which still needs approval.
- Matching quality on any real provider is unmeasured until its data can be accessed.

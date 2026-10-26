# Phase 5A: operational data readiness

Phase 5A asks one question: can the frozen system obtain trustworthy, current, legally accessible ATP data, and
what exactly stops it? It adds no new ingestion framework. Real provider records go through the unchanged Phase 4A
path: raw store, identity resolution, `rec-1` reconciliation, Phase 1 validation, state snapshot. Evidence is
judged by an explicit readiness evaluator.

**Status (2026-10-26):**

- Backfill INCOMPLETE and prospective predictions DISABLED (unchanged).
- Current ATP provider: **none reachable**.
- External timestamp authority: NOT CONFIGURED.
- Genuine prospective forecasts: 0.
- Readiness verdict: **NOT_READY**.

## 1. What already worked (reused unchanged)

| capability | where |
|---|---|
| immutable raw snapshots, content IDs | `prospective/providers/raw_store.py` |
| exchange format for converted authorized exports (results, fixtures, birth dates, provider IDs) | `prospective/providers/fixture.py` |
| identity resolution with human approval of high-confidence matches | `prospective/identity.py` |
| deterministic reconciliation, quarantine, versions, cross-source conflicts | `prospective/reconciliation.py` |
| Phase 1 validation, state snapshots, replay from the earliest affected batch | `prospective/backfill.py`, `prospective/state.py` |
| freshness checks, frozen-model integrity, status gate | `prospective/freshness.py`, `prospective/pipeline.py`, `config/prospective.toml` |
| ledger, evaluation, checkpoint anchoring | Phases 4B and 4C |

## 2. Gap analysis

| gap | resolution in Phase 5A |
|---|---|
| no reachable source for 2026-05-25 onward | **blocked by the environment's network policy and absent credentials.** Documented with exact hosts and variables (`reports/operations/provider_access.md`) |
| no record of which providers are reachable, and why not | `operations/access.py`: one probe per catalogued provider, classified (network / DNS / credentials absent / refused / rate limit / not found / schema / not probed) |
| a Sackmann-format source with ATP player codes (TML-Database) cannot use the `sackmann_csv` adapter (it treats IDs as archive IDs) | `operations/tml.py` converts to the Phase 4A exchange format; the unchanged Phase 4A path does the rest |
| no check that a new source's serve statistics mean the same thing | `operations/compatibility.py`: value-by-value comparison on matches present in both sources |
| no week-level gap audit | `operations/coverage.py` |
| no consolidated readiness decision | `operations/readiness.py` (pure, deterministic, never changes the gate) |
| identity defect: names in another order created new players | fixed in `prospective/identity.py` (§5), with regression tests |

## 3. Source availability (measured from this environment)

| source | result | kind of problem |
|---|---|---|
| Sportradar Tennis, api-tennis.com, TennisData.app, RapidAPI Tennis API | proxy refused CONNECT (403) | network restriction; credentials also absent (`SPORTRADAR_API_KEY`, `API_TENNIS_KEY`, `TENNISDATA_APP_KEY`, `RAPIDAPI_KEY`) |
| tennis-data.co.uk | proxy refused | network restriction. Results and odds only, no serve statistics |
| stats.tennismylife.org (TML live database) | proxy refused | network restriction. No documented bulk-download licence |
| ATP Tour website | not probed | terms of use do not permit automated collection |
| JeffSackmann/tennis_atp | 404 (raw), git requires credentials | source not publicly reachable |
| archive mirror (pinned) | data verified; ends 2026-05-25 | no gap data |
| TML-Database (GitHub) | data verified; 2025 and 2026-01-02 to 2026-01-21 | no gap data; serve statistics fail the compatibility study |

Only GitHub-hosted files and package registries are reachable. No reachable source contains a single ATP match
after the archive. The gap cannot be filled, and the system correctly refuses to issue forecasts.

### Minimum owner action

Any one of these:

1. **Allow a licensed API and provide its key.** Example: Sportradar.
   - Add `api.sportradar.com` to the environment's allowed domains (environment settings → Network access;
     https://code.claude.com/docs/en/cloud-environments#network-access).
   - Set `SPORTRADAR_API_KEY` as an environment variable of the environment.

   Its response layout must then be verified against real responses before mapping (Phase 4A refuses unverified
   mappings).
2. **Provide an authorized export** of the gap in Sackmann layout. Use `prospective-import --format sackmann_csv`
   if it uses archive IDs; otherwise convert it with the TML converter.
3. **Allow `www.tennis-data.co.uk`** for results only. Elo could then be updated for the gap; serve/return state
   would remain explicitly missing.

## 4. TML-Database conversion

- **Download:** files are pinned to a TML commit (`[tml].commit`), stored with SHA-256 and retrieval time, and kept
  in the git-ignored workspace. TML's README forbids redistribution, so only aggregates appear in reports.
- **IDs:** ATP alphanumeric codes are kept as provider IDs. No archive ID is claimed; the identity resolver decides
  the mapping.
- **Real-data quirks handled explicitly:**
  - the player file is CP-1252 encoded (detected and recorded);
  - levels `250`/`500` are mapped to `A`; unmapped levels pass through and Phase 4A quarantines them;
  - 489 matches of several 2025 events (US Open, Cincinnati, Winston-Salem, …) have no `match_num`, so the ID is
    derived from tournament, round and sorted player pair;
  - one 2026 record has no loser ID and is rejected and logged, never guessed;
  - two Davis Cup records fail Phase 1 plausibility (first serves in > service points) and are quarantined by
    Phase 4A.
- **Statistics:** serve statistics are labelled `tml_unverified`. Phase 4A keeps only `sackmann_v1` statistics, so
  they never reach the feature engine; outcomes are kept.

## 5. Identity reconciliation

The policy is unchanged:
- verified = manual override or a declared ID whose name agrees;
- high-confidence = unique exact name plus birth date, **awaiting approval**;
- ambiguous = never auto-resolved;
- unmatched = new ID.

On 459 real TML players: 430 are high-confidence awaiting approval, 28 are ambiguous and 1 is new. Every match
involving an unresolved player is held, so a wrong mapping cannot contaminate Elo or serve/return state.

**Defect fixed.** Three known archive players were classified as *new*:
- "Yunchaokete Bu", stored in the archive as first `Bu`, last `Yunchaokete`;
- "Mukund Sasikumar", stored as `Sasi Kumar Mukund`;
- "M Rifki Fitriadi", stored as first `Fitriadi`, last `M Rifqi`.

The surname index found no candidate, and the append-only new-player registry would have made the split
permanent. The resolver now also searches:
- the same name tokens in any order;
- the same birth date with a related name (a shared token, or one token equal to two joined).

Such players are held as **ambiguous** (`name_order_or_spelling_variant`). Regression tests are in
`tests/test_identity_regressions.py`; all Phase 4A identity tests still pass.

## 6. Statistical compatibility (method and result)

**Method:**
- Matching is analysis only: high-confidence mappings are used for the comparison, nothing is written.
- A match is the same unordered canonical pair and round, with tournament dates within 7 days.
- Zero or several candidates are counted, never guessed.
- Per field and per player, the report gives: exact agreement, mean difference, |difference| quantiles,
  missingness, and a split by completed and retired matches.

The thresholds were fixed before the study: at least 500 values and at least 98% exact agreement for every field
Phase 3A uses (`svpt`, `1stIn`, `1stWon`, `2ndWon`).

**Result on 2,952 matched matches (2025-01 to 2026-01):**
- Winners agree in 99.97% of matched matches and scores in 98.4%.
- Exact agreement: `svpt` 97.7%, `1stIn` 94.2%, `1stWon` 94.9%, `2ndWon` 94.7%. Not compatible.
- The differences are concentrated: 125 of 139 events agree exactly on every required field; the 2025 US Open
  (80% of matches differ), the Australian Open (34%) and the Next Gen Finals (83%) do not.
- 26 differences are winner and loser statistics swapped in one source.

Excluding those events now would be a post-hoc choice. TML statistics therefore stay out of the frozen feature
engine; a results-only use is possible after identity approval.

## 7. Corrections and late data

These are reused from Phase 4A and exercised with converted TML files in tests:
- byte-identical copies are collapsed;
- a newer `source_version` supersedes the older one, and both are kept;
- the same version reissued with other content is quarantined (`version_reissued`, tested with a converted TML
  file);
- cross-provider disagreement is quarantined;
- the archive is never overwritten;
- corrections replay the state from the earliest affected batch (`replay_from_batch`).

Committed ledger predictions are never modified, and late corrections cannot improve recorded forecasts
(Phase 4B/4C).

## 8. State reconstruction

The frozen Phase 2 Elo and Phase 3A feature engines run unchanged over the archive plus the accepted backfill. In
Phase 5A nothing in the gap was validated, so:
- the match component of the state is unchanged (`b7496182d8f41a85`);
- 0 Elo and 0 serve/return updates came from the gap.

The workspace backfill was run twice with identical summaries, and the full engine reconstruction was run twice
with identical state-table hashes (`reports/operations/backfill_status.md`). Completing a reconstruction is not
readiness.

## 9. Schedules, actual start and outcomes

There is no verified schedule or actual-start source. The exchange format and the Phase 4A/4B/4C paths support:
- fixtures with timezone-qualified scheduled starts, postponements and cancellations;
- replacement players (superseded match IDs);
- sourced actual starts, of which the earliest is used and disagreements are flagged.

These are tested with synthetic fixtures only. Actual starts are never inferred from tournament schedules.

## 10. Readiness evaluator

`operations/readiness.py` checks 16 criteria, grouped into these dimensions:
- historical completeness;
- statistical validity;
- identity integrity;
- freshness;
- schedule integrity;
- model integrity;
- audit infrastructure.

Every criterion listed in `[readiness].required` must PASS before activation can even be *proposed*. The evaluator
is a pure function of its evidence and never writes configuration; there is no bypass.

The current verdict, NOT_READY, is in `reports/operations/readiness_report.md`. The gate stays closed:
`backfill = "INCOMPLETE"`, `prospective_predictions = "DISABLED"`.

## 11. External timing (non-secret example for later authorization)

Genuine timestamp-verified prospective forecasts need an RFC 3161 TSA trusted by auditors (Phase 4C), and
optionally an independently controlled checkpoint repository. When authorized, `config/evaluation.toml` would
contain:

```toml
[publishers.rfc3161]
enabled = true
tsa_url = "https://<authorized-tsa>/tsr"                 # chosen and authorized by the operator
trusted_ca_files = ["config/tsa/<authorized-tsa>-ca.pem"]  # public CA certificate, distributed to auditors

[publishers.github]
enabled = true
repository = "<independent-owner>/<checkpoint-repo>"      # branch protection on, force pushes off
token_env = "LEDGER_CHECKPOINT_GITHUB_TOKEN"              # contents:write on that repository only
```

Neither is enabled. The TSA host must also be allowed by the environment's network policy.

## 12. Commands

| command | purpose |
|---|---|
| `tennis-data ops-access` | probe and classify provider access |
| `tennis-data ops-run [--as-of DATE] [--no-probe] [--no-fetch]` | full Phase 5A run and reports |
| `tennis-data ops-state-check` | frozen-engine reconstruction twice, compared |
| `tennis-data ops-reports` | re-render the reports from the last run |
| `tennis-data ops-readiness` | readiness verdict (exit code 1 while not ready) |
| `tennis-data ops-benchmark` | performance of conversion, identity, reconciliation, compatibility, schedule normalization |

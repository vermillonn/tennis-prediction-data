# Phase 4A source feasibility: current ATP data providers

Assessment date: 2026-10-25. This is a research document, written by hand. Each statement is marked:

- **verified**: observed directly from this environment;
- **documented**: read in public documentation or search results, not tested;
- **unknown**: could not be established.

No credentials were available, and none were requested or used.

**Status: every real-provider integration is UNVERIFIED.** No provider could be accessed, so no adapter has been
run against real responses. Historical backfill: **INCOMPLETE**. Prospective predictions: **DISABLED**.

## 1. What was actually accessible

The Phase 4A container runs behind an egress proxy with an allow-list. Results of direct access attempts on
2026-10-25 (**verified**):

| host | purpose | result |
|---|---|---|
| tennisdata.app | candidate 1 (downloads page) | HTTPS CONNECT refused by the environment proxy (403, policy); web fetch: DNS not resolvable |
| docs.tennis-api.com, tennis-api.com | candidate 2 (API docs) | proxy refused (403); web fetch: DNS not resolvable |
| developer.sportradar.com, api.sportradar.com | candidate 3 (docs, API) | proxy refused (403); web fetch: DNS not resolvable |
| www.tennis-data.co.uk | results + odds files | proxy refused (403); web fetch: DNS not resolvable |
| huggingface.co (archive mirror) | alternative mirror | proxy refused (403) |
| atptour.com, tennis-db.com, sofascore, tennisabstract, ultimatetennisstatistics | other sites | proxy refused (403) |
| github.com/JeffSackmann/tennis_atp | upstream of the historical archive | **HTTP 404** (repository not publicly available) |
| raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive | pinned historical archive | **reachable**. `main` serves `atp/atp_matches_2026.csv` byte-identical to the pinned commit `83733587…` (SHA-256 `ceb157ae…`), so there is **no newer data**. Its README says the ATP snapshot comes from upstream commits made in June 2026. |

**No provider of post-2026-05-25 ATP data was accessible.** Every candidate was blocked by this environment's
network policy before any authentication step. Nothing was bypassed, and no third-party scraper or mirror of
restricted content was used. To test a provider, a person must add its host to the environment's network allow-list
(Network access settings of the cloud environment) and supply a key as an environment variable
(`SPORTRADAR_API_KEY` for the adapter in this phase).

## 2. Candidate comparison

"unknown" means the item could not be checked. It does not mean the capability is absent.

| criterion | tennisdata.app | tennis-api.com | Sportradar Tennis v3 | Sackmann-format continuation | tennis-data.co.uk |
|---|---|---|---|---|---|
| access tested | no (blocked) | no (blocked) | no (blocked) | yes (archive mirror, no new data) | no (blocked) |
| May 24 – Oct 25, 2026 results | unknown | unknown | documented in general (live + historical feeds); not checked for these dates | not available (mirror ends 2026-05-25) | unknown |
| upcoming schedules | unknown | unknown | documented (schedule feeds) | no (results only) | no |
| player / tournament IDs | unknown | unknown | documented (provider IDs, `sr:` style) | archive IDs = canonical IDs | names only (per older descriptions) |
| match dates / timestamps | unknown | unknown | documented (scheduled start times) | none (event start date only) | match date (per older descriptions) |
| service/return point statistics | unknown | unknown | documented: a "Service Points Won" statistic exists; which endpoint, and whether counts match `svpt`/`1stWon`/`2ndWon`, unknown | yes (`sackmann_v1`, the only convention the Phase 3A estimators use) | no (results and odds) |
| result status (RET, W/O) | unknown | unknown | unknown | yes | partly (comment column, per older descriptions) |
| surface / tournament level | unknown | unknown | unknown | yes | yes |
| ATP main tour / Grand Slams | unknown | unknown | documented (coverage tiers per competition) | yes | yes |
| qualifying | unknown | unknown | unknown | separate files (not used by the frozen pool) | no |
| authentication / pricing | search results describe an iOS app (about $0.99 one-off) with a cloud database; no data download found | no documentation found | API key. Trial documented as 30 days, 1,000 requests per rolling 30 days, 1 query per second. Production pricing by sales contact | free; CC BY-NC-SA 4.0 | free download (per older descriptions) |
| licence / redistribution | unknown | unknown | commercial terms, not reviewed; typically restrict storage and redistribution | non-commercial, share-alike | unknown |
| corrections / history | unknown | unknown | unknown | git history of the mirror | unknown |
| automated ingestion | no API found | unknown | yes, in principle (REST, raw storage implemented) | yes (CSV import implemented) | yes, in principle (files) |

Other providers that came up in search results (**documented** only, not assessed in depth): Goalserve tennis feed
(about $150 per month, live statistics), Statscore (demo-based sales), Live Tennis API (free tier of 100 requests per
day, statistics coverage unclear). Apify "scraper" actors were ruled out: they repackage third-party websites, and
their licence status is doubtful.

## 3. Recommended source strategy

1. **Historical backfill (2026-05-25 → 2026-10-25), first choice: a Sackmann-format continuation** from an authorized
   source, either an updated archive mirror or a licensed export in the same columns. It is the only source whose
   statistics are known to follow the convention the frozen Phase 3A estimators need. Import path:
   `tennis-data prospective-import <file> --format sackmann_csv`, then `prospective-backfill`.
2. **Schedules, and backfill if (1) is unavailable: a licensed commercial API (Sportradar Tennis v3 is the
   best-documented candidate).** Before any of its records update the state:
   - write and fixture-test the response mapping against real responses;
   - verify that its service-point counts reproduce `sackmann_v1` on an overlap period (January–May 2026 is in the
     archive). Until both are done, its statistics are stored as raw snapshots only and are excluded from the
     serve/return state.
3. **Secondary verification** of outcomes and dates (no statistics): any second licensed source. Outcome
   disagreements quarantine the match. They are never resolved automatically.
4. **Local immutable raw snapshots** for reproducibility (implemented: content-addressed raw store).

## 4. Expected coverage and operational cost (estimates)

- **Volume to backfill:** about 1,441 tour-level matches in about 62 events (Wimbledon, the US Open and Masters
  events included). This is based on the same weeks one year earlier; see `backfill_coverage.md`. In that year about
  88% had usable statistics.
- **Request budget (Sportradar-style API):** about 150 daily schedule calls plus one summary or statistics call per
  match comes to roughly 1,600 or more requests for the backfill, which exceeds the documented trial quota (1,000 per
  30 days). Ongoing prospective use needs daily schedules plus per-match results, roughly 1,500 to 2,000 requests per
  month. **A production licence is likely required.**
- **Cost:** not publicly listed for Sportradar. Goalserve lists about $150 per month (documented, not verified).

## 5. Unresolved constraints

- **Licences.** The archive is CC BY-NC-SA 4.0 (non-commercial, share-alike). Any commercial API's terms on storing
  raw responses, deriving ratings and publishing predictions must be reviewed before use. Mixing CC BY-NC-SA data with
  a commercial feed may restrict how outputs can be used.
- **Statistics compatibility** of any non-Sackmann provider is unverified (definitions of service points,
  retirements, tie-break counting).
- **Upstream status.** Jeff Sackmann's repository returned 404 on 2026-10-25. Whether the dataset is maintained
  elsewhere is unknown.
- **Access** from this environment requires an allow-list change by the environment owner.

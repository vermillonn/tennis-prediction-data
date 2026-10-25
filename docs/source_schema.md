# Source schema and coverage (as inspected)

Everything here was measured on the real files at the pinned commit, before writing any
transformation. Re-check this document whenever `[source].commit` changes.

## Source

- Repository: <https://github.com/Aneeshers/tennis-sackmann-archive>
- Pinned commit: `83733587353df8a41f2fd4f516147d5aa83f5a8d` (2026-06-25, "Archive of Jeff Sackmann tennis
  datasets"). The archive's README says the `atp/` snapshot comes from upstream `JeffSackmann/tennis_atp`
  commits made in June 2026. The upstream repository itself is not used.
- License: CC BY-NC-SA 4.0, data compiled by Jeff Sackmann. The archive's `LICENSE`, `README.md`,
  `atp/UPSTREAM_README.md` and `atp/matches_data_dictionary.txt` are ingested with the data for provenance.

## Files used (`atp/` directory)

| file pattern | family | seasons | rows | notes |
|---|---|---|---|---|
| `atp_matches_YYYY.csv` | `tour` (default) | 1968-2026 | 199,389 | tour-level main draws, Davis Cup, Olympics, Finals |
| `atp_matches_qual_chall_YYYY.csv` | `qual_chall` | 1978-2026 | 240,471 | tour-level qualifying (Q1-Q4) and Challenger main draws |
| `atp_matches_futures_YYYY.csv` | `futures` | 1991-2026 | 523,355 | ITF Futures / satellites |
| `atp_players.csv` | reference | - | 66,912 players | `player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id` |
| `atp_rankings_{70s..20s,current}.csv` | reference | 1973-08-27 - 2026-06-08 | 3,420,595 | `ranking_date,rank,player,points`; 2,333 ranking dates |

Not used: `atp_matches_amateur.csv`, `atp_matches_doubles_*.csv` (doubles, team stats), and the `wta/`
and `slam_pointbypoint/` folders. 2026 is a partial season (through June 2026).

## Match-file columns

All three singles families share the same 49 columns in the same order (checked on every file):

```
tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num,
winner_id, winner_seed, winner_entry, winner_name, winner_hand, winner_ht, winner_ioc, winner_age,
loser_id,  loser_seed,  loser_entry,  loser_name,  loser_hand,  loser_ht,  loser_ioc,  loser_age,
score, best_of, round, minutes,
w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced,
winner_rank, winner_rank_points, loser_rank, loser_rank_points
```

The layout is **winner/loser oriented**: the winner is always in the `winner_*` / `w_*` columns, and
`score` lists games from the winner's perspective. This is why the canonical table re-orients rows into
player A / player B. Using the source orientation directly would leak the label.

### Null rates (% of rows)

| column group | tour | qual_chall | futures |
|---|---|---|---|
| `tourney_*`, `match_num`, ids, names, `score`, `best_of`, `round` | 0.0 | 0.0 | 0.0 |
| `surface` | 1.5 | 0.0 | 0.0 |
| `draw_size` | 0.4 | 0.0 | 0.0 |
| serve stats (all 18 together) | 48.9 | 44.5 | 95.4 |
| `minutes` | 50.4 | 46.3 | 95.3 |
| `winner_rank` / `loser_rank` | 18.0 / 22.2 | 3.6 / 6.4 | 7.8 / 18.6 |
| `winner_ht` / `loser_ht` | 8.4 / 14.7 | 7.4 / 15.6 | 46.3 / 62.1 |
| `winner_seed` / `winner_entry` | 62.8 / 91.0 | 56.1 / 84.7 | 55.9 / 79.7 |

Serve statistics are almost always all-present or all-absent (tour: 101,925 complete, 97,461 none,
3 partial). Tour-level stats start in 1991, as the upstream README says.

### Observed value domains

- `surface`: Hard, Clay, Grass, Carpet, and null (2,990 tour rows). The futures files also contain
  lowercase `clay` (1,144) and `carpet` (93).
- `tourney_level`: tour `A, G, M, D, F, O`. `O` (Olympics) is not in the upstream dictionary.
  qual_chall `C, A, G, M`; futures `S, 15, 25` (ITF prize tiers).
- `round`: `R128, R64, R32, R16, QF, SF, F, RR, BR, ER, Q1, Q2, Q3`.
- `best_of`: `3, 5`, plus `1` (36 tour rows: one-set exhibitions and round-robin formats).
- `tourney_date`: always 8 digits (`YYYYMMDD`), usually the Monday of the event week. 1,552 tour rows have a
  `tourney_date` in the year *before* the file's season (December events that count toward the next season).
- `score` markers: `RET` (4,012), `W/O` (1,301), `DEF` (157), `ABD` (22), `?`-garbled (14),
  `Played and unfinished` (9), `NA` (6), `Walkover` (6), `Def.` (4), `UNK`, and 3 nulls (tour).
- `winner_entry`: `WC, Q, LL, PR, ALT, Alt, SE, ITF, NG, UP, W`.

### No match timestamps

The files contain no match date and no time. The only temporal field is `tourney_date`, the event start.
`match_num` is not chronological ("often starting from 1, sometimes counting down from 300, and
sometimes arbitrary"). The canonical table therefore keeps `match_date` / `match_start_time` null and
offers `round_order` for ordering within an event.

## Anomalies found during inspection

| finding | rows (tour) | handling |
|---|---|---|
| Placeholder player `199999` ("U Unknown") shared by many people, sometimes on both sides | 90 | quarantine `placeholder_player_id` |
| `winner_id == loser_id` | 3 | quarantine `same_player_both_sides` |
| Identical rows except `match_num` (Davis Cup 2024/2025, Denver 1971, Barcelona 1969) | 5 copies | quarantine copies `duplicate_record`, keep first |
| Same pair twice in one round-robin event with different scores | 11 pairs (22 rows) | legitimate, kept |
| Impossible serve stats (negative, breaks > service games, ...) | 15 | quarantine (stats category) |
| `best_of` contradicting the score (1970s Slam best-of-3 rounds labelled 5; NextGen Finals; Davis Cup) | 1,060 | warning `score_inconsistent_with_best_of` |
| Names differing from `atp_players.csv` (Bob/Robert, hyphenation) | 1,439 | warning |
| Duplicate `(ranking_date, player)` keys in the rankings files | 780 keys | reported; the comparison uses the best rank |
| Match-file rank equals a list published 1-2 days *after* `tourney_date` | ~9,800 appearances | reported (see report) |

Additional findings that only affect the non-default families are in `reports/all_families/`. The main one:
US Open qualifying from 2018 on records `SvGms = 0` for both players while break points are populated.

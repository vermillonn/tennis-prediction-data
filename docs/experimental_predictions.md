# Experimental single-match predictions

`tennis-data predict-experimental` gives the frozen M4A probability for one named match, for personal research. Its
output is **always labelled EXPERIMENTAL**. It is **never** an independently verified prospective forecast, and it is
never written to the prediction ledger. The official prospective gate stays DISABLED.

```bash
uv run tennis-data predict-experimental \
    --player-a "Jannik Sinner" --player-b "Carlos Alcaraz" --surface Hard --date 2026-11-15
# optional: --event-start YYYY-MM-DD (tournament start; default --date), --json (full result),
#           --state-version v0|v1|v2 (read-only registered snapshot; default: the active state, version 0)
```

It needs the local research state (`data/`, built by the full rebuild in [reproducibility.md](reproducibility.md)).
A public clone without `data/` is refused with "no verified state is available". One run replays the full Elo and
feature engines. On the maintainer's machine that took about four minutes.

## What it does

The command reuses the Phase 4A inference path unchanged ([prospective_data.md](prospective_data.md)):

1. **State.** It loads the active state (historical version 0, archive through May 2026). The state is verified by
   recomputing its content hash against the recorded one. Inactive backfill snapshots (versions 1 and 2) are not
   used.
2. **Players.** It resolves each player against the archive player master, accepting a full name (any order), a
   unique surname or an ATP player ID. Only players with results in the state are accepted. Ambiguous names are
   refused with the candidates listed.
3. **Pending row.** It appends an outcome-free pending row. The unchanged engines give it the Elo and serve/return
   state frozen at the start of its weekly batch, the same rule every historical prediction used.
4. **Model.** It applies the hash-checked frozen production artifact (`bc4e5c870c6e21b9`, raw probabilities), with no
   refitting or recalibration. A missing input is set to the frozen neutral value 0, and the output reports this.
5. **Report.** It reports:
   - both players' M4A probabilities and the surface-Elo baseline;
   - the prediction cutoff, the data cutoff (when the last results and point statistics entered the state), the state
     version and snapshot ID, and the model artifact;
   - every freshness check;
   - the Elo ratings and statistics history of both players.

## Selecting a registered historical state (`--state-version`)

Without the option, the command uses the active state (version 0). With `--state-version v0|v1|v2`, it loads a
registered snapshot from the Phase 5A store `data/operations/snapshots/` **read-only**. Before anything is loaded, it
checks:

1. **Registry and manifest.** The registry entry exists and agrees with the manifest (name, state ID, tree hash).
   Every stored object matches its SHA-256, and the tree hash matches.
2. **Parent provenance** (v1, v2). The parent is the registered version-0 baseline, its manifest verifies, and it was
   built on the same archive commit.
3. **State ID.** The snapshot is restored into a temporary directory and every file is re-checked. The state's content
   hash is then recomputed and must equal the recorded, manifest and registry IDs.

The snapshot store, the registry and the active `data/prospective/` directory are never written. Selecting v2 does
not activate it, and version 0 stays active.

**Point-in-time reconstruction.** Whatever state is used, results released at or after the prediction cutoff are
removed before the engines run. The result is labelled `POINT-IN-TIME RECONSTRUCTION`, with the derived state ID and
the number of results removed. The engines already ignore such results for this match, so the probability is the
same; the removal makes that guarantee explicit, so that a later, more complete snapshot cannot leak information. For
example, v2 used for a match on 2026-08-31 drops the 505 results released from 2026-08-29 onwards.

**What v2 is.** v2 is a results-only owner-supplied backfill (June to October 2026), inactive and not prospective. It
advances Elo but adds no point statistics: serve/return inputs still end in May 2026, which is labelled
`MIXED FRESHNESS`. Its coverage is incomplete (1,210 results in 20 weeks), and matches of players with unresolved
identities were held out. The output reports each player's results after the archive end, because a missing result
cannot be told apart from inactivity.

## Refusals

The command refuses rather than guesses when:

- **the prediction cutoff** (the start of the match's weekly batch) is before the production fit's training cutoff
  (2026-06-06), because the coefficients were estimated on later results. Historical matches have walk-forward
  predictions in the dashboard instead;
- **the state** cannot be loaded or verified: a missing registry, version, manifest or object; a hash, registry or
  parent mismatch; or a recomputed state ID that differs from the recorded one;
- **a player** is unknown, ambiguous or has no results in the state, or both names resolve to the same player;
- **the surface or event start** is invalid.

## Labels

| label | meaning |
|---|---|
| `EXPERIMENTAL` | always present: research output from historical state, not a prospective forecast |
| `STALE DATA` | at least one freshness check fails (see `config/prospective.toml [freshness]`) |
| `MIXED FRESHNESS` | results (Elo) are fresh enough, but the serve/return statistics fail freshness |
| `INACTIVE STATE vN` | a registered, inactive snapshot was used read-only |
| `POINT-IN-TIME RECONSTRUCTION` | results released at or after the cutoff were removed from the selected state |

With version 0 (last results May 2026), every match after mid-2026 is **STALE DATA**. The ratings and form
reflect the players as of the end of May 2026, and nothing that happened afterwards is used.

## Not supported

- **Tournament format.** The frozen M4A has no best-of-three or best-of-five input, so there is no format option.
  Grand Slam matches were predicted slightly worse than by Elo in the historical evaluation.
- **Live data.** The command adds no data source. Making v2 (or a later backfill) the active state remains an owner
  decision through the operational process; this command never does it.

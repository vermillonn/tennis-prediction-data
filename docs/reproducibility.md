# Reproducibility guide

There are two levels of reproduction:

- **A public clone** can run the dashboard and the test suite, and verify the published dataset, with no data
  download.
- **A full research rebuild** downloads the public archive and regenerates every Phase 1–3B output.

Owner-supplied sources (the Phase 5A TennisMyLife exports) are **not** in this repository and are not needed for
any published research result.

## Environment

| item | value |
|---|---|
| Python | 3.11 or later. CI runs 3.11, and a clean-clone dashboard start was also checked on 3.12 |
| dependency manager | [uv](https://docs.astral.sh/uv/). `uv.lock` pins every package; CI uses uv 0.11.32 with `uv sync --locked` |
| dependency groups | default (Polars, DuckDB, NumPy); `dev` (pytest, SciPy for optimiser cross-checks); `dashboard` (Streamlit, Plotly) |
| dashboard only | `dashboard/requirements.txt` pins Streamlit, Plotly, Polars and NumPy for pip or Streamlit Community Cloud |
| determinism | Every run records configuration hashes. Random seeds are fixed in config (for example, the bootstrap seed 20261025), and rebuilds are byte-identical for a given `uv.lock` (Parquet bytes depend on the pinned Polars version) |

## 1. Public-clone reproduction (no data download)

```bash
git clone https://github.com/vermillonn/tennis-prediction-data.git
cd tennis-prediction-data
uv sync --group dashboard
```

### Launch the dashboard

```bash
uv run streamlit run dashboard/streamlit_app.py        # http://localhost:8501
```

The app reads `reports/` and the committed public dataset `dashboard/data/`. It imports the project code from
`src/`, and needs no secrets or network access. Without uv, run
`pip install -r dashboard/requirements.txt && streamlit run dashboard/streamlit_app.py`.

### Run the automated tests

```bash
uv run pytest            # 532 tests; synthetic fixtures, no network, about a minute
```

The same command runs in GitHub Actions on every pull request and on every push to `main`
([workflow](../.github/workflows/ci.yml)). The suite covers:

- validation rules and byte-reproducible builds;
- Elo chronology, and point-in-time feature correctness under shuffling and future perturbation;
- player-order symmetry;
- frozen-model hash checks against the committed fixture `tests/fixtures/m4a_production/`;
- ledger integrity and the prospective gates;
- dashboard metric consistency with the published reports;
- documentation links and figures.

### Verify the public dataset

`dashboard/data/manifest.json` records the SHA-256 of every dataset file and of the exact inputs it was built from.
To check the files:

```bash
python -c "import hashlib,json,pathlib; d=pathlib.Path('dashboard/data'); m=json.loads((d/'manifest.json').read_text()); [print(f, 'OK' if hashlib.sha256((d/f).read_bytes()).hexdigest()==v['sha256'] else 'MISMATCH') for f,v in m['files'].items()]"
```

The dashboard tests also recompute the published development and exploratory metrics from
`predictions.parquet`, and check that they match `reports/modeling/` exactly.

### Inspect the frozen model and the gates

- The frozen hash (`bc4e5c870c6e21b9`) and the configuration hash (`5e72010100065889`) are recorded in
  `reports/modeling/modeling_summary.json` and `dashboard/data/manifest.json`.
- The gates are plain TOML:
  - `config/prospective.toml` holds `[status]`;
  - `config/availability.toml` holds `enabled`;
  - `config/evaluation.toml` holds the publication backends.

## 2. Full research reproduction (downloads the public archive)

The research outputs are derived only from Jeff Sackmann's ATP data. `tennis-data ingest` downloads it from the
pinned mirror commit and checks every file's SHA-256. Use of the data falls under its CC BY-NC-SA 4.0 licence
(non-commercial).

```bash
uv run tennis-data run              # Phase 1: ingest + build + report (validated table, quarantine, data-quality report)
uv run tennis-data verify           # re-check every raw file against its recorded SHA-256
uv run tennis-data elo-backtest     # Phase 2: tune Elo on 1991-2009, write predictions, evaluate 2010-2026
uv run tennis-data features-build   # Phase 3A: point-in-time serve/return features + coverage report
uv run tennis-data model-backtest   # Phase 3B: development, selection, freeze, walk-forward evaluation
git diff --stat reports/            # no changes expected: outputs are deterministic
```

- **Outputs.** Everything goes to `data/`, which is git-ignored. Regenerated reports overwrite `reports/`.
- **Provenance.** `reports/build_summary.json` and `data/modeling/manifest.json` hold the input and output hashes, so
  two machines can compare builds directly.
- **Existing clone.** To use an existing clone of the archive instead of downloading, set
  `[source].mode = "local"` and `local_path` in `config/default.toml`. The clone's `HEAD` must equal the pinned
  commit.
- **Resources.** The maintainer's machine recorded these runs:
  - a full Phase 2 + 3A engine reconstruction over 193,634 matches: about 230 s, peak RSS about 1.9 GB
    ([backfill status](../reports/operations/backfill_status.md));
  - a Phase 4A replay run: about 227 s, peak RSS about 2.4 GB
    ([verification record](../reports/prospective/verification_record.md)).

  These figures vary by machine. Phase 3B fits about 1,360 logistic models.
- **Rebuilding the dashboard dataset** (maintainers): `uv run python -m tennis_data.dashboard.build`.
  - It reads the local Phase 1–3B outputs under `data/`, and writes `dashboard/data/` and its manifest.
  - `predictions.parquet`, `snapshots.parquet` and `players.parquet` are determined by those inputs.
  - `status.json` summarises the local operational state as aggregate counts. It needs the Phase 4A manifest
    (`uv run tennis-data prospective-run`), and its backfill counts appear only where the local Phase 5A records
    exist.

### Operational phases (not needed for any published result)

These commands exercise the gated infrastructure; prerequisites are in each phase's documentation. Prospective
commitments are refused while the gate is DISABLED.

```bash
uv run tennis-data prospective-run    # Phase 4A: gap audit, backfill, replay, readiness
uv run tennis-data ledger-init && uv run tennis-data ledger-verify    # Phase 4B ledger
uv run tennis-data eval-run --write-reports                           # Phase 4C read-only evaluation
uv run tennis-data eval-demo --out /tmp/demo                          # labelled SYNTHETIC demonstration
uv run tennis-data ops-readiness                                      # Phase 5A verdict (exit 1 while NOT_READY)
```

Phase 5A imports owner-supplied files (`prospective-import`, `ops-run`). They are not redistributed, and their
record-level outputs stay in git-ignored local storage. See [operational_data.md](operational_data.md).

## 3. Data layer reference

### Outputs

| path | content |
|---|---|
| `data/raw/<owner>__<repo>/<commit>/...` | raw files, untouched, with a provenance manifest (URL, commit, retrieval time, SHA-256, size) |
| `data/validated/matches/season=YYYY/part-0.parquet` | the validated canonical table (zstd Parquet) |
| `data/quarantine/matches.parquet` | quarantined rows, with reasons, categories and the original raw record |
| `data/tennis.duckdb` | DuckDB views over the Parquet files |
| `reports/data_quality_report.md`, `reports/tables/*.csv` | the data-quality report |

### Query interface

```bash
uv run tennis-data query "SELECT surface, count(*) FROM matches GROUP BY ALL ORDER BY 2 DESC"
```

The available views are:

- `matches`, the validated table ([data dictionary](data_dictionary.md));
- `quarantine`;
- `player_matches`, with one row per player per match;
- `players`;
- the later phases' views: `elo_predictions`, `serve_return_features`, `model_selected_predictions`.

From Python, `tennis_data.storage.query.connect(load_config())` returns the same DuckDB connection.

### Validation model

Every check is a registered rule with a code, a severity and a category (`src/tennis_data/validation/rules.py`):

- An `error` sends the row to quarantine. Examples:
  - missing identifiers;
  - impossible serve statistics;
  - duplicate or conflicting records.
- A `warning` keeps the row and records a flag.
- Legitimately missing statistics are neither: they stay null, with `stats_status = 'none'`.

Retirements, walkovers and defaults are kept and labelled by `match_status`. Quarantine is not deletion: a row
quarantined only for statistics still has a trustworthy result.

### Source limitations

- **No match dates or times.** Only the tournament start date exists, and `round_order` is an ordinal, not a clock.
- **Rank timing.** About 9,800 tour appearances match the ranking list published 1–2 days after `tourney_date`.
- **Biographical fields.** Height and hand are current values, not time-stamped.
- **`best_of`.** It contradicts the score in some 1970s Slams, the NextGen Finals and some Davis Cup rubbers. These
  rows are flagged, not corrected.
- **Coverage.** Serve statistics start in 1991 and are thin for Davis Cup.

Details: [source_schema.md](source_schema.md) and the [data-quality report](../reports/data_quality_report.md).

# Research dashboard (Phase 5B)

An interactive Streamlit presentation of the frozen ATP match-winner model **M4A** and its historical evaluation
against surface Elo. It is a portfolio and research demonstration: **retrospective results only, no live
forecasts, no betting advice**.

## Run it

```bash
git clone https://github.com/vermillonn/tennis-prediction-data && cd tennis-prediction-data
uv sync --group dashboard            # or: pip install -r dashboard/requirements.txt
uv run streamlit run dashboard/streamlit_app.py
```

The app works from a clean clone: it reads the published reports in `reports/` and the committed public dataset in
`dashboard/data/`, and imports the project code from `src/` (no package install needed). Nothing heavy runs on page
load: no state reconstruction, model fitting or network access. Data files are cached per server process.

## What each section shows

| section | content | source |
|---|---|---|
| **Overview** | research question, method, the six M4A inputs and coefficients, model and configuration hashes, evaluation sample sizes, reproducibility (tests, leakage audit, dataset integrity), and the retrospective notice | published `reports/modeling/modeling_summary.json`, dataset manifest |
| **Match explorer** | every recorded historical M4A prediction (2002–2026) with the paired surface-Elo probability and the actual winner; filters by player, opponent, surface, season, tournament category and evaluation period; per-match inspector; confidence, accuracy-by-confidence, M4A-versus-Elo scatter and largest misses | `dashboard/data/predictions.parquet` |
| **Player comparison** | two players side by side as of a chosen date: global and surface Elo, serve/return indicators, sample sizes, shrinkage and missingness, rating history, recent form, and recorded predictions when they met. **No win probability** | `dashboard/data/snapshots.parquet` |
| **Model evaluation** | published log loss, Brier score, accuracy and ECE per period with block-bootstrap intervals; reliability diagrams; differences by surface, tournament category, Elo confidence and statistics history; stability over periods; paired comparison on any subset | published tables in `reports/modeling/tables/`, predictions dataset |
| **Data & status** | operational gates (read live from `config/`), historical coverage, latest archive event, the 2026 backfill as aggregate counts, freshness of the active state, and why live predictions are unavailable | `config/*.toml`, `dashboard/data/status.json`, `reports/build_summary.json` |

## Retrospective versus prospective

- **Retrospective (this dashboard).** Each historical prediction was produced by an annual walk-forward refit of the
  frozen M4A specification, using only results released before the match's weekly batch. These are honest
  point-in-time reconstructions, but the development period (2002–2009) was used to select the model and the
  exploratory period (2010–2026) was seen while building the Elo baseline. **Neither is an untouched test.**
- **Prospective (not available).** A genuine forecast must be made and independently timestamped *before* the match,
  from data actually available then. The infrastructure exists (Phases 4A–4C: inference path, append-only ledger,
  checkpoint anchoring), but the gate is `DISABLED`.
- The production fit (frozen hash `bc4e5c870c6e21b9`, configuration `5e72010100065889`) uses all labels before
  2026-06-06, so it has no out-of-sample history of its own; the dashboard never presents it as if it had.

## Public and private data

| available publicly (committed) | local only (never published) |
|---|---|
| published reports (`reports/`), configuration, code | owner-supplied TennisMyLife files and anything derived from them at record level |
| `dashboard/data/` (CC BY-NC-SA 4.0, derived from Jeff Sackmann's archive; see its data card) | Phase 5A backfill snapshots (v1, v2), identity approvals, private evidence, backups |

`dashboard/data/` is rebuilt by a maintainer with `python -m tennis_data.dashboard.build`, which reads only the
archive-derived research outputs under `data/` and writes a manifest with SHA-256 hashes. Without `dashboard/data/`
the app still serves the Overview and the published evaluation (aggregate mode) and explains what is missing.

## Modeling limitations (shown in the app where relevant)

- The archive has no match dates; chronology is weekly and the dates shown are **event start dates**.
- Late-round results of an event can be released into the rating state at the end of the event's batch, before
  the weekend matches were played; this is a documented property of the frozen batching (Phase 5A chronology audit).
- Missing feature differences are set to neutral with a missingness indicator; much of M4A's gain over Elo is a
  cold-start correction for players with little history.
- Bootstrap intervals resample weekly batches and ignore dependence through shared players; they may be too narrow.
- Tour-level ATP men's singles only; serve/return statistics from 1991.

## What live predictions would require

1. Point-level serve/return statistics up to the week of each match (the archive ends in May 2026; the 2026
   backfill carries results only).
2. A verified, authorised live provider for schedules and results.
3. A complete backfill to the prospective start, with every player identity resolved (21 remain under review).
4. Independent timestamping of forecasts (RFC 3161 authority or an independently controlled checkpoint repository).
5. A reviewed configuration change of the status gate. Enabling `state_availability_v1` for inference is part of
   that work (Phase 5C).

## Deployment (Streamlit Community Cloud, not deployed)

The repository is ready for Streamlit Community Cloud: main file `dashboard/streamlit_app.py`, dependencies
`dashboard/requirements.txt` (found next to the entry point), theme in `.streamlit/config.toml`, Python 3.11 or
later. The app needs no secrets. Deploying is the owner's decision; it publishes only what is already public in the
repository.

## Tests

`tests/test_dashboard.py` (no browser, no private data): metric and calibration consistency with the published
reports, dataset integrity and licensing restrictions, filters, prediction lookup, player-order symmetry,
point-in-time snapshots, missing data, closed gates, and a headless start-up of every section (`AppTest`), with and
without the dataset. CI installs the dashboard group (`uv sync --locked --group dashboard`).

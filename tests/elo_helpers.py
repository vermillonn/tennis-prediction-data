"""Synthetic inputs for Elo tests."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from conftest import match_row, write_config

SAT0 = date(2020, 1, 4)  # a Saturday: batch starts with week_start = "saturday"


def frame(rows: list[tuple]) -> pl.DataFrame:
    """Engine input from (match_id, week_index, a, b, a_wins[, surface[, release_offset_weeks]]) tuples."""
    out = []
    for r in rows:
        mid, week, a, b, a_wins = r[:5]
        surface = r[5] if len(r) > 5 else "Hard"
        rel = r[6] if len(r) > 6 else 0
        pb = SAT0 + timedelta(weeks=week)
        out.append(
            {
                "match_id": mid, "pool": "p", "prediction_batch": pb, "release_batch": pb + timedelta(weeks=rel),
                "player_a_id": a, "player_b_id": b, "target_a_wins": a_wins, "surface_key": surface,
                "is_target": True, "is_update": True,
            }
        )  # fmt: skip
    return pl.DataFrame(out)


def random_frame(n_weeks: int = 30, players: int = 12, per_week: int = 5, seed: int = 1) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for w in range(n_weeks):
        for j in range(per_week):
            a, b = rng.sample(range(1, players + 1), 2)
            rows.append((f"m{w:03d}-{j}", w, a, b, rng.random() < 0.5, rng.choice(["Hard", "Clay", "Grass", None])))
    return frame(rows)


def synthetic_season_rows(years=(2017, 2018, 2019, 2020), seed: int = 7) -> dict[int, list[dict]]:
    """Weekly one-round events among 8 players with persistent skill differences."""
    rng = random.Random(seed)
    players = [100, 200, 300, 400, 500, 600, 700, 800]
    skill = {p: i * 60.0 for i, p in enumerate(players)}
    surfaces = ["Hard", "Clay", "Grass", "Hard"]
    by_year = {}
    for y in years:
        rows = []
        d = date(y, 1, 6)
        while d.weekday() != 0:
            d += timedelta(days=1)
        for week in range(20):
            start = d + timedelta(weeks=week)
            order = players[:]
            rng.shuffle(order)
            surface = surfaces[week % 4]
            for j in range(4):
                p, q = order[2 * j], order[2 * j + 1]
                p_win = 1 / (1 + 10 ** ((skill[q] - skill[p]) / 400))
                w, l = (p, q) if rng.random() < p_win else (q, p)
                rows.append(
                    match_row(j + 1, w, l, stats=False, tourney_id=f"{y}-{week:04d}", tourney_name=f"Event {week}",
                              tourney_date=start.strftime("%Y%m%d"), surface=surface, round="R32")
                )  # fmt: skip
        by_year[y] = rows
    return by_year


def write_elo_config(tmp: Path, data_config: Path, **overrides) -> Path:
    text = f'''
data_config = "{data_config}"

[outputs]
elo_dir = "{tmp / "elo"}"
reports_dir = "{tmp / "reports_elo"}"

[model]
model_version = "elo-test"
initial_rating = 1500.0
surface_init = "constant"
carpet_policy = "separate"
shrinkage_matches = 20.0

[model.k]
mode = "constant"
constant = {overrides.get("k", 32.0)}

[chronology]
week_start = "saturday"
[[chronology.multi_week]]
levels = ["G"]
weeks = 2

[eligibility]
update_statuses = {overrides.get("update_statuses", ["completed"])!r}
target_statuses = ["completed"]
sensitivity_retirements = true

[[pools]]
name = "atp_tour"
families = ["tour"]
levels = ["G", "M", "A", "F", "D", "O"]

[evaluation]
validation_start = "2018-01-01"
test_start = "2020-01-01"
test_end = "2020-12-31"
calibration_bins = 5
bootstrap_samples = 50
bootstrap_seed = 1

[tuning]
k_constant = [16.0, 32.0]
k_decay = [{{ numerator = 250.0, offset = 5.0, shape = 0.4 }}]
surface_init = ["constant", "global"]
shrinkage_matches = [0.0, 20.0]
'''.replace("'", '"')
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "elo.toml"
    path.write_text(text)
    return path


def build_phase1(tmp: Path, archive, rows_by_year: dict[int, list[dict]]):
    """Run Phase 1 on synthetic seasons; return the data config path."""
    from tennis_data.config import load_config
    from tennis_data.ingestion.fetch import ingest
    from tennis_data.pipeline import build

    for y, rows in rows_by_year.items():
        archive.set_season(y, rows)
    cfg_path = write_config(tmp, archive.root, min(rows_by_year), max(rows_by_year))
    cfg = load_config(cfg_path)
    ingest(cfg)
    build(cfg)
    return cfg_path

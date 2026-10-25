"""Synthetic Phase 1-3B world plus Phase 4A config and fixture-provider files (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from tennis_data.modeling.config import load_modeling_config
from tennis_data.modeling.pipeline import run_modeling
from tennis_data.prospective.config import load_prospective_config

SNAPSHOT_END = "2020-05-18"  # last event start in elo_helpers.synthetic_season_rows for 2020
PROSPECTIVE_START = "2020-07-31"


def prospective_config(tmp: Path, modeling_config: Path, frozen_hash: str, **over) -> Path:
    default = (Path(__file__).resolve().parents[1] / "config" / "prospective.toml").read_text()
    body = default[default.index("[chronology]") :]
    body = body.replace('historical_snapshot_end = "2026-05-25"', f'historical_snapshot_end = "{SNAPSHOT_END}"')
    body = body.replace(
        'prospective_start = "2026-10-25"', f'prospective_start = "{over.get("start", PROSPECTIVE_START)}"'
    )
    body = body.replace('overrides = "config/identity_overrides.toml"', f'overrides = "{tmp / "overrides.toml"}"')
    body = body.replace("cases_per_category = 25", "cases_per_category = 6")
    if over.get("auto_approve"):
        body = body.replace("auto_approve_high_confidence = false", "auto_approve_high_confidence = true")
    text = (f'modeling_config = "{modeling_config}"\n\n[model]\nexpected_frozen_hash = "{frozen_hash}"\n\n'
            f'[outputs]\nprospective_dir = "{tmp / "prospective"}"\nreports_dir = "{tmp / "reports_prospective"}"\n\n'
            f"{body}")  # fmt: skip
    if not (tmp / "overrides.toml").exists():
        (tmp / "overrides.toml").write_text("")
    path = tmp / "prospective.toml"
    path.write_text(text)
    return path


def build_world(tmp: Path, archive):
    from modeling_helpers import build_upstream

    mpath = build_upstream(tmp, archive)
    mcfg = load_modeling_config(mpath)
    res = run_modeling(mcfg)
    ppath = prospective_config(tmp, mpath, res["frozen"]["frozen_hash"])
    return load_prospective_config(ppath), res


PLAYERS = [
    {"id": "fx-100", "first_name": "Alice", "last_name": "Archer", "atp_id": 100},
    {"id": "fx-200", "first_name": "Bob", "last_name": "Baker", "atp_id": 200},
    {"id": "fx-300", "first_name": "Carl", "last_name": "Cole", "atp_id": 300},
    {"id": "fx-400", "first_name": "Dan", "last_name": "Dorn", "atp_id": 400},
    {"id": "fx-500", "first_name": "Eve", "last_name": "Eng", "atp_id": 500},
    {"id": "fx-600", "first_name": "Finn", "last_name": "Fox", "atp_id": 600},
    {"id": "fx-700", "first_name": "Gus", "last_name": "Gray", "atp_id": 700},
    {"id": "fx-800", "first_name": "Hal", "last_name": "Hunt", "atp_id": 800},
    {"id": "fx-new", "first_name": "Zed", "last_name": "Zulu", "country": "NZL"},  # unseen player
    {"id": "fx-amb", "first_name": "B.", "last_name": "Baker"},  # abbreviated, no confirmation -> ambiguous
]

STATS_W = {"svpt": 60, "first_in": 38, "first_won": 30, "second_won": 12, "aces": 5, "double_faults": 2,
           "service_games": 10, "bp_saved": 1, "bp_faced": 1}  # fmt: skip
STATS_L = {"svpt": 65, "first_in": 40, "first_won": 25, "second_won": 10, "aces": 3, "double_faults": 4,
           "service_games": 10, "bp_saved": 3, "bp_faced": 5}  # fmt: skip


def result(mid, tid, rnd, p1, p2, winner, *, version=1, score="6-4 6-4", status="completed", stats=True, **kw):
    m = {"id": mid, "version": version, "tournament_id": tid, "round": rnd, "best_of": 3, "status": status,
         "player1_id": p1, "player2_id": p2, "winner_id": winner, "score": score, "minutes": 90,
         "stats_definition": "sackmann_v1", "published_at": "2020-06-10T12:00:00+00:00"}  # fmt: skip
    if stats:
        m["stats"] = {"player1": STATS_W if winner == p1 else STATS_L, "player2": STATS_L if winner == p1 else STATS_W}
    m.update(kw)
    return m


TOURNAMENTS = [
    {"id": "t-june1", "name": "Summer Open", "level": "A", "surface": "Grass", "draw_size": 32, "start_date": "2020-06-01"},
    {"id": "t-june2", "name": "Lakeside Cup", "level": "A", "surface": "Hard", "draw_size": 32, "start_date": "2020-06-08"},
    {"id": "t-slam", "name": "Test Slam", "level": "G", "surface": "Grass", "draw_size": 128, "start_date": "2020-06-22"},
    {"id": "t-next", "name": "Autumn Open", "level": "A", "surface": "Hard", "draw_size": 32, "start_date": "2020-08-03"},
]  # fmt: skip


def base_matches():
    return [
        result("m1", "t-june1", "R32", "fx-100", "fx-200", "fx-100"),
        result("m2", "t-june1", "R32", "fx-300", "fx-400", "fx-400"),
        result("m3", "t-june1", "QF", "fx-100", "fx-400", "fx-400", score="6-3 2-1 RET", status="retired"),
        result("m4", "t-june2", "R32", "fx-500", "fx-600", "fx-600", stats=False),  # outcome only
        result("m5", "t-june2", "R32", "fx-700", "fx-800", "fx-700", score="W/O", status="walkover", stats=False),
        result("m6", "t-june2", "R32", "fx-new", "fx-100", "fx-new"),  # new player
        result("m7", "t-june2", "R16", "fx-amb", "fx-300", "fx-300"),  # ambiguous identity -> held
        result("m8", "t-slam", "R128", "fx-100", "fx-500", "fx-500"),  # two-week event
        result("m9", "t-slam", "R64", "fx-500", "fx-600", "fx-500"),
        {
            **result("m10", "t-june2", "R16", "fx-200", "fx-600", "fx-200"),
            "stats": {"player1": {**STATS_W, "first_won": 70}, "player2": STATS_L},
        },  # invalid service counts
        {
            "id": "f1",
            "tournament_id": "t-next",
            "round": "R32",
            "best_of": 3,
            "status": "scheduled",
            "player1_id": "fx-100",
            "player2_id": "fx-200",
            "scheduled_start_at": "2020-08-04T10:00:00+00:00",
        },
    ]


def fixture_doc(matches=None, provider="fixture", generated_at="2020-07-01T00:00:00+00:00", players=None):
    return {"schema": "tennis-fixture-provider-1", "provider": provider, "generated_at": generated_at,
            "players": players or PLAYERS, "tournaments": TOURNAMENTS,
            "matches": base_matches() if matches is None else matches}  # fmt: skip


def write_doc(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return path

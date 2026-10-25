"""Synthetic archive fixtures: a tiny fake copy of the archive layout, no network needed."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tennis_data.config import load_config
from tennis_data.ingestion.fetch import ingest
from tennis_data.ingestion.sources import METADATA_FILES, PLAYERS_FILE, RANKINGS_FILES
from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS

FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# A self-consistent completed 6-4 6-4 match (winner perspective).
_W_STATS = dict(ace=5, df=2, svpt=60, **{"1stIn": 38, "1stWon": 30, "2ndWon": 12}, SvGms=10, bpSaved=1, bpFaced=1)
_L_STATS = dict(ace=3, df=4, svpt=65, **{"1stIn": 40, "1stWon": 25, "2ndWon": 10}, SvGms=10, bpSaved=3, bpFaced=5)

PLAYERS = {
    100: ("Alice", "Archer"), 200: ("Bob", "Baker"), 300: ("Carl", "Cole"), 400: ("Dan", "Dorn"),
    500: ("Eve", "Eng"), 600: ("Finn", "Fox"), 700: ("Gus", "Gray"), 800: ("Hal", "Hunt"),
    900: ("Ian", "Ives"), 199999: ("U", "Unknown"),
}  # fmt: skip


def match_row(match_num: int, winner: int, loser: int, *, stats: bool = True, **overrides) -> dict:
    """One valid source row; override any source column by name."""
    row = dict.fromkeys(SOURCE_MATCH_COLUMNS, "")
    row.update(
        tourney_id="2020-0001", tourney_name="Testopen", surface="Hard", draw_size="32", tourney_level="A",
        tourney_date="20200106", match_num=str(match_num), winner_id=str(winner), loser_id=str(loser),
        winner_name=" ".join(PLAYERS.get(winner, ("X", "Y"))), loser_name=" ".join(PLAYERS.get(loser, ("X", "Y"))),
        winner_hand="R", loser_hand="L", winner_ht="185", loser_ht="180", winner_ioc="ESP", loser_ioc="FRA",
        winner_age="25.1", loser_age="28.3", score="6-4 6-4", best_of="3", round="R32", minutes="90",
        winner_rank="10", winner_rank_points="3000", loser_rank="20", loser_rank_points="1500",
    )  # fmt: skip
    if stats:
        row.update({f"w_{k}": str(v) for k, v in _W_STATS.items()})
        row.update({f"l_{k}": str(v) for k, v in _L_STATS.items()})
    row.update({k: "" if v is None else str(v) for k, v in overrides.items()})
    return row


def write_csv(path: Path, rows: list[dict], columns=SOURCE_MATCH_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class FakeArchive:
    def __init__(self, root: Path):
        self.root = root
        for rel in METADATA_FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(f"fixture copy of {rel}\n")
        write_csv(
            root / PLAYERS_FILE,
            [{"player_id": i, "name_first": f, "name_last": l, "hand": "R"} for i, (f, l) in PLAYERS.items()],
            columns=["player_id", "name_first", "name_last", "hand", "dob", "ioc", "height", "wikidata_id"],
        )
        ranks = [{"ranking_date": "20200106", "rank": r, "player": p, "points": 1} for p, r in ((100, 10), (200, 21))]
        for rel in RANKINGS_FILES:
            write_csv(
                root / rel, ranks if rel.endswith("20s.csv") else [], ["ranking_date", "rank", "player", "points"]
            )

    def set_season(self, season: int, rows: list[dict], family_prefix: str = "atp_matches_") -> Path:
        path = self.root / "atp" / f"{family_prefix}{season}.csv"
        write_csv(path, rows)
        return path


def write_config(tmp_path: Path, archive_root: Path, start: int = 2020, end: int = 2020, **source) -> Path:
    src = {"mode": "local", "local_path": str(archive_root), "verify_local_commit": False, **source}
    src_toml = "\n".join(f"{k} = {v!r}".replace("'", '"').replace("False", "false").replace("True", "true")
                         for k, v in src.items())  # fmt: skip
    text = f"""
[source]
name = "fixture-archive"
repo = "example/fixture-archive"
commit = "{FAKE_COMMIT}"
license = "CC BY-NC-SA 4.0"
attribution = "synthetic test fixture"
{src_toml}

[selection]
families = ["tour"]
start_year = {start}
end_year = {end}

[paths]
raw_dir = "{tmp_path / "out/raw"}"
validated_dir = "{tmp_path / "out/validated"}"
quarantine_dir = "{tmp_path / "out/quarantine"}"
reports_dir = "{tmp_path / "out/reports"}"
duckdb_path = "{tmp_path / "out/tennis.duckdb"}"
"""
    default = (Path(__file__).resolve().parents[1] / "config" / "default.toml").read_text()
    text += default[default.index("[validation]") :]
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


@pytest.fixture
def archive(tmp_path) -> FakeArchive:
    return FakeArchive(tmp_path / "archive")


@pytest.fixture
def make_pipeline(tmp_path, archive):
    """Write the given 2020 rows, ingest them, and return the config."""

    def _make(rows: list[dict], subdir: str = "run"):
        archive.set_season(2020, rows)
        cfg = load_config(write_config(tmp_path / subdir, archive.root))
        ingest(cfg)
        return cfg

    return _make


@pytest.fixture
def validate_rows(make_pipeline):
    """rows -> (validated, quarantine) DataFrames, without writing outputs."""
    from tennis_data.pipeline import validate

    calls = iter(range(1_000))

    def _run(rows: list[dict]):
        # A fresh output tree per call: a raw snapshot is immutable once ingested.
        validated, quarantine, _ = validate(make_pipeline(rows, subdir=f"run{next(calls)}"))
        return validated, quarantine

    return _run

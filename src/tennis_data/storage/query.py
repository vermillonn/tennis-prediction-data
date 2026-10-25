"""DuckDB query interface over the Parquet outputs.

Views:
  matches          validated canonical match table
  quarantine       quarantined rows with reasons and the original raw record
  player_matches   long format: one row per (match, player) with opponent columns,
                   the shape needed later for rolling, leakage-free player features
  players          raw atp_players.csv (if ingested)
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from tennis_data.config import Config
from tennis_data.ingestion.sources import PLAYERS_FILE
from tennis_data.normalization.schema import PLAYER_FIELDS, STAT_FIELDS

_PER_PLAYER = ["id", *PLAYER_FIELDS, *STAT_FIELDS]


def _sql_path(p: Path) -> str:
    return str(p).replace("'", "''")


def _player_matches_sql() -> str:
    def side(me: str, opp: str) -> str:
        cols = [f"player_{me}_{c} AS {c}" for c in _PER_PLAYER]
        cols += [f"player_{opp}_{c} AS opponent_{c}" for c in _PER_PLAYER]
        won = "player_a_won" if me == "a" else "NOT player_a_won"
        return (
            f"SELECT match_id, tournament_id, tournament_date, season, surface, tournament_level, round, "
            f"round_order, best_of, match_status, stats_status, '{me.upper()}' AS side, {won} AS won, "
            + ", ".join(cols)
            + " FROM matches"
        )

    return f"{side('a', 'b')} UNION ALL {side('b', 'a')}"


def connect(config: Config, database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    validated = config.paths.validated_dir / "matches"
    if not validated.exists():
        raise FileNotFoundError(f"No validated data at {validated}. Run `tennis-data build` first.")
    con = duckdb.connect(str(database))
    glob = _sql_path(validated / "**" / "*.parquet")
    con.execute(f"CREATE OR REPLACE VIEW matches AS SELECT * FROM read_parquet('{glob}', hive_partitioning = false)")
    quarantine = config.paths.quarantine_dir / "matches.parquet"
    if quarantine.exists():
        con.execute(f"CREATE OR REPLACE VIEW quarantine AS SELECT * FROM read_parquet('{_sql_path(quarantine)}')")
    con.execute(f"CREATE OR REPLACE VIEW player_matches AS {_player_matches_sql()}")
    players = config.raw_source_dir / PLAYERS_FILE
    if players.exists():
        con.execute(
            f"CREATE OR REPLACE VIEW players AS SELECT * FROM read_csv('{_sql_path(players)}', all_varchar = true)"
        )
    return con


def build_database(config: Config) -> Path:
    """Materialise the views into a DuckDB file (views only; data stays in Parquet)."""
    path = config.paths.duckdb_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    connect(config, path).close()
    return path

"""Manual import of authorized Sackmann-format CSV exports (same columns as the archive's atp_matches files).

Player and tournament IDs in these files are archive IDs, so they are *declared* canonical IDs; identity checks
still compare names with the player master. The serve statistics follow the archive convention (sackmann_v1).
Only files the user is licensed to use may be imported; nothing here downloads anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.prospective.providers.raw_store import RawStore
from tennis_data.prospective.schema import MATCH_SCHEMA, PLAYER_SCHEMA, STATS, RecordError, frame

NAME = "sackmann_csv"


def _status(score: str | None) -> str:
    from tennis_data.normalization.score import match_status

    return match_status(score)


class SackmannCsvProvider:
    name = NAME
    verified = True  # the format is the archive's own

    def __init__(self, store: RawStore):
        self.store = store

    def ingest_file(self, path: Path, retrieved_at: datetime | None = None):
        content = Path(path).read_bytes()
        rec = self.store.put(NAME, "file", {"name": Path(path).name}, content, retrieved_at or datetime.now(UTC))
        df = pl.read_csv(content, infer_schema=False)
        missing = [c for c in SOURCE_MATCH_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing Sackmann columns {missing}")
        return self.parse(df, rec.sha256, datetime.fromisoformat(rec.retrieved_at))

    def parse(self, df: pl.DataFrame, sha: str, retrieved: datetime):
        errors: list[RecordError] = []
        players: dict[str, dict] = {}
        rows = []

        def num(v):
            try:
                return None if v in (None, "") else int(float(v))
            except ValueError:
                return None

        for r in df.iter_rows(named=True):
            mid = f"{r['tourney_id']}:{r['match_num']}"
            try:
                start = datetime.strptime(r["tourney_date"], "%Y%m%d").date()
            except (TypeError, ValueError):
                errors.append(RecordError(NAME, mid, "tourney_date", "unparseable", str(r["tourney_date"])))
                continue
            for side in ("winner", "loser"):
                pid = r[f"{side}_id"]
                if pid and pid not in players:
                    name = (r[f"{side}_name"] or "").strip()
                    first, _, last = name.partition(" ")
                    players[pid] = {"provider": NAME, "provider_player_id": pid, "first_name": first or None,
                                    "last_name": last or first or None, "country": r[f"{side}_ioc"], "birth_date": None,
                                    "hand": r[f"{side}_hand"], "declared_canonical_id": num(pid), "raw_sha256": sha,
                                    "retrieved_at": retrieved}  # fmt: skip
            row = {"provider": NAME, "provider_match_id": mid, "source_version": 1,
                   "provider_tournament_id": r["tourney_id"], "tournament_name": r["tourney_name"],
                   "tournament_level": r["tourney_level"], "surface": r["surface"], "draw_size": num(r["draw_size"]),
                   "event_start_date": start, "event_start_at": None, "round": r["round"], "best_of": num(r["best_of"]),
                   "event_completed_at": None, "source_published_at": None, "status": _status(r["score"]),
                   "score": r["score"], "winner_provider_id": r["winner_id"], "loser_provider_id": r["loser_id"],
                   "minutes": num(r["minutes"]), "stats_definition": "sackmann_v1", "raw_sha256": sha,
                   "retrieved_at": retrieved}  # fmt: skip
            for s in STATS:
                row[f"w_{s}"], row[f"l_{s}"] = num(r[f"w_{s}"]), num(r[f"l_{s}"])
            rows.append(row)
        return frame(list(players.values()), PLAYER_SCHEMA), frame(rows, MATCH_SCHEMA), None, errors

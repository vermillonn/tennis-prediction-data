"""Documented JSON exchange format (``tennis-fixture-provider-1``) used for CI fixtures and for authorized exports
that a provider or a human has converted to it. Format (all times ISO 8601 with offset, all optional unless noted):

    {"schema": "tennis-fixture-provider-1", "provider": "<name>", "generated_at": "...",
     "players": [{"id"*, "first_name", "last_name"*, "country", "birth_date", "hand", "atp_id"}],
     "tournaments": [{"id"*, "name", "level", "surface", "draw_size", "start_date"*}],
     "matches": [{"id"*, "version", "tournament_id"*, "round"*, "best_of", "status"*,
                  "scheduled_start_at", "started_at", "completed_at", "published_at",
                  "player1_id"*, "player2_id"*, "winner_id", "score", "minutes",
                  "stats_definition", "stats": {"player1": {...}, "player2": {...}}}]}

``status`` "scheduled"/"postponed"/"cancelled" makes a fixture; any result status makes a result. Stats keys:
svpt, first_in, first_won, second_won, aces, double_faults, service_games, bp_saved, bp_faced. ``atp_id`` is the
provider's own claim of a canonical (archive) player ID and is only a candidate until identity checks pass.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from tennis_data.prospective.providers.raw_store import RawStore
from tennis_data.prospective.schema import (
    FIXTURE_SCHEMA,
    FIXTURE_STATUSES,
    MATCH_SCHEMA,
    PLAYER_SCHEMA,
    RecordError,
    frame,
)

FORMAT = "tennis-fixture-provider-1"
STAT_KEYS = {"svpt": "svpt", "first_in": "1stIn", "first_won": "1stWon", "second_won": "2ndWon", "aces": "ace",
             "double_faults": "df", "service_games": "SvGms", "bp_saved": "bpSaved", "bp_faced": "bpFaced"}  # fmt: skip


def _ts(v):
    if v is None:
        return None
    t = datetime.fromisoformat(v)
    if t.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {v!r} (times are never assumed)")
    return t.astimezone(UTC)


def _d(v):
    return None if v is None else date.fromisoformat(v)


class FixtureProvider:
    """Reads files in the exchange format. ``verified`` is True only for the format, not for any real source."""

    verified = True

    def __init__(self, store: RawStore):
        self.store = store

    def ingest_file(self, path: Path, retrieved_at: datetime | None = None):
        content = Path(path).read_bytes()
        doc = json.loads(content)
        if doc.get("schema") != FORMAT:
            raise ValueError(f"{path}: expected schema {FORMAT!r}, got {doc.get('schema')!r}")
        provider = doc["provider"]
        retrieved = retrieved_at or _ts(doc.get("generated_at")) or datetime.now(UTC)
        rec = self.store.put(provider, "file", {"name": Path(path).name}, content, retrieved)
        return self.parse(doc, rec.sha256, datetime.fromisoformat(rec.retrieved_at))

    def parse(self, doc: dict, sha: str, retrieved: datetime):
        provider = doc["provider"]
        errors: list[RecordError] = []
        players = [
            {"provider": provider, "provider_player_id": str(p["id"]), "first_name": p.get("first_name"),
             "last_name": p.get("last_name"), "country": p.get("country"), "birth_date": _d(p.get("birth_date")),
             "hand": p.get("hand"), "declared_canonical_id": p.get("atp_id"), "raw_sha256": sha,
             "retrieved_at": retrieved}
            for p in doc.get("players", [])
        ]  # fmt: skip
        tours = {str(t["id"]): t for t in doc.get("tournaments", [])}
        results, fixtures = [], []
        for m in doc.get("matches", []):
            mid = str(m["id"])
            t = tours.get(str(m.get("tournament_id")))
            if t is None:
                errors.append(
                    RecordError(provider, mid, "tournament_id", "unknown_tournament", "tournament not listed")
                )
                continue
            try:
                common = {
                    "provider": provider, "provider_match_id": mid, "source_version": int(m.get("version", 1)),
                    "provider_tournament_id": str(t["id"]), "tournament_name": t.get("name"),
                    "tournament_level": t.get("level"), "surface": t.get("surface"), "draw_size": t.get("draw_size"),
                    "event_start_date": _d(t.get("start_date")), "event_start_at": _ts(m.get("started_at")),
                    "round": m.get("round"), "best_of": m.get("best_of"),
                }  # fmt: skip
                status = m.get("status")
                if status in FIXTURE_STATUSES:
                    fixtures.append({**common, "event_start_at": _ts(m.get("scheduled_start_at")), "status": status,
                                     "player1_provider_id": str(m["player1_id"]),
                                     "player2_provider_id": str(m["player2_id"]), "raw_sha256": sha,
                                     "retrieved_at": retrieved})  # fmt: skip
                    continue
                p1, p2, w = str(m["player1_id"]), str(m["player2_id"]), m.get("winner_id")
                w = None if w is None else str(w)
                if w not in (p1, p2):
                    errors.append(RecordError(provider, mid, "winner_id", "winner_not_a_player", "winner not in match"))
                    continue
                loser = p2 if w == p1 else p1
                st = m.get("stats") or {}
                ws, ls = (
                    st.get("player1" if w == p1 else "player2") or {},
                    st.get("player2" if w == p1 else "player1") or {},
                )
                row = {**common, "event_completed_at": _ts(m.get("completed_at")),
                       "source_published_at": _ts(m.get("published_at")), "status": status, "score": m.get("score"),
                       "winner_provider_id": w, "loser_provider_id": loser, "minutes": m.get("minutes"),
                       "stats_definition": m.get("stats_definition"), "raw_sha256": sha, "retrieved_at": retrieved}  # fmt: skip
                for k, s in STAT_KEYS.items():
                    row[f"w_{s}"], row[f"l_{s}"] = ws.get(k), ls.get(k)
                results.append(row)
            except (KeyError, ValueError, TypeError) as e:
                errors.append(RecordError(provider, mid, "record", "malformed", str(e)))
        return frame(players, PLAYER_SCHEMA), frame(results, MATCH_SCHEMA), frame(fixtures, FIXTURE_SCHEMA), errors

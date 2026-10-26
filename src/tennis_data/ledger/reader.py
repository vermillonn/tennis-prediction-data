"""Read-only queries: events, predictions and derived lifecycle status (never modifies the ledger)."""

from __future__ import annotations

import json

from tennis_data.ledger import storage
from tennis_data.ledger.lifecycle import MatchState, apply, timing


class LedgerReader:
    def __init__(self, path, busy_timeout_ms: int = 30000):
        self.con = storage.connect(path, busy_timeout_ms, read_only=True)

    def close(self):
        self.con.close()

    def meta(self) -> dict:
        return storage.meta(self.con)

    def head(self) -> tuple[int, str]:
        return storage.head(self.con)

    def event(self, event_id: str) -> dict | None:
        r = self.con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return _row(r) if r else None

    def events_for_match(self, match_id: str) -> list[dict]:
        return [_row(r) for r in self.con.execute("SELECT * FROM events WHERE match_id = ? ORDER BY seq", (match_id,))]

    def match_state(self, match_id: str) -> MatchState:
        evs = self.events_for_match(match_id)
        by_id = {e["event_id"]: e for e in evs}
        st = MatchState(match_id)
        for e in evs:
            apply(st, e, by_id)
        return st

    def predictions_for_match(self, match_id: str) -> list[dict]:
        """Every prediction of the match (voided ones included), with derived status and timing."""
        st = self.match_state(match_id)
        out = []
        for e in self.events_for_match(match_id):
            if e["event_type"] != "PREDICTION_COMMITTED":
                continue
            p = st.predictions[e["event_id"]]
            out.append({"event": e, "voided": p["voided"], "timing": timing(st, e["event_id"]),
                        "match_status": st.status, "current_result": st.result})  # fmt: skip
        return out

    def conflicts(self) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM idempotency_conflicts ORDER BY id")]


def _row(r) -> dict:
    d = dict(r)
    d["payload"] = json.loads(d["payload"])
    d["refs"] = json.loads(d["refs"])
    return d

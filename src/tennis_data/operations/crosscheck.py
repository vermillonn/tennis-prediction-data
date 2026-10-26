"""Independent corroboration of a TML-format export with other result sources (outcomes and coverage only).

The other sources are never imported. They use different player IDs and name forms (``Munar J.`` or full names) and
record each match's own date and time, whereas the archive and the frozen batching use event start dates. Matching
is therefore by names and match dates:

- a TML record and a source record correspond when both players' name tokens agree (abbreviated first names reduce
  to surname tokens) and the dates differ by at most one day (time zones); only a unique candidate counts;
- the populations differ: the sources include qualifying, Challenger, WTA or ITF records, so only ATP main-tour rows
  (by each source's own category field) are compared, and unmatched source rows are reported by tournament for
  review, not counted as TML gaps.

Per-event completeness of the TML export itself is checked against its draw size (a knockout draw of size *n*,
byes included, has *n* − 1 matches; round-robin and team events are reported, not judged).
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta

from tennis_data.operations.tml import PLAYER_FILE, decode
from tennis_data.prospective.identity import norm

VBT_MAIN = ("Main tour", "Masters", "Grand Chelem", "Coupe Davis / BJK Cup")


def _tokens(name: str) -> frozenset:
    return frozenset(t for t in norm(name).split() if len(t) > 1)


def tml_records(files: dict[str, bytes]) -> list[dict]:
    people = {r["id"]: r for r in csv.DictReader(io.StringIO(decode(files[PLAYER_FILE])[0]))}
    out = []
    for name in sorted(n for n in files if n != PLAYER_FILE):
        for r in csv.DictReader(io.StringIO(decode(files[name])[0])):
            if not r["winner_id"] or not r["loser_id"]:
                continue
            w, lo = people.get(r["winner_id"], {}), people.get(r["loser_id"], {})
            out.append({"date": datetime.strptime(r["tourney_date"], "%Y%m%d").date(), "tid": r["tourney_id"],
                        "tournament": r["tourney_name"], "level": r["tourney_level"], "round": r["round"],
                        "draw_size": int(r["draw_size"]) if r["draw_size"] else None,
                        "w": _tokens(w.get("player") or r["winner_name"]), "l": _tokens(lo.get("player") or r["loser_name"]),
                        "score": r["score"]})  # fmt: skip
    return out


def season_records(data: bytes) -> list[dict]:
    out = []
    for r in csv.DictReader(io.StringIO(decode(data)[0])):
        if r["tour_type_human"] != "ATP Tour" or r["status"] != "FINISHED" or r["winner_code"] not in ("1", "2"):
            continue
        h, a = _tokens(r["home_name"]), _tokens(r["away_name"])
        w, lo = (h, a) if r["winner_code"] == "1" else (a, h)
        out.append({"date": datetime.fromtimestamp(int(r["date_timestamp"]), UTC).date(), "tournament": r["tournament"],
                    "w": w, "l": lo, "status": r["status_extra"]})  # fmt: skip
    return out


def vbt_records(data: bytes) -> list[dict]:
    out = []
    for r in csv.DictReader(io.StringIO(decode(data)[0].lstrip("﻿")), delimiter=";"):
        if r["genre"] != "atp" or r["categorie"] not in VBT_MAIN or not r["vainqueur_id"]:
            continue
        p1, p2 = _tokens(r["joueur1"]), _tokens(r["joueur2"])
        if r["vainqueur_id"] == r["joueur1_id"]:
            w, lo = p1, p2
        elif r["vainqueur_id"] == r["joueur2_id"]:
            w, lo = p2, p1
        else:
            continue
        out.append({"date": datetime.strptime(r["date"][:10], "%Y-%m-%d").date(), "tournament": r["tournoi"],
                    "w": w, "l": lo, "status": r["score"]})  # fmt: skip
    return out


def _same(t: frozenset, s: frozenset) -> bool:
    return bool(s) and bool(t) and (s <= t or t <= s)


def corroborate(tml: list[dict], other: list[dict], since: date) -> dict:
    idx = defaultdict(list)
    for i, o in enumerate(other):
        for tok in o["w"] | o["l"]:
            idx[tok].append(i)
    acct = Counter()
    used: set = set()
    disagreements = []
    for t in (x for x in tml if x["date"] >= since):
        acct["tml_records"] += 1
        cands = {i for tok in t["w"] | t["l"] for i in idx.get(tok, ())}
        hits = []
        for i in cands:
            o = other[i]
            if abs((o["date"] - t["date"]).days) > 1:
                continue
            if _same(t["w"], o["w"]) and _same(t["l"], o["l"]):
                hits.append((i, True))
            elif _same(t["w"], o["l"]) and _same(t["l"], o["w"]):
                hits.append((i, False))
        if not hits:
            acct["no_counterpart"] += 1
            continue
        if len(hits) > 1:
            acct["several_counterparts"] += 1
            continue
        i, agree = hits[0]
        used.add(i)
        acct["matched"] += 1
        acct["winner_agrees" if agree else "winner_differs"] += 1
        if not agree:
            disagreements.append({"date": str(t["date"]), "tournament": t["tournament"], "round": t["round"],
                                  "tml_score": t["score"], "other_status": other[i]["status"]})  # fmt: skip
    unmatched_other = Counter(o["tournament"] for i, o in enumerate(other) if o["date"] >= since and i not in used)
    return {"accounting": dict(acct), "winner_disagreements": disagreements,
            "source_rows_since": sum(1 for o in other if o["date"] >= since),
            "unmatched_source_rows_by_tournament": dict(unmatched_other.most_common())}  # fmt: skip


def completeness(tml: list[dict], since: date) -> list[dict]:
    ev = defaultdict(list)
    for t in tml:
        ev[(t["tid"], t["tournament"], t["level"])].append(t)
    out = []
    for (tid, name, level), rows in ev.items():
        start = min(r["date"] for r in rows)
        if start < since - timedelta(days=0):
            continue
        n, draw = len(rows), rows[0]["draw_size"]
        rounds = Counter(r["round"] for r in rows)
        knockout = level not in ("D",) and "RR" not in rounds and draw and draw >= 4
        exp = draw - 1 if knockout else None
        out.append({"tournament": name, "tid": tid, "level": level, "start": str(start),
                    "last_match_date": str(max(r["date"] for r in rows)), "matches": n, "draw_size": draw,
                    "expected_matches": exp, "has_final": "F" in rounds,
                    "status": ("complete" if exp == n and "F" in rounds else "INCOMPLETE" if exp else "not judged")})  # fmt: skip
    return sorted(out, key=lambda e: (e["start"], e["tournament"]))

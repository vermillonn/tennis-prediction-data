"""TML-Database (Sackmann-format CSVs with ATP player codes) → the Phase 4A exchange format.

TML match files use the Sackmann column layout but **ATP alphanumeric player codes** (e.g. ``X0Y1``) instead of archive
player IDs, and ``250``/``500`` instead of ``A`` as the tournament level of ATP 250/500 events. The Phase 4A
``sackmann_csv`` adapter would treat the codes as archive IDs, so TML is converted into the documented
``tennis-fixture-provider-1`` format instead and then imported with the unchanged Phase 4A path (raw store →
identity → ``rec-1`` reconciliation → Phase 1 validation → state snapshot):

- players come from TML's ``ATP_Database.csv`` (full name, birth date, nationality, hand), keyed by ATP code, with
  no claimed archive ID: the Phase 4A identity resolver decides the mapping (names, birth dates);
- levels are mapped with ``[tml].level_map``; an unmapped level is passed through and Phase 4A quarantines it;
- the result status comes from the Phase 1 score parser, as in the Sackmann adapter;
- serve statistics are labelled ``tml_unverified``: Phase 4A only feeds ``sackmann_v1`` statistics to the feature
  engine, so TML statistics stay out of serve/return estimation unless the compatibility study verifies them and a
  reviewed change adds the definition;
- the player file's encoding is detected (UTF-8, else CP-1252) and recorded;
- some TML events have no ``match_num``: the match id is then derived from (tournament, round, sorted player pair);
  a record with a missing player id is rejected and logged, never guessed; a repeated id is logged and the first
  occurrence kept (Phase 4A collapses byte-identical copies anyway).

Downloads are pinned to a TML commit and stored with their SHA-256; nothing is downloaded in tests. TML's README
permits research/educational use and forbids redistribution of the raw data, so raw and converted files stay in the
git-ignored workspace.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

from tennis_data.ledger.hashing import ts
from tennis_data.normalization.score import match_status
from tennis_data.prospective.providers.fixture import FORMAT as FIXTURE_FORMAT

PROVIDER = "tml_github"
STATS_DEFINITION = "tml_unverified"
STAT_COLS = {"svpt": "svpt", "first_in": "1stIn", "first_won": "1stWon", "second_won": "2ndWon", "aces": "ace",
             "double_faults": "df", "service_games": "SvGms", "bp_saved": "bpSaved", "bp_faced": "bpFaced"}  # fmt: skip
PLAYER_FILE = "ATP_Database.csv"


def decode(data: bytes) -> tuple[str, str]:
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252"), "cp1252"


def download(cfg, transport, now: datetime | None = None) -> dict:
    """Fetch the configured files at the pinned commit into ``<workspace>/tml/<commit>/`` (idempotent)."""
    t = cfg.tml
    out = cfg.workspace_dir / "tml" / t["commit"]
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"repository": t["repository"],
                                                                             "commit": t["commit"], "files": {}}  # fmt: skip
    for name in t["files"]:
        if name in manifest["files"] and (out / name).exists():
            continue
        url = f"{t['base_url']}/{t['repository']}/{t['commit']}/{name}"
        status, body = transport(url, {}, 60.0)
        if status != 200:
            raise RuntimeError(f"{url}: HTTP {status}")
        (out / name).write_bytes(body)
        manifest["files"][name] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "url": url,
                                   "retrieved_at": ts(now or datetime.now(UTC))}  # fmt: skip
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _int(v):
    try:
        return None if v in (None, "") else int(float(v))
    except ValueError:
        return None


def _date(v: str) -> str | None:
    try:
        return datetime.strptime(v, "%Y%m%d").date().isoformat()
    except (TypeError, ValueError):
        return None


def convert(files: dict[str, bytes], level_map: dict, generated_at: str, provider: str = PROVIDER,
            min_event_start: str | None = None) -> tuple[dict, dict]:  # fmt: skip
    """TML files (name → bytes; one player file plus match files) → exchange-format document and a conversion log.

    Deterministic: the same bytes and ``generated_at`` give the same document.

    Event start date: Sackmann's ``tourney_date`` is the event start, the only date the frozen Phase 2 batching uses.
    Some TML exports put each match's own date in that column instead (several dates per ``tourney_id``). The event
    start is then the earliest date of the event's matches; on the January–May 2026 overlap this falls in the same
    weekly batch as the archive's date for 28 of 29 non–Davis Cup events (United Cup starts on a Friday). Match
    dates are not used otherwise (they are not timestamps), and the log lists every event with per-match dates.

    ``min_event_start`` (ISO date) keeps only events starting on or after it. A backfill uses the events after the
    pinned archive's last event: records inside the archive period are validation data, never imported, because
    Phase 4A's duplicate detection matches tournaments by name and misses naming variants ("Monte Carlo" vs
    "Monte Carlo Masters", Davis Cup ties with the teams in the other order).
    """
    ptxt, penc = decode(files[PLAYER_FILE])
    people = {r["id"]: r for r in csv.DictReader(io.StringIO(ptxt))}
    log = {"player_file_encoding": penc, "files": {}, "unmapped_levels": {}, "duplicate_match_ids": [],
           "missing_match_num": 0, "rejected_missing_player_id": [], "players_missing_from_player_file": [],
           "unparseable_dates": 0}  # fmt: skip
    tours: dict = {}
    matches: dict = {}
    used: set = set()
    log["events_with_per_match_dates"] = []
    for name in sorted(n for n in files if n != PLAYER_FILE):
        txt, enc = decode(files[name])
        rows = list(csv.DictReader(io.StringIO(txt)))
        log["files"][name] = {"encoding": enc, "rows": len(rows)}
        starts: dict = {}
        for r in rows:
            d = _date(r["tourney_date"])
            if d is not None:
                starts.setdefault(r["tourney_id"], set()).add(d)
        multi = sorted(t for t, ds in starts.items() if len(ds) > 1)
        log["events_with_per_match_dates"] += [f"{name}:{t}" for t in multi]
        for r in rows:
            start = _date(r["tourney_date"])
            if start is None:
                log["unparseable_dates"] += 1
                continue
            raw_level = r["tourney_level"]
            level = level_map.get(raw_level, raw_level)
            if raw_level not in level_map:
                log["unmapped_levels"][raw_level] = log["unmapped_levels"].get(raw_level, 0) + 1
            tid = r["tourney_id"]
            start = min(starts[tid])  # event start (equals tourney_date when the file uses one date per event)
            if min_event_start is not None and start < min_event_start:
                log["excluded_before_min_event_start"] = log.get("excluded_before_min_event_start", 0) + 1
                continue
            tours.setdefault(tid, {"id": tid, "name": r["tourney_name"], "level": level, "surface": r["surface"] or None,
                                   "draw_size": _int(r["draw_size"]), "start_date": start})  # fmt: skip
            w, lo = r["winner_id"], r["loser_id"]
            if not w or not lo:  # never guess a missing participant
                log["rejected_missing_player_id"].append(
                    f"{name}:{tid}:{r['round']}:{r['winner_name']}|{r['loser_name']}"
                )
                continue
            if r["match_num"]:
                mid = f"{tid}:{r['match_num']}"
            else:  # some TML events carry no match numbers: derive a stable singles key
                log["missing_match_num"] += 1
                mid = f"{tid}:{r['round']}:{'-'.join(sorted((w, lo)))}"
            if mid in matches:
                log["duplicate_match_ids"].append(mid)
                continue
            used |= {w, lo}
            stats = {"player1": {k: _int(r[f"w_{c}"]) for k, c in STAT_COLS.items()},
                     "player2": {k: _int(r[f"l_{c}"]) for k, c in STAT_COLS.items()}}  # fmt: skip
            matches[mid] = {"id": mid, "version": 1, "tournament_id": tid, "round": r["round"],
                            "best_of": _int(r["best_of"]), "status": match_status(r["score"] or None),
                            "player1_id": w, "player2_id": lo, "winner_id": w, "score": r["score"] or None,
                            "minutes": _int(r["minutes"]), "stats_definition": STATS_DEFINITION, "stats": stats}  # fmt: skip
    players = []
    for pid in sorted(used):
        p = people.get(pid)
        if p is None:
            log["players_missing_from_player_file"].append(pid)
            players.append({"id": pid, "first_name": None, "last_name": None})
            continue
        first, _, last = (p.get("player") or "").strip().partition(" ")
        players.append({"id": pid, "first_name": first or None, "last_name": last or first or None,
                        "country": p.get("ioc") or None, "birth_date": _date(p.get("birthdate") or ""),
                        "hand": p.get("hand") or None})  # fmt: skip
    doc = {"schema": FIXTURE_FORMAT, "provider": provider, "generated_at": generated_at, "players": players,
           "tournaments": [tours[k] for k in sorted(tours)], "matches": [matches[k] for k in sorted(matches)]}  # fmt: skip
    log["matches"], log["players"], log["tournaments"] = len(matches), len(players), len(tours)
    return doc, log


def convert_workspace(cfg) -> tuple[Path, dict]:
    """Convert the pinned download into ``converted.json`` (generated_at = latest retrieval time of the inputs)."""
    d = Path(cfg.tml["source_dir"]) if cfg.tml.get("source_dir") else cfg.workspace_dir / "tml" / cfg.tml["commit"]
    manifest = json.loads((d / "manifest.json").read_text())
    files = {n: (d / n).read_bytes() for n in cfg.tml["files"]}
    for n, b in files.items():
        if hashlib.sha256(b).hexdigest() != manifest["files"][n]["sha256"]:
            raise RuntimeError(f"{n} differs from its recorded download hash")
    gen = max(manifest["files"][n]["retrieved_at"] for n in files)
    doc, log = convert(files, cfg.tml["level_map"], gen, cfg.tml.get("provider", PROVIDER),
                       cfg.tml.get("min_event_start"))  # fmt: skip
    out = cfg.workspace_dir / "converted.json" if cfg.tml.get("source_dir") else d / "converted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    log["converted_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    log["source_commit"] = cfg.tml["commit"]
    (out.parent / "conversion_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    return out, log

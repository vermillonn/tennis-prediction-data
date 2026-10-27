"""Evidence for unresolved identity mappings and an isolated simulation of resolving them (read-only for state).

- ``overlap_assignment``: for each provider player, the archive ID the archive itself uses in the same matches of
  the overlap period. A TML record is paired with an archive match through the *opponent's* mapped ID, the round
  and the event week (within ``tol`` days); the provider player's side of that archive match is the archive's
  assignment. The disputed mapping is never used for the pairing.
- ``candidate_profile``: the archive record of a candidate (full name, birth date, hand, height, country, Wikidata
  ID) and its archive match history.
- ``simulate``: restores the version 0 baseline into a new isolated directory, adds *proposed* mappings to the
  version 1 approvals in a sandbox-only overrides file, imports the matches whose players are all approved or
  proposed, and runs the unchanged Phase 4A backfill there. Nothing is approved, version 1 and the active
  ``data/prospective/`` are not touched, and the result is labelled SIMULATION.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, date, datetime

import polars as pl

from tennis_data.operations import controlled_backfill as cb
from tennis_data.operations import exports


def overlap_assignment(matches: list[dict], starts: dict[str, date], canon: dict[str, int], hist: pl.DataFrame,
                       players: set[str], tol: int = 7) -> dict[str, dict]:  # fmt: skip
    idx = defaultdict(list)
    for r in hist.iter_rows(named=True):
        for me, other in ((r["player_a_id"], r["player_b_id"]), (r["player_b_id"], r["player_a_id"])):
            idx[(me, r["round"])].append((r["tournament_date"], other, r["winner_id"] == other))
    out = {p: Counter() for p in players}
    for m in matches:
        for p, q in ((m["player1_id"], m["player2_id"]), (m["player2_id"], m["player1_id"])):
            if p not in players:
                continue
            cq = canon.get(q)
            if cq is None:
                out[p]["opponent_unmapped"] += 1
                continue
            hits = [(o, w) for d, o, w in idx.get((cq, m["round"]), ())
                    if abs((d - starts[m["tournament_id"]]).days) <= tol]  # fmt: skip
            if len(hits) != 1:
                out[p]["no_archive_match" if not hits else "several_archive_matches"] += 1
                continue
            o, o_won = hits[0]
            out[p][str(o) if o_won == (m["winner_id"] == p) else f"{o}:winner_differs"] += 1
    return {p: dict(c) for p, c in out.items()}


def candidate_profile(player_file, hist: pl.DataFrame, ids: set[int]) -> dict[int, dict]:
    ppl = {int(r["player_id"]): r for r in csv.DictReader(open(player_file)) if int(r["player_id"]) in ids}
    both = pl.concat([hist.select(pl.col("player_a_id").alias("pid"), "tournament_date"),
                      hist.select(pl.col("player_b_id").alias("pid"), "tournament_date")])  # fmt: skip
    g = {r["pid"]: r for r in both.filter(pl.col("pid").is_in(list(ids))).group_by("pid").agg(
        pl.len().alias("n"), pl.col("tournament_date").min().alias("first"),
        pl.col("tournament_date").max().alias("last")).iter_rows(named=True)}  # fmt: skip
    return {i: {"name": f"{r['name_first']} {r['name_last']}".strip(), "birth_date": r["dob"] or None,
                "hand": r["hand"], "height": r["height"] or None, "country": r["ioc"], "wikidata": r["wikidata_id"] or None,
                "archive_matches": g.get(i, {}).get("n", 0), "first": str(g[i]["first"]) if i in g else None,
                "last": str(g[i]["last"]) if i in g else None} for i, r in ppl.items()}  # fmt: skip


def simulate(cfg, label: str, name: str, additions: dict[str, int]) -> dict:
    """Isolated SIMULATION: version 1 approvals plus ``additions`` (provider id -> canonical id)."""
    from tennis_data.prospective.backfill import ingest_file, run_backfill

    run_dir = cfg.workspace_dir / "controlled_backfill" / label
    rec = json.loads((run_dir / "run_record.json").read_text())
    store = cfg.workspace_dir / "snapshots"
    base = json.loads((store / "manifests" / f"{rec['steps']['baseline']['name']}.json").read_text())
    sim = cfg.workspace_dir / "identity_resolution" / label / name
    if sim.exists():
        shutil.rmtree(sim)
    cb.restore(store, base, sim / "prospective")
    approved = {e["provider_player_id"] for e in rec["identity"]["approved"]} | set(additions)
    gap = exports.export_config(cfg, label, gap_only=True)
    conv = json.loads((gap.workspace_dir / "converted.json").read_text())
    doc, held = cb.filter_import(conv, approved)
    (sim / "import.json").write_text(json.dumps(doc, sort_keys=True, default=str) + "\n")
    v1 = (exports.private_dir(cfg, label) / "approvals_v1.toml").read_text()
    extra = ["# SIMULATION ONLY: proposed, NOT approved"] + [
        f'[[mapping]]\nprovider = "tml_export"\nprovider_player_id = "{p}"\ncanonical_player_id = {c}\n'
        f'reviewer = "PROPOSED (simulation)"\n' for p, c in sorted(additions.items())]  # fmt: skip
    (sim / "overrides.toml").write_text(v1 + "\n" + "\n".join(extra))
    ws = replace(cfg.prospective, prospective_dir=sim / "prospective", reports_dir=sim / "reports",
                 overrides_path=sim / "overrides.toml")  # fmt: skip
    ingest_file(
        ws, sim / "import.json", "tennis-fixture-provider-1", retrieved_at=datetime.fromisoformat(rec["started_at"])
    )
    b = run_backfill(ws)
    led = b["ledger"]
    pids = {m["id"]: (m["player1_id"], m["player2_id"]) for m in conv["matches"]}
    per = defaultdict(Counter)
    for r in led.iter_rows(named=True):
        for p in pids[r["provider_match_id"]]:
            if p in additions:
                per[p]["matches"] += 1
                per[p]["elo_updates"] += int(r["is_update"])
    return {"label": "SIMULATION (proposed mappings, nothing approved)", "name": name, "additions": additions,
            "accepted": led.height, "elo_updates": int(led["is_update"].sum()), "held": len(held),
            "state_snapshot_id": b["summary"]["state_snapshot_id"],
            "quarantine": b["summary"]["reconciliation_quarantine"], "new_identities": [
                r for r in b["summary"]["identity"] if r["category"] != "verified"],
            "per_player": {p: dict(c) for p, c in per.items()}, "run_at": datetime.now(UTC).isoformat()}  # fmt: skip

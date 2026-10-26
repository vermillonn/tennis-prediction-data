"""Dry run of an owner-supplied backfill: identity proposals for review, and a sandbox simulation of the result.

Nothing here approves an identity or touches Phase 4A's own outputs:

- ``identity_proposal`` lists the high-confidence mappings (unique exact name + birth date) with their evidence and
  whether matches on the archive overlap corroborate them (same pair, round, event week and winner). It writes a
  *proposed* overrides file whose entries carry ``reviewer = "PENDING"``; the owner must review it and copy approved
  entries into ``config/identity_overrides.toml``. Ambiguous and new players are listed for review, never proposed.
- ``simulate`` copies the export's Phase 4A workspace into a sandbox, points the identity overrides at the proposal,
  and runs the unchanged Phase 4A backfill there. Its counts and state hash describe what an approved backfill
  would produce; they are labelled SIMULATION and change no state.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import timedelta

import polars as pl

from tennis_data.operations.compatibility import match_pairs
from tennis_data.prospective.backfill import load_archive, parse_store, run_backfill
from tennis_data.prospective.identity import load_archive_players


def identity_proposal(ecfg) -> dict:
    ws = ecfg.workspace_prospective()
    dec = pl.read_parquet(ws.prospective_dir / "backfill" / "identity_decisions.parquet")
    parsed = parse_store(ws)
    hist, _ = load_archive(ws)
    m = match_pairs(parsed["matches"], dec, hist.filter(pl.col("season").is_in(list(ecfg.overlap_seasons))),
                    ecfg.date_tolerance_days)  # fmt: skip
    confirmed: dict = {}
    for c, h, w, _lo in m["pairs"]:
        ok = h["winner_id"] == w
        for pid in (c["winner_provider_id"], c["loser_provider_id"]):
            a, b = confirmed.get(pid, (0, 0))
            confirmed[pid] = (a + ok, b + (not ok))
    arch = {r["player_id"]: r for r in load_archive_players(ws).iter_rows(named=True)}
    players = {r["provider_player_id"]: r for r in parsed["players"].iter_rows(named=True)}
    rows = []
    for d in dec.filter(pl.col("category") == "high_confidence").sort("provider_player_id").iter_rows(named=True):
        p, a = players[d["provider_player_id"]], arch[d["canonical_player_id"]]
        agree, disagree = confirmed.get(d["provider_player_id"], (0, 0))
        rows.append({"provider": d["provider"], "provider_player_id": d["provider_player_id"],
                     "canonical_player_id": d["canonical_player_id"], "method": d["method"],
                     "provider_name": f"{p['first_name'] or ''} {p['last_name'] or ''}".strip(),
                     "archive_name": f"{a['first_name'] or ''} {a['last_name'] or ''}".strip(),
                     "birth_date": str(p["birth_date"]) if p["birth_date"] else None,
                     "overlap_matches_confirming": agree, "overlap_matches_contradicting": disagree,
                     "evidence_class": ("name+birth date, corroborated on the archive overlap" if agree and not disagree
                                        else "CONTRADICTED on the overlap" if disagree
                                        else "name+birth date only (no overlap matches)")})  # fmt: skip
    return {"proposed": rows, "corroborated": sum(1 for r in rows if r["overlap_matches_confirming"] and not
                                                  r["overlap_matches_contradicting"]),
            "contradicted": sum(1 for r in rows if r["overlap_matches_contradicting"]),
            "uncorroborated": sum(1 for r in rows if not r["overlap_matches_confirming"]
                                  and not r["overlap_matches_contradicting"])}  # fmt: skip


def write_proposal(path, proposal: dict, reviewer: str = "PENDING") -> None:
    lines = ["# PROPOSED identity mappings (Phase 5A dry run). NOT APPLIED. A reviewer must check each entry and copy",
             "# approved ones into config/identity_overrides.toml with their own name as reviewer.", ""]  # fmt: skip
    for r in proposal["proposed"]:
        if r["overlap_matches_contradicting"]:
            continue  # contradicted mappings are never proposed
        lines += ["[[mapping]]", f'provider = "{r["provider"]}"', f'provider_player_id = "{r["provider_player_id"]}"',
                  f"canonical_player_id = {r['canonical_player_id']}", f'reviewer = "{reviewer}"',
                  f'note = "{r["provider_name"]} = {r["archive_name"]} (born {r["birth_date"]}); {r["evidence_class"]}; '
                  f'{r["overlap_matches_confirming"]} overlap matches confirm"', ""]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def simulate(ecfg, proposal_path, sandbox) -> dict:
    """Run the unchanged Phase 4A backfill in a sandbox copy with the proposed mappings treated as approved."""
    src = ecfg.workspace_prospective().prospective_dir
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(src / "raw", sandbox / "raw")
    ws = replace(ecfg.workspace_prospective(), prospective_dir=sandbox, reports_dir=sandbox / "reports",
                 overrides_path=proposal_path)  # fmt: skip
    b = run_backfill(ws)
    b2 = run_backfill(ws, write=False)
    led = b["ledger"]
    gap = led.filter(pl.col("tournament_date") > ecfg.archive_last_week) if led is not None else None
    q = b["reconciliation"]["quarantine"]
    val = b["validated"]
    gap_val = val.filter(pl.col("tournament_date") > ecfg.archive_last_week) if val.height else val
    last_release = gap["release_batch"].max() if gap is not None and gap.height else None
    return {"label": "SIMULATION assuming the proposed identity mappings are approved",
            "summary": b["summary"], "deterministic_rerun": b["summary"] == b2["summary"],
            "duplicates_of_archive": int(q.filter(pl.col("reason") == "duplicate_of_historical").height)
            if q.height else 0,
            "quarantine_by_reason": dict(q.group_by("reason").len().iter_rows()) if q.height else {},
            "notes_by_reason": dict(b["reconciliation"]["notes"].group_by("reason").len().iter_rows())
            if b["reconciliation"]["notes"].height else {},
            "gap_validated": gap_val.height,
            "gap_by_level": dict(gap_val.group_by("tournament_level").len().sort("tournament_level").iter_rows())
            if gap_val.height else {},
            "gap_elo_updates": int(gap["is_update"].sum()) if gap is not None and gap.height else 0,
            "gap_stats_eligible": int(gap["stats_eligible"].sum()) if gap is not None and gap.height else 0,
            "gap_with_complete_stats_in_state": int(gap_val.filter(pl.col("stats_status") == "complete").height)
            if gap_val.height else 0,
            "gap_excluded_from_elo": dict(gap.filter(~pl.col("is_update")).group_by("exclusion_reason").len()
                                          .iter_rows()) if gap is not None and gap.height else {},
            "latest_release_batch": str(last_release) if last_release else None,
            "state_eligible_through": str(last_release + timedelta(days=7)) if last_release else None,
            "sandbox": str(sandbox), "state_snapshot_id": b["summary"]["state_snapshot_id"]}  # fmt: skip

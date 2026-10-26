"""Phase 5A run: access audit → authorized download → conversion → unchanged Phase 4A backfill (twice, for
determinism) → compatibility study → gap audit → readiness. Writes ``<workspace>/results.json`` (deterministic for a
fixed snapshot, configuration and ``as_of``) and ``<workspace>/run_metadata.json`` (wall-clock acquisition times,
timings, memory), which are kept apart.
"""

from __future__ import annotations

import json
import resource
import time
from datetime import date
from pathlib import Path

import polars as pl

from tennis_data.ledger.hashing import canonical, sha256_hex
from tennis_data.operations import access, coverage, readiness, tml
from tennis_data.operations.compatibility import study
from tennis_data.prospective.backfill import ingest_file, load_archive, run_backfill
from tennis_data.prospective.providers.fixture import FORMAT as FIXTURE_FORMAT


def _clean(obj):
    return json.loads(json.dumps(obj, default=str, sort_keys=True))


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def baseline_state(cfg) -> dict:
    """The state the backfill would extend: Phase 4A's own checkpoint (archive + its accepted rows), read-only."""
    from tennis_data.prospective.backfill import load_state_snapshot

    snap = load_state_snapshot(cfg.prospective)
    return {"state_snapshot_id": snap.snapshot_id, "matches_component": snap.components["matches"]}


def review_queue(decisions: pl.DataFrame, players: pl.DataFrame, ws) -> list[dict]:
    """Ambiguous and new players with their names and archive candidates (for a human reviewer; nothing approved)."""
    from tennis_data.prospective.identity import load_archive_players

    arch = {r["player_id"]: r for r in load_archive_players(ws).iter_rows(named=True)}
    pl_by = {r["provider_player_id"]: r for r in players.iter_rows(named=True)}
    out = []
    for d in decisions.filter(pl.col("category").is_in(["ambiguous", "unmatched"])).iter_rows(named=True):
        p = pl_by.get(d["provider_player_id"], {})
        out.append({"provider_player_id": d["provider_player_id"], "category": d["category"], "method": d["method"],
                    "name": f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip(),
                    "birth_date": str(p.get("birth_date")) if p.get("birth_date") else None,
                    "country": p.get("country"),
                    "candidates": [{"id": c, "name": f"{arch[c]['first_name'] or ''} {arch[c]['last_name'] or ''}".strip(),
                                    "birth_date": str(arch[c]["birth_date"]) if arch[c]["birth_date"] else None,
                                    "country": arch[c]["country"]} for c in (d["candidates"] or []) if c in arch]})  # fmt: skip
    return sorted(out, key=lambda r: (r["method"], r["name"]))


def run(cfg, as_of: date, transport=access.urllib_transport, probe: bool = True, fetch: bool = True) -> dict:
    t = {}
    meta: dict = {"stage_seconds": t}
    ws = cfg.workspace_prospective()
    acc = None
    if probe:
        s = time.perf_counter()
        acc = access.audit(cfg.providers, transport)
        t["access_audit"] = round(time.perf_counter() - s, 2)
        out = cfg.workspace_dir / "access"
        out.mkdir(parents=True, exist_ok=True)
        (out / "latest.json").write_text(json.dumps(acc, indent=2, sort_keys=True) + "\n")
    elif (cfg.workspace_dir / "access" / "latest.json").exists():
        acc = json.loads((cfg.workspace_dir / "access" / "latest.json").read_text())
    tml_ok = acc is not None and "tml_github" in acc.get("verified_sources", [])
    conv_log = None
    if fetch and tml_ok:
        s = time.perf_counter()
        meta["tml_manifest"] = tml.download(cfg, transport)
        t["download"] = round(time.perf_counter() - s, 2)
    tml_dir = cfg.workspace_dir / "tml" / cfg.tml["commit"]
    if (tml_dir / "manifest.json").exists():
        s = time.perf_counter()
        conv_path, conv_log = tml.convert_workspace(cfg)
        t["conversion"] = round(time.perf_counter() - s, 2)
        ingest_file(ws, conv_path, FIXTURE_FORMAT)  # idempotent: identical bytes are stored once
    s = time.perf_counter()
    b1 = run_backfill(ws)
    t["phase4a_backfill_identity_reconciliation_validation"] = round(time.perf_counter() - s, 2)
    b2 = run_backfill(ws, write=False)
    deterministic = canonical(_clean(b1["summary"])) == canonical(_clean(b2["summary"]))
    hist, _ = load_archive(ws)
    parsed = b1["parsed"]
    s = time.perf_counter()
    comp = study(parsed["matches"], b1["decisions"], hist, cfg)
    t["compatibility_study"] = round(time.perf_counter() - s, 2)
    s = time.perf_counter()
    q = b1["reconciliation"]["quarantine"]
    cov = coverage.audit(cfg, as_of, hist, parsed["matches"], b1["validated"], q, b1["decisions"])
    gap_rows = b1["validated"].filter(pl.col("tournament_date") > cfg.archive_last_week) if b1["validated"].height \
        else b1["validated"]  # fmt: skip
    cov["validated_levels"] = sorted(set(gap_rows["tournament_level"].to_list())) if gap_rows.height else []
    cov["newest_validated_date"] = str(gap_rows["tournament_date"].max()) if gap_rows.height else None
    t["gap_audit"] = round(time.perf_counter() - s, 2)
    fixtures = parsed["fixtures"]
    sched = {"verified_sources": [], "fixtures": fixtures.height, "fixtures_resolved": False,
             "actual_start_sources": [],
             "note": "no reachable provider offers schedules or actual starts (see provider_access.md)"}  # fmt: skip
    s = time.perf_counter()
    model = readiness.model_evidence(cfg)
    t["model_integrity"] = round(time.perf_counter() - s, 2)
    ev = readiness.gather(cfg, as_of, cov, comp, b1["decisions"], sched, acc, model)
    verdict = readiness.evaluate(ev, cfg.required_checks)
    base = baseline_state(cfg)
    summ = b1["summary"]
    state = {"starting_state_snapshot_id": base["state_snapshot_id"], "starting_matches_component":
             base["matches_component"], "new_source_snapshots": summ["raw_snapshots"],
             "validated_backfill_rows": summ["phase1_validated"], "elo_updates": summ["state_eligible_updates"],
             "serve_return_updates": summ["stats_eligible"], "final_state_snapshot_id": summ["state_snapshot_id"],
             "final_matches_component": summ["state_components"]["matches"],
             "match_state_unchanged": summ["state_components"]["matches"] == base["matches_component"],
             "archive_last_tournament_date": cov["archive_last_tournament_date"],
             "deterministic_rerun": deterministic}  # fmt: skip
    decisions = b1["decisions"]
    ident = {"by_category": _clean(decisions.group_by("category", "method", "usable").len()
                                   .sort("category", "method").to_dicts()),
             "players": decisions.height,
             "review_queue": _clean(review_queue(decisions, parsed["players"], ws))}  # fmt: skip
    results = _clean({"format": "tennis-operations-5a-1", "as_of": as_of.isoformat(),
                      "access": {"verified_sources": (acc or {}).get("verified_sources", []),
                                 "results": [{k: r[k] for k in ("provider", "kind", "status", "http_status", "credentials",
                                                                "credential_env", "detail", "data")}
                                             for r in (acc or {}).get("results", [])]},
                      "conversion": conv_log, "backfill": summ, "state": state, "identity": ident,
                      "compatibility": comp, "coverage": cov, "schedule": sched, "readiness": verdict})  # fmt: skip
    meta.update(peak_rss_mb=_rss_mb(), probed_at=(acc or {}).get("probed_at"))
    out = cfg.workspace_dir
    (out / "results.json").write_bytes(canonical(results) + b"\n")
    (out / "run_metadata.json").write_text(json.dumps(_clean(meta), indent=2, sort_keys=True) + "\n")
    return {"results": results, "meta": meta, "results_sha256": sha256_hex(canonical(results) + b"\n")}


def load_results(cfg) -> tuple[dict, dict]:
    d: Path = cfg.workspace_dir
    return json.loads((d / "results.json").read_text()), json.loads((d / "run_metadata.json").read_text())


def reconstruct_state(cfg, runs: int = 2) -> dict:
    """Run the frozen Phase 2 Elo + Phase 3A feature engines over the workspace state snapshot ``runs`` times and
    compare the resulting player-state tables byte for byte (deterministic reconstruction check)."""
    from tennis_data.prospective.backfill import load_state_snapshot
    from tennis_data.prospective.state import compute_state, table_hash

    ws = cfg.workspace_prospective()
    snap = load_state_snapshot(ws)
    out = {"state_snapshot_id": snap.snapshot_id, "components": snap.components, "runs": []}
    for _ in range(runs):
        s = time.perf_counter()
        st = compute_state(snap, [], ws, keep_history=True)
        hist = st["history"].sort("match_id")
        out["runs"].append({"seconds": round(time.perf_counter() - s, 1), "rows": hist.height,
                            "state_table_hash": table_hash(hist), "elo_configuration_hash": st["elo_configuration_hash"],
                            "feature_configuration_hash": st["feature_configuration_hash"]})  # fmt: skip
    out["deterministic"] = len({r["state_table_hash"] for r in out["runs"]}) == 1
    out["peak_rss_mb"] = _rss_mb()
    (cfg.workspace_dir / "state_reconstruction.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out

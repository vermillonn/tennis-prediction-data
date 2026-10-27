"""Controlled historical results-only backfill of an owner export (Phase 5A authorization of 2026-10-26).

Steps, each verified before the next; any failure stops the run without promotion and leaves production untouched:

1. **Approvals.** From the reviewed identity package, only overlap-corroborated mappings and name+birth-date
   mappings without blocking flags are candidates. Every candidate must pass every invariant (``INVARIANTS``),
   recomputed from the sources: proposal provenance, the Phase 4A decision (high confidence, single candidate,
   ``name_dob``), exact names and full birth dates under ``identity.norm``, one-to-one mapping (also against existing
   overrides), no indistinguishable namesake in the archive, not a candidate of an ambiguous player, plausible
   chronology, and for the corroborated class at least one confirming and no contradicting overlap match. Failures
   are removed and reported. Explicitly pending: the ``name_country`` mapping, every ambiguous and new player.
   Version 2 adds an explicit list of **individual overrides** approved by the owner after review; each must pass the
   separate ``INDIVIDUAL_CHECKS`` (the automatic rule above is not weakened). Approvals are written to a local
   manifest (``data/operations/exports/<label>/private/approvals_v<N>.toml``, git-ignored: it holds source-derived
   player codes) with evidence, class and the authorization reference; the proposed files and
   ``config/identity_overrides.toml`` are never modified.
2. **Baseline.** ``data/prospective/`` is stored as a content-addressed snapshot (objects by SHA-256 plus a manifest)
   and verified by restoring it into the staging directory and recomputing its state snapshot ID.
3. **Staging.** The import file keeps only matches whose two players are approved (matches with a pending player are
   quarantined here, so Phase 4A never assigns a new player ID); it is imported into the staging copy and the
   unchanged Phase 4A run (backfill, replay against the published Phase 2/3A/M4A outputs, freshness, readiness)
   runs there with ``existing overrides + approvals``.
4. **Verification.** Accepted rows: main-tour levels only, no statistics, no archive duplicates (including Roland
   Garros 2026), no new players, frozen weekly batching and ``historical_reconstruction`` mode, nothing after the
   prospective start, replay and integrity passed, frozen hashes unchanged, state reconstruction deterministic twice,
   prospective gate still closed, and every file outside ``data/operations`` unchanged.
5. **Promotion.** Only if every check passes: the staged directory is stored as a new versioned content-addressed
   snapshot (``historical-v<N>-<state id>``) and registered next to the baseline; earlier versions are never touched. ``data/prospective/`` (the active
   Phase 4A directory) is not modified; switching to the new snapshot is a separate owner decision.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.operations import exports, licence, readiness
from tennis_data.prospective.identity import load_archive_players, norm

AUTHORIZATION = "Phase 5A controlled historical backfill authorization, 2026-10-26 (reviewed package 40519b3)"
REVIEWER = "repository owner"
MAIN_TOUR_LEVELS = {"A", "D", "G", "M"}
INVARIANTS = ("proposal_provenance", "provider_in_verified_source", "phase4a_decision", "blocking_flags_clear",
              "names_exact", "birth_date_exact", "one_to_one", "no_indistinguishable_namesake",
              "not_ambiguous_candidate", "chronology", "overlap_corroboration")  # fmt: skip
INDIVIDUAL_CHECKS = ("provider_in_verified_source", "canonical_in_archive", "birth_date_exact", "one_to_one",
                     "not_candidate_of_pending", "holds_archive_history", "chronology", "evidence_recorded")  # fmt: skip


def run_name(label: str, version: int) -> str:
    return label if version == 1 else f"{label}-v{version}"


# ------------------------------------------------------------------------------------------ content-addressed store
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree(src: Path) -> dict[str, dict]:
    return {p.relative_to(src).as_posix(): {"sha256": _sha(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(src.rglob("*")) if p.is_file()}  # fmt: skip


def snapshot_dir(src: Path, store: Path, name: str, meta: dict) -> dict:
    """Store every file of ``src`` by content and write an immutable manifest (refuses to replace a different one)."""
    files = tree(src)
    for rel, f in files.items():
        obj = store / "objects" / f["sha256"][:2] / f["sha256"]
        if not obj.exists():
            obj.parent.mkdir(parents=True, exist_ok=True)
            tmp = obj.with_suffix(".tmp")
            tmp.write_bytes((src / rel).read_bytes())
            tmp.rename(obj)
    man = {"format": "content-addressed-snapshot-1", "name": name, "files": files,
           "tree_hash": _sha(json.dumps(files, sort_keys=True).encode()), **meta}  # fmt: skip
    path = store / "manifests" / f"{name}.json"
    text = json.dumps(man, indent=1, sort_keys=True, default=str) + "\n"
    if path.exists():
        old = json.loads(path.read_text())
        if old["tree_hash"] != man["tree_hash"]:
            raise FileExistsError(f"snapshot {name} already exists with different content")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return man


def verify_snapshot(store: Path, man: dict) -> list[str]:
    bad = []
    for rel, f in man["files"].items():
        obj = store / "objects" / f["sha256"][:2] / f["sha256"]
        if not obj.exists() or _sha(obj.read_bytes()) != f["sha256"]:
            bad.append(rel)
    if _sha(json.dumps(man["files"], sort_keys=True).encode()) != man["tree_hash"]:
        bad.append("<tree_hash>")
    return bad


def restore(store: Path, man: dict, dest: Path) -> None:
    """Rebuild a snapshot into an empty directory and check every file against the manifest."""
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    for rel, f in man["files"].items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((store / "objects" / f["sha256"][:2] / f["sha256"]).read_bytes())
    if tree(dest) != man["files"]:
        raise RuntimeError(f"restored tree of {man['name']} differs from its manifest")


# ------------------------------------------------------------------------------------------------------ approvals
def _full(first, last) -> str:
    return f"{first or ''} {last or ''}".strip()


def select_approvals(review: dict, decisions: dict[str, dict], provider: dict[str, dict], archive: dict[int, dict],
                     history: dict[int, dict], first_gap: dict[str, date], existing: dict[tuple, int],
                     proposed: dict[str, int]) -> dict:  # fmt: skip
    """Apply the authorization's rules to the reviewed package. Returns approved, rejected and pending entries.

    ``decisions``: provider id -> Phase 4A decision (category, canonical_player_id, method, candidates);
    ``provider``/``archive``: id -> {name, birth_date, country}; ``proposed``: provider id -> proposed canonical id.
    """
    by_name: dict = {}
    for cid, a in archive.items():
        by_name.setdefault(norm(a["name"]), []).append(cid)
    amb_cands = {c["id"] for q in review["ambiguous"] for c in q.get("candidates", [])}
    pending = [{"provider_player_id": q["provider_player_id"], "class": "ambiguous" if q["category"] != "unmatched"
                else "new_player", "reason": "not approved by the authorization (kept quarantined)",
                "defensible_proposal": q.get("proposed_canonical_id")} for q in review["ambiguous"]]  # fmt: skip
    cands, rejected = [], []
    for m in review["mappings"]:
        if m["method"] != "name_dob":
            pending.append({"provider_player_id": m["provider_player_id"], "class": m["evidence_class"],
                            "reason": f"Phase 4A method {m['method']}: no birth-date corroboration (explicitly left "
                                      "pending)", "defensible_proposal": m["canonical_player_id"]})  # fmt: skip
            continue
        cands.append(m)
    canon_count: dict = {}
    for m in cands:
        canon_count[m["canonical_player_id"]] = canon_count.get(m["canonical_player_id"], 0) + 1
    existing_canon = {v: k for k, v in existing.items()}
    approved = []
    for m in cands:
        pid, cid = m["provider_player_id"], m["canonical_player_id"]
        p, a, dec = provider.get(pid, {}), archive.get(cid, {}), decisions.get(pid, {})
        h, fg = history.get(cid, {}), first_gap.get(pid)
        dob = a.get("birth_date")
        namesakes = [x for x in by_name.get(norm(a.get("name")), []) if x != cid and (
            archive[x]["birth_date"] is None or archive[x]["birth_date"] == dob)]  # fmt: skip
        age = (fg - dob).days / 365.25 if fg and dob else None
        first_age = (h["first"] - dob).days / 365.25 if h.get("first") and dob else None
        checks = {
            "proposal_provenance": proposed.get(pid) == cid,
            "provider_in_verified_source": pid in provider,
            "phase4a_decision": dec.get("category") == "high_confidence" and dec.get("canonical_player_id") == cid
            and list(dec.get("candidates") or []) == [cid] and dec.get("method") == "name_dob",
            "blocking_flags_clear": not m["blocking"],
            "names_exact": bool(p.get("name")) and norm(p["name"]) == norm(a.get("name")),
            "birth_date_exact": p.get("birth_date") is not None and p.get("birth_date") == dob,
            "one_to_one": canon_count[cid] == 1 and existing.get(("tml_export", pid), cid) == cid
            and existing_canon.get(cid, ("tml_export", pid)) == ("tml_export", pid),
            "no_indistinguishable_namesake": not namesakes,
            "not_ambiguous_candidate": cid not in amb_cands,
            "chronology": age is not None and 14 <= age <= 45 and (first_age is None or first_age >= 14)
            and (h.get("last") is None or fg is None or h["last"] <= fg),
            "overlap_corroboration": m["evidence_class"] != "overlap_corroborated"
            or (m["overlap_matches_confirming"] >= 1 and m["overlap_matches_contradicting"] == 0),
        }  # fmt: skip
        failed = [k for k in INVARIANTS if not checks[k]]
        entry = {"provider": "tml_export", "provider_player_id": pid, "canonical_player_id": cid,
                 "evidence_class": m["evidence_class"], "provider_name": p.get("name"), "archive_name": a.get("name"),
                 "birth_date": str(dob) if dob else None, "overlap_matches_confirming": m["overlap_matches_confirming"],
                 "checks": checks}  # fmt: skip
        (rejected if failed else approved).append({**entry, "failed": failed} if failed else entry)
    return {"approved": approved, "rejected": rejected, "pending": pending}


def check_individual(overrides: dict[str, dict], approved: list[dict], decisions: dict[str, dict],
                     review: dict, provider: dict[str, dict], archive: dict[int, dict], history: dict[int, dict],
                     first_gap: dict[str, date], existing: dict[tuple, int]) -> list[dict]:  # fmt: skip
    """Checks for owner-approved individual overrides (provider id -> {canonical_player_id, justification}).

    The candidates of a provider player are its Phase 4A candidates plus those of the review package. A same-birth-
    date candidate (a duplicate archive record) must not hold more archive matches than the chosen ID, so the
    mapping never splits history; no archive ID is merged or rewritten.
    """
    auto = {e["provider_player_id"]: e["canonical_player_id"] for e in approved}
    chosen = {pid: o["canonical_player_id"] for pid, o in overrides.items()}
    used = list(auto.values()) + list(chosen.values()) + list(existing.values())
    cands = {pid: set(r.get("candidates") or []) for pid, r in decisions.items()}
    for q in review["ambiguous"]:
        cands.setdefault(q["provider_player_id"], set()).update(c["id"] for c in q.get("candidates", []))
    resolved = set(auto) | set(chosen) | {k[1] for k in existing}
    pending_cands = {c for pid, cs in cands.items() if pid not in resolved for c in cs}
    pending_cands |= {m["canonical_player_id"] for m in review["mappings"] if m["provider_player_id"] not in resolved}
    out = []
    for pid, o in sorted(overrides.items()):
        cid = o["canonical_player_id"]
        p, a, h, fg = provider.get(pid, {}), archive.get(cid, {}), history.get(cid, {}), first_gap.get(pid)
        dob = a.get("birth_date")
        age = (fg - dob).days / 365.25 if fg and dob else None
        first_age = (h["first"] - dob).days / 365.25 if h.get("first") and dob else None
        dup = [c for c in cands.get(pid, set()) - {cid} if archive.get(c, {}).get("birth_date") == dob]
        checks = {
            "provider_in_verified_source": pid in provider,
            "canonical_in_archive": cid in archive,
            "birth_date_exact": p.get("birth_date") is not None and p.get("birth_date") == dob,
            "one_to_one": used.count(cid) == 1 and pid not in auto and existing.get(("tml_export", pid), cid) == cid,
            "not_candidate_of_pending": cid not in pending_cands,
            "holds_archive_history": all(history.get(c, {}).get("archive_matches", 0) <= h.get("archive_matches", 0)
                                         for c in dup),
            "chronology": age is not None and 14 <= age <= 45 and (first_age is None or first_age >= 14),
            "evidence_recorded": bool(o.get("justification")),
        }  # fmt: skip
        out.append({"provider": "tml_export", "provider_player_id": pid, "canonical_player_id": cid,
                    "evidence_class": "individual_override", "provider_name": p.get("name"),
                    "archive_name": a.get("name"), "birth_date": str(dob) if dob else None,
                    "overlap_matches_confirming": o.get("overlap_matches_confirming", 0),
                    "justification": o["justification"], "authorization": o["authorization"],
                    "duplicate_archive_records": sorted(dup), "checks": checks,
                    "failed": [k for k in INDIVIDUAL_CHECKS if not checks[k]]})  # fmt: skip
    return out


def write_approvals(path: Path, approvals: list[dict], label: str, authorization: str = AUTHORIZATION) -> str:
    """Write the approved mappings (new file; refuses to replace a different one). Returns its SHA-256."""
    lines = [f"# Identity approvals for owner export {label}. Authorization: {authorization}.",
             "# Automatic entries passed every check in INVARIANTS; individual overrides passed INDIVIDUAL_CHECKS.",
             "# LOCAL ONLY (git-ignored): source-derived player codes. Never publish.",
             "# Proposed files and config/identity_overrides.toml are unchanged; this file is used together with them.", ""]  # fmt: skip
    for e in sorted(approvals, key=lambda e: e["provider_player_id"]):
        if e.get("justification"):  # individual override: the owner's decision with its recorded evidence
            cls, ev = "individual_override", e["justification"]
        else:
            cls = (
                "overlap_corroborated" if e["evidence_class"] == "overlap_corroborated" else "name_and_birth_date_exact"
            )
            ev = (f"name '{e['provider_name']}' = archive '{e['archive_name']}' (identity.norm); born {e['birth_date']} "
                  f"(exact); Phase 4A name_dob, single candidate; {e['overlap_matches_confirming']} overlap matches confirm")  # fmt: skip
        lines += ["[[mapping]]", 'provider = "tml_export"', f'provider_player_id = "{e["provider_player_id"]}"',
                  f"canonical_player_id = {e['canonical_player_id']}", f'reviewer = "{REVIEWER}"',
                  f'approval_class = "{cls}"', f'authorization = "{e.get("authorization", AUTHORIZATION)}"',
                  f'note = "{ev.replace(chr(34), chr(39))}"', ""]  # fmt: skip
    text = "\n".join(lines)
    tomllib.loads(text)
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} exists with different content; approvals are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _sha(text.encode())


def filter_import(conv: dict, approved: set[str]) -> tuple[dict, list[dict]]:
    """Keep only matches whose two players are approved; the others are quarantined with their pending players."""
    keep, held = [], []
    for m in conv["matches"]:
        pend = sorted({m["player1_id"], m["player2_id"]} - approved)
        if pend:
            held.append({"provider_match_id": m["id"], "tournament_id": m["tournament_id"], "round": m["round"],
                         "reason": "identity_pending", "pending_players": pend})  # fmt: skip
        else:
            keep.append(m)
    used = {p for m in keep for p in (m["player1_id"], m["player2_id"])}
    tids = {m["tournament_id"] for m in keep}
    doc = {**conv, "players": [p for p in conv["players"] if p["id"] in used],
           "tournaments": [t for t in conv["tournaments"] if t["id"] in tids], "matches": keep}  # fmt: skip
    return doc, held


# ---------------------------------------------------------------------------------------------------------- run
def data_tree(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``data/`` except the operations workspace (Phase 1-4C artifacts)."""
    d = root / "data"
    return {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in sorted(d.rglob("*"))
            if p.is_file() and not p.relative_to(d).as_posix().startswith("operations/")}  # fmt: skip


def reconstruct(ws, runs: int = 2) -> dict:
    """Frozen Elo + feature engines over the snapshot ``runs`` times (as ``pipeline.reconstruct_state``)."""
    import time

    from tennis_data.prospective.backfill import load_state_snapshot
    from tennis_data.prospective.state import compute_state, table_hash

    snap = load_state_snapshot(ws)
    out = {"state_snapshot_id": snap.snapshot_id, "runs": []}
    for _ in range(runs):
        t = time.perf_counter()
        st = compute_state(snap, [], ws, keep_history=True)
        hist = st["history"].sort("match_id")
        out["runs"].append({"seconds": round(time.perf_counter() - t, 1), "rows": hist.height,
                            "state_table_hash": table_hash(hist), "elo_configuration_hash": st["elo_configuration_hash"],
                            "feature_configuration_hash": st["feature_configuration_hash"]})  # fmt: skip
    out["deterministic"] = len({r["state_table_hash"] for r in out["runs"]}) == 1
    return out


def prepare(cfg, label: str, now: datetime, version: int = 1, individual: dict[str, dict] | None = None,
            authorization: str = AUTHORIZATION) -> dict:  # fmt: skip
    """Steps 1-3: approvals, verified baseline, staged import and the Phase 4A run in staging.

    ``individual``: owner-approved overrides (version 2 onwards); any failing ``INDIVIDUAL_CHECKS`` stops the run."""
    from tennis_data.prospective.backfill import ingest_file, load_state_snapshot
    from tennis_data.prospective.pipeline import run_phase4a

    root = PROJECT_ROOT
    run = run_name(label, version)
    rec: dict = {"label": label, "run": run, "version": version, "authorization": authorization,
                 "started_at": now.isoformat(), "steps": {}}  # fmt: skip
    before = data_tree(root)
    rec["data_tree_before"] = _sha(json.dumps(before, sort_keys=True).encode())
    d = exports.export_dir(cfg, label)
    man = json.loads((d / "manifest.json").read_text())
    hashes = licence.verify_hashes(d, man)
    rec["steps"]["source_hashes"] = {"ok": all(hashes.values()), "files": {f["original_name"]: f["sha256"] for f in
                                                                         man["files"]}}  # fmt: skip
    if not all(hashes.values()):
        raise RuntimeError("source files do not match their recorded hashes")

    # 1. approvals
    gap = exports.export_config(cfg, label, gap_only=True)
    gws = gap.workspace_prospective()
    review = json.loads((exports.private_dir(cfg, label) / "identity_review.json").read_text())
    conv = json.loads((gap.workspace_dir / "converted.json").read_text())
    provider = {p["id"]: {"name": _full(p.get("first_name"), p.get("last_name")),
                          "birth_date": date.fromisoformat(str(p["birth_date"])) if p.get("birth_date") else None}
                for p in conv["players"]}  # fmt: skip
    archive = {r["player_id"]: {"name": _full(r["first_name"], r["last_name"]), "birth_date": r["birth_date"]}
               for r in load_archive_players(gws).iter_rows(named=True)}  # fmt: skip
    from tennis_data.operations.authorization_review import history_table
    from tennis_data.prospective.backfill import load_archive

    hist, _ = load_archive(gws)
    history = history_table(hist)
    starts = {t["id"]: date.fromisoformat(str(t["start_date"])) for t in conv["tournaments"]}
    first_gap: dict = {}
    for m in conv["matches"]:
        for pid in (m["player1_id"], m["player2_id"]):
            first_gap[pid] = min(first_gap.get(pid, starts[m["tournament_id"]]), starts[m["tournament_id"]])
    dec = pl.read_parquet(gws.prospective_dir / "backfill" / "identity_decisions.parquet")
    decisions = {r["provider_player_id"]: r for r in dec.iter_rows(named=True)}
    prod = cfg.prospective
    existing = {}
    if Path(prod.overrides_path).exists():
        for e in tomllib.loads(Path(prod.overrides_path).read_text()).get("mapping", []):
            existing[(e["provider"], str(e["provider_player_id"]))] = int(e["canonical_player_id"])
    prop_path = d / "identity_overrides.proposed.toml"
    prop_sha = _sha(prop_path.read_bytes())
    proposed = {str(e["provider_player_id"]): int(e["canonical_player_id"])
                for e in tomllib.loads(prop_path.read_text()).get("mapping", [])}  # fmt: skip
    sel = select_approvals(review, decisions, provider, archive, history, first_gap, existing, proposed)
    ind = check_individual(individual or {}, sel["approved"], decisions, review, provider, archive, history,
                           first_gap, existing)  # fmt: skip
    bad = [(e["provider_player_id"], e["failed"]) for e in ind if e["failed"]]
    if bad:
        raise RuntimeError(f"individual overrides fail their checks (nothing staged): {bad}")
    ids = {e["provider_player_id"] for e in ind}
    sel = {"approved": sel["approved"], "individual": ind,
           "rejected": [e for e in sel["rejected"] if e["provider_player_id"] not in ids],
           "rejected_then_individually_approved": [e for e in sel["rejected"] if e["provider_player_id"] in ids],
           "pending": [e for e in sel["pending"] if e["provider_player_id"] not in ids]}  # fmt: skip
    appr_path = exports.private_dir(cfg, label) / f"approvals_v{version}.toml"
    appr_sha = write_approvals(appr_path, sel["approved"] + ind, label, authorization)
    rec["steps"]["approvals"] = {"approved": len(sel["approved"]), "individual": len(ind),
                                 "rejected": len(sel["rejected"]),
                                 "pending": len(sel["pending"]), "approvals_file": str(appr_path.relative_to(root)),
                                 "approvals_sha256": appr_sha, "proposed_file_sha256": prop_sha,
                                 "proposed_unchanged": _sha(prop_path.read_bytes()) == prop_sha}  # fmt: skip
    rec["identity"] = sel

    # 2. baseline: content-addressed, verified by restoring it as the staging directory
    base_state = load_state_snapshot(prod)
    store = cfg.workspace_dir / "snapshots"
    bname = f"prospective-baseline-{base_state.snapshot_id}"
    bman = snapshot_dir(prod.prospective_dir, store, bname, {
        "state_snapshot_id": base_state.snapshot_id, "state_components": base_state.components,
        "source": str(prod.prospective_dir.relative_to(root)), "created_at": now.isoformat(),
        "role": "baseline before the controlled historical backfill"})  # fmt: skip
    run_dir = cfg.workspace_dir / "controlled_backfill" / run
    if run_dir.exists():
        shutil.rmtree(run_dir)  # staging only; never the baseline or promoted snapshots
    staging = run_dir / "staging" / "prospective"
    restore(store, bman, staging)
    sws = replace(prod, prospective_dir=staging, reports_dir=run_dir / "staging" / "reports")
    rec["steps"]["baseline"] = {"name": bname, "tree_hash": bman["tree_hash"], "files": len(bman["files"]),
                                "objects_verified": not verify_snapshot(store, bman),
                                "restored_state_snapshot_id": load_state_snapshot(sws).snapshot_id,
                                "expected_state_snapshot_id": "3253426224f7ebc0"}  # fmt: skip
    b = rec["steps"]["baseline"]
    if not (b["objects_verified"] and b["restored_state_snapshot_id"] == b["expected_state_snapshot_id"]
            == base_state.snapshot_id):  # fmt: skip
        raise RuntimeError(f"baseline verification failed: {b}")

    # 3. staging import (approved players only) and the unchanged Phase 4A run
    approved_ids = {e["provider_player_id"] for e in sel["approved"] + ind}
    doc, held = filter_import(conv, approved_ids)
    imp = run_dir / "import.json"
    imp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    eff = run_dir / "identity_overrides.effective.toml"
    eff.write_text((Path(prod.overrides_path).read_text() if Path(prod.overrides_path).exists() else "") + "\n"
                   + appr_path.read_text())  # fmt: skip
    sws = replace(sws, overrides_path=eff)
    ing = ingest_file(sws, imp, "tennis-fixture-provider-1", retrieved_at=now)
    res = run_phase4a(sws, write_reports=True)
    rec["steps"]["staging"] = {"original_import_sha256": _sha((gap.workspace_dir / "converted.json").read_bytes()),
                               "filtered_import_sha256": _sha(imp.read_bytes()), "records_in": len(conv["matches"]),
                               "records_imported": len(doc["matches"]), "held_before_import": len(held),
                               "effective_overrides_sha256": _sha(eff.read_bytes()), "ingested": ing,
                               "staging_dir": str(staging.relative_to(root))}  # fmt: skip
    rec["held_before_import"] = held
    rec["phase4a"] = {"backfill": res["backfill"]["summary"], "replay": res["replay"]["summary"],
                      "integrity": res["integrity"], "freshness": res["freshness"], "readiness": res["readiness"],
                      "state_snapshot_id": res["snapshot"].snapshot_id}  # fmt: skip
    rec["data_tree_after_prepare"] = _sha(json.dumps(data_tree(root), sort_keys=True).encode())
    (run_dir / "run_record.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return rec


def verify(cfg, label: str, rec: dict, runs: int = 2) -> dict:
    """Step 4: every check on the staged state (including two reconstructions)."""
    from tennis_data.operations.compatibility import match_pairs
    from tennis_data.prospective.backfill import load_archive, load_state_snapshot, parse_store
    from tennis_data.prospective.pipeline import PredictionBlocked, check_enabled

    root = PROJECT_ROOT
    run_dir = cfg.workspace_dir / "controlled_backfill" / rec.get("run", label)
    prod = cfg.prospective
    sws = replace(prod, prospective_dir=run_dir / "staging" / "prospective",
                  reports_dir=run_dir / "staging" / "reports",
                  overrides_path=run_dir / "identity_overrides.effective.toml")  # fmt: skip
    bdir = sws.prospective_dir / "backfill"
    led, mt = pl.read_parquet(bdir / "ledger.parquet"), pl.read_parquet(bdir / "matches.parquet")
    q = pl.read_parquet(bdir / "reconciliation_quarantine.parquet")
    notes = pl.read_parquet(bdir / "reconciliation_notes.parquet")
    dec = pl.read_parquet(bdir / "identity_decisions.parquet")
    summ = rec["phase4a"]["backfill"]
    hist, _ = load_archive(sws)
    parsed = parse_store(sws)
    pairs = match_pairs(parsed["matches"], dec, hist, 7)["accounting"]["matched"]
    reg_base = _sha((prod.prospective_dir / "identity" / "new_players.parquet").read_bytes())
    reg_stage = _sha((sws.prospective_dir / "identity" / "new_players.parquet").read_bytes())
    try:
        check_enabled(sws)
        gate_closed = False
    except PredictionBlocked:
        gate_closed = True
    mo = readiness.model_evidence(cfg)
    rcn = reconstruct(sws, runs)
    after = data_tree(root)
    c = {
        "accepted_equals_imported_minus_quarantine": summ["accepted_records"]
        == rec["steps"]["staging"]["records_imported"] - (q.height if q.height else 0),
        "phase4a_quarantine_empty": q.height == 0,
        "main_tour_levels_only": set(mt["tournament_level"].unique().to_list()) <= MAIN_TOUR_LEVELS,
        "no_statistics_in_state": summ["with_complete_stats"] == 0 and summ["stats_eligible"] == 0
        and mt.filter(pl.col("player_a_svpt").is_not_null() | pl.col("player_b_svpt").is_not_null()).height == 0,
        "no_archive_duplicates": summ["rows_in_archive_tournaments"] == 0 and pairs == 0
        and notes.filter(pl.col("reason") == "duplicate_of_historical").height == 0,
        "no_roland_garros_2026": mt.filter(pl.col("tournament_name").str.to_lowercase().str.contains("roland")).height == 0,
        "no_new_player_identities": reg_base == reg_stage and dec.filter(pl.col("category") != "verified").height == 0,
        "all_players_manually_approved": dec.filter(pl.col("method") != "manual_override").height == 0,
        "frozen_weekly_batching": summ["chronology_policy"] == "weekly_batch_v1"
        and led["eligibility_basis"].unique().to_list() == ["inferred_weekly_batch"],
        "historical_reconstruction_only": led["prediction_mode"].unique().to_list() == ["historical_reconstruction"]
        and led["period"].unique().to_list() == ["backfill_gap"],
        "nothing_after_prospective_start": led.filter(pl.col("tournament_date") > prod.prospective_start).height == 0,
        "prospective_gate_closed": gate_closed and prod.backfill_status == "INCOMPLETE",
        "no_prediction_ledger": not (root / "data" / "ledger").exists(),
        "replay_passed": bool(rec["phase4a"]["replay"].get("passed")),
        "integrity_passed": bool(rec["phase4a"]["integrity"].get("passed")),
        "frozen_hashes_match": mo["hashes_match"],
        "reconstruction_deterministic": rcn["deterministic"],
        "elo_feature_configuration_unchanged": {r["elo_configuration_hash"] for r in rcn["runs"]} == {"1c330d4db8801ff6"}
        and {r["feature_configuration_hash"] for r in rcn["runs"]} == {"3d889b94456995d1"},
        "phase1_4c_data_unchanged": _sha(json.dumps(after, sort_keys=True).encode()) == rec["data_tree_before"],
        "proposed_identity_file_unchanged": rec["steps"]["approvals"]["proposed_unchanged"],
        "state_snapshot_reproducible": load_state_snapshot(sws).snapshot_id == rec["phase4a"]["state_snapshot_id"],
    }  # fmt: skip
    rec["verification"] = {"checks": c, "passed": all(c.values()), "reconstruction": rcn, "model": mo,
                           "archive_pair_matches": pairs, "phase4a_quarantine": q.height}  # fmt: skip
    (run_dir / "run_record.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return rec


def promote(cfg, label: str, rec: dict, now: datetime) -> dict:
    """Step 5 (only after every check passed): new versioned content-addressed snapshot plus registry entry."""
    from tennis_data.prospective.backfill import load_state_snapshot

    if not rec.get("verification", {}).get("passed"):
        raise RuntimeError("verification did not pass: nothing is promoted")
    version = rec.get("version", 1)
    run_dir = cfg.workspace_dir / "controlled_backfill" / rec.get("run", label)
    staging = run_dir / "staging" / "prospective"
    store = cfg.workspace_dir / "snapshots"
    sid = rec["phase4a"]["state_snapshot_id"]
    name = f"historical-v{version}-{sid}"
    man = snapshot_dir(staging, store, name, {
        "state_snapshot_id": sid, "parent": rec["steps"]["baseline"]["name"], "authorization": rec["authorization"],
        "label": label, "created_at": now.isoformat(), "approvals_sha256": rec["steps"]["approvals"]["approvals_sha256"],
        "effective_overrides_sha256": rec["steps"]["staging"]["effective_overrides_sha256"],
        "filtered_import_sha256": rec["steps"]["staging"]["filtered_import_sha256"],
        "source_files": rec["steps"]["source_hashes"]["files"],
        "reconstruction_hash": rec["verification"]["reconstruction"]["runs"][0]["state_table_hash"],
        "role": "results-only historical backfill (main tour), frozen weekly batching; not a prospective state"})  # fmt: skip
    check = run_dir / "promoted_restore_check"
    shutil.rmtree(check, ignore_errors=True)
    restore(store, man, check)
    restored = load_state_snapshot(replace(cfg.prospective, prospective_dir=check)).snapshot_id
    shutil.rmtree(check)
    reg_path = store / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"format": "snapshot-registry-1", "snapshots": []}
    names = {s["name"] for s in reg["snapshots"]}
    for entry in ({"name": rec["steps"]["baseline"]["name"], "version": 0, "state_snapshot_id": "3253426224f7ebc0",
                   "tree_hash": rec["steps"]["baseline"]["tree_hash"], "active_phase4a_directory": True},
                  {"name": name, "version": version, "state_snapshot_id": sid, "tree_hash": man["tree_hash"],
                   "parent": rec["steps"]["baseline"]["name"], "active_phase4a_directory": False,
                   "registered_at": now.isoformat()}):  # fmt: skip
        if entry["name"] not in names:
            reg["snapshots"].append(entry)
    reg_path.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    rec["promotion"] = {"promoted": restored == sid and not verify_snapshot(store, man), "name": name,
                        "tree_hash": man["tree_hash"], "restored_state_snapshot_id": restored,
                        "registry": str(reg_path.relative_to(PROJECT_ROOT))}  # fmt: skip
    (run_dir / "run_record.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return rec


# ---------------------------------------------------------------------------------------------------------- report
def match_status(cfg, label: str, rec: dict) -> list[dict]:
    """Status of every record of the backfill set: accepted (with Elo eligibility) or quarantined (with reason).
    Record-level and source-derived: written only to the export's local ``private`` directory."""
    run_dir = cfg.workspace_dir / "controlled_backfill" / rec.get("run", label)
    gap = exports.export_config(cfg, label, gap_only=True)
    conv = json.loads((gap.workspace_dir / "converted.json").read_text())
    tours = {t["id"]: t for t in conv["tournaments"]}
    bdir = run_dir / "staging" / "prospective" / "backfill"
    led = pl.read_parquet(bdir / "ledger.parquet")
    acc = {r["provider_match_id"]: r for r in led.iter_rows(named=True)}
    q = pl.read_parquet(bdir / "reconciliation_quarantine.parquet")
    p4q = {r["provider_match_id"]: r["reason"] for r in q.iter_rows(named=True)} if q.height else {}
    held = {h["provider_match_id"]: h for h in rec["held_before_import"]}
    cls = {p["provider_player_id"]: p["class"] for p in rec["identity"]["pending"]}
    cls |= {e["provider_player_id"]: "rejected_by_invariant" for e in rec["identity"]["rejected"]}
    out = []
    for m in conv["matches"]:
        t = tours[m["tournament_id"]]
        row = {"provider_match_id": m["id"], "tournament": t["name"], "level": t["level"], "event_start": str(t["start_date"]),
               "round": m["round"]}  # fmt: skip
        if m["id"] in acc:
            a = acc[m["id"]]
            row |= {"status": "accepted", "match_id": a["match_id"], "prediction_batch": str(a["prediction_batch"]),
                    "release_batch": str(a["release_batch"]), "elo_update": bool(a["is_update"]),
                    "elo_exclusion": a["exclusion_reason"]}  # fmt: skip
        elif m["id"] in held:
            row |= {"status": "quarantined", "reason": "identity_pending",
                    "pending_players": [{"id": p, "class": cls.get(p, "unknown")} for p in held[m["id"]]["pending_players"]]}  # fmt: skip
        else:
            row |= {"status": "quarantined", "reason": p4q.get(m["id"], "not_accepted")}
        out.append(row)
    return out


def render(cfg, label: str, rec: dict, rows: list[dict]) -> str:
    from collections import Counter

    s, v, ph = rec["steps"], rec["verification"], rec["phase4a"]
    acc = [r for r in rows if r["status"] == "accepted"]
    qua = [r for r in rows if r["status"] != "accepted"]
    ident = rec["identity"]
    by_cls = Counter(e["evidence_class"] for e in ident["approved"])
    rej_cls = Counter(e["evidence_class"] for e in ident["rejected"])
    pend_cls = Counter(p["class"] for p in ident["pending"])
    rcn = v["reconstruction"]
    promo = rec.get("promotion", {})
    ind = ident.get("individual", [])
    rti = ident.get("rejected_then_individually_approved", [])
    version = rec.get("version", 1)
    L = [f"# Controlled historical backfill, version {version} ({label})", "",
         f"Authorization: {rec['authorization']}. Run started {rec['started_at']}. Results only, ATP main tour, frozen weekly "
         "batching. Production `data/prospective/` (state `3253426224f7ebc0`) is unchanged; the backfilled state is a "
         "new versioned local snapshot. Gates: backfill INCOMPLETE, prospective predictions DISABLED.", "",
         "## 1. Identity decisions", ""]  # fmt: skip
    L += ["| evidence class | proposed | approved | rejected by an invariant | pending |", "|---|---|---|---|---|",
          f"| overlap-corroborated | 205 | {by_cls['overlap_corroborated']} | {rej_cls['overlap_corroborated']} | 0 |",
          f"| name and birth date | 103 | {by_cls['name_and_birth_date_only']} | {rej_cls['name_and_birth_date_only']} | "
          f"{pend_cls['name_and_birth_date_only']} (the `name_country` mapping) |",
          f"| ambiguous | 26 | 0 | 0 | {pend_cls['ambiguous']} |", f"| new player | 1 | 0 | 0 | {pend_cls['new_player']} |",
          *([f"| **individual overrides** (owner decision after review) | - | **{len(ind)}** "
             f"({len(rti)} previously rejected by the automatic rule, {len(ind) - len(rti)} previously ambiguous) | - "
             "| - |"] if ind else []),
          "", f"Approval manifest: local and git-ignored (`data/operations/exports/<label>/private/approvals_v{version}"
          f".toml`, SHA-256 "
          f"`{s['approvals']['approvals_sha256'][:16]}…`), each entry with reviewer, approval class, authorization "
          "reference and evidence; it holds source-derived player codes and is not published. The proposed mappings "
          f"(`identity_overrides.proposed.toml`, `{s['approvals']['proposed_file_sha256'][:16]}…`) and "
          "`config/identity_overrides.toml` are unchanged. Invariants: " + ", ".join(f"`{i}`" for i in INVARIANTS)
          + ". Names are compared with `identity.norm` (NFKD accents removed, case-folded, punctuation-insensitive, "
          "listed letter folds) on `first last`; birth dates must be equal full dates on both sides."
          + (" Individual overrides passed " + ", ".join(f"`{i}`" for i in INDIVIDUAL_CHECKS) + "." if ind else ""),
          "", f"Rejected by the automatic rule and not individually approved: {len(ident['rejected'])}"
          + (" (" + ", ".join(f"`{k}` {n}" for k, n in sorted(Counter(f for e in ident["rejected"]
                                                                    for f in e["failed"]).items())) + ")"
             if ident["rejected"] else "") + ". Per-mapping evidence is kept in the local run record."]  # fmt: skip
    st = Counter(r["status"] for r in rows)
    reasons = Counter(c["class"] for r in qua for c in r.get("pending_players", []))
    L += ["", "## 2. Matches", "", "| item | count |", "|---|---|",
          f"| backfill-set records (events after the archive) | {len(rows):,} |",
          f"| **accepted (newly imported)** | **{st['accepted']:,}** |",
          f"| **Elo updates (frozen eligibility)** | **{sum(r['elo_update'] for r in acc):,}** |",
          "| not Elo-eligible | " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(r["elo_exclusion"] for r in acc
                                                                                  if not r["elo_update"]).items())) + " |",
          f"| quarantined (identity pending) | {len(qua)} (player slots: " + ", ".join(
              f"{k} {n}" for k, n in sorted(reasons.items())) + ") |",
          f"| Phase 4A quarantine | {v['phase4a_quarantine']} |",
          f"| deduplicated against the archive | {v['archive_pair_matches']} in the backfill set (Roland Garros 2026 "
          "starts before the gap: 0 records; on the full-file validation run 1,233 duplicates were recognised, 117 of "
          "them Roland Garros) |",
          f"| statistics removed (incompatible definition) | {ph['backfill']['reconciliation_notes'][0]['len'] if ph['backfill']['reconciliation_notes'] else 0} "
          f"(state has {ph['backfill']['with_complete_stats']} rows with statistics) |",
          "| excluded | all January–May records (archive period, validation only); `2026_challenger.csv` (never parsed) |",
          "", "Per-match status is record-level and source-derived: it is kept locally "
          "(`data/operations/exports/<label>/private/`), with its SHA-256 below.", "",
          "## 3. Coverage by batch", ""]  # fmt: skip
    wk: dict = {}
    for r in rows:
        b = r.get("prediction_batch") or _batch(r["event_start"])
        w = wk.setdefault(b, Counter())
        w["records"] += 1
        w["accepted"] += r["status"] == "accepted"
        w["elo"] += bool(r.get("elo_update"))
        w["quarantined"] += r["status"] != "accepted"
    L += _tbl(
        ["batch", "records", "accepted", "Elo updates", "quarantined"],
        [[b, w["records"], w["accepted"], w["elo"], w["quarantined"]] for b, w in sorted(wk.items())],
    )
    L += ["", "## 4. Coverage by tournament", ""]
    ev: dict = {}
    for r in rows:
        e = ev.setdefault((r["event_start"], r["tournament"], r["level"]), Counter())
        e["records"] += 1
        e["accepted"] += r["status"] == "accepted"
        e["elo"] += bool(r.get("elo_update"))
    L += _tbl(["start", "tournament", "level", "records", "accepted", "Elo updates", "quarantined"],
              [[k[0], k[1], k[2], e["records"], e["accepted"], e["elo"], e["records"] - e["accepted"]]
               for k, e in sorted(ev.items())])  # fmt: skip
    L += ["", "## 5. Quarantined players (identity pending)", ""]
    pc: dict = {}
    for r in qua:
        for c in r.get("pending_players", []):
            pc.setdefault(c["class"], {}).setdefault(c["id"], 0)
            pc[c["class"]][c["id"]] += 1
    L += _tbl(["class", "players", "match slots"], [[k, len(v), sum(v.values())] for k, v in sorted(pc.items())])
    L += ["", "## 6. State, reproducibility and integrity", "", "| item | value |", "|---|---|",
          f"| baseline (content-addressed, verified by restore) | `{s['baseline']['name']}`, tree `{s['baseline']['tree_hash'][:16]}…`, "
          f"{s['baseline']['files']} files, restored state `{s['baseline']['restored_state_snapshot_id']}` |",
          f"| new state snapshot | `{ph['state_snapshot_id']}` |",
          "| reconstruction (frozen engines) | " + ", ".join(f"`{r['state_table_hash']}` ({r['rows']:,} matches, {r['seconds']} s)"
                                                       for r in rcn["runs"]) + f"; deterministic **{rcn['deterministic']}** |",
          f"| replay (published Phase 2/3A/M4A) | passed **{ph['replay']['passed']}**, {ph['replay']['cases']} cases, max abs "
          f"error {ph['replay']['max_abs_error']:.1e}, max Elo error {ph['replay']['max_elo_abs_error']} |",
          f"| M4A frozen / configuration hash | `{v['model']['frozen_hash']}` / `{v['model']['configuration_hash']}` "
          f"(match {v['model']['hashes_match']}) |",
          f"| Elo / feature configuration hash | `{rcn['runs'][0]['elo_configuration_hash']}` / "
          f"`{rcn['runs'][0]['feature_configuration_hash']}` |",
          f"| promoted | **{promo.get('promoted', False)}**: `{promo.get('name', '-')}` (tree `{str(promo.get('tree_hash', '-'))[:16]}…`, "
          f"restored state `{promo.get('restored_state_snapshot_id', '-')}`), registry `{promo.get('registry', '-')}` |",
          "", "Verification checks:", ""]  # fmt: skip
    L += [f"- `{k}`: {'pass' if ok else '**FAIL**'}" for k, ok in v["checks"].items()]
    L += ["", "## 7. Operational readiness after the backfill (staged state)", ""]
    L += [f"- {b}" for b in ph["readiness"]["blockers"]]
    fr = ph["freshness"]["checks"]
    L += ["", "Freshness probe: " + ", ".join(f"`{k}` {c['status']}" for k, c in fr.items()) + ".", ""]
    if rec.get("matches_sha256"):
        L += [f"Local per-match status SHA-256: `{rec['matches_sha256']}`.", ""]
    return "\n".join(L)


def _tbl(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |"
                                                                       for r in rows]  # fmt: skip


def _batch(start: str) -> str:
    from datetime import timedelta

    d = date.fromisoformat(start)
    return str(d - timedelta(days=(d.weekday() - 5) % 7))

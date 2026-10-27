"""Final safety review of an owner export before backfill authorization (read-only; nothing is approved or applied).

Builds, from the dry-run outputs already on disk:

- **chronology**: the effect of the proposed ``state_availability_v1`` safeguard (``operations.availability``) on the
  backfill rows, and a release review of every new event under the unchanged frozen batching, with flags and the
  future cases this export cannot exercise;
- **identity review package** for the proposed high-confidence mappings, split into overlap-corroborated and
  name+birth-date-only, with machine-checkable tests: one-to-one mapping, conflicts with existing overrides or with
  ambiguous candidates, same-name archive records (with and without the same birth date), birth-date and country
  conflicts, and the archive history of the canonical player (unexpected ages, long inactivity, no history);
- **ambiguous and new players**: per provider ID the candidates with their evidence, a proposal only where one
  candidate is defensible (same birth date, same country, compatible name, and either the only such record or the
  only one with archive matches), and the number of affected matches. Everything else stays unresolved;
- **simulation verification**: the counts recomputed from the sandbox tables, a fresh no-write re-run compared with
  the recorded snapshot, and the recorded double reconstruction;
- frozen-model integrity, production state and status gates.

``python -m tennis_data.operations.authorization_review --label L --as-of DATE`` writes the full package and the
machine-auditable identity package (record-level, source-derived) to the git-ignored export directory
(``private/identity_review.json``) and an aggregate-only report to ``reports/operations/``. Approval stays manual: copy approved entries into ``config/identity_overrides.toml``.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.operations import availability, backfill_plan, chronology_audit, exports, licence, pipeline, readiness
from tennis_data.prospective.backfill import load_archive, run_backfill
from tennis_data.prospective.identity import load_archive_players, norm

HARD = ("not_one_to_one", "conflicts_existing_override", "canonical_is_ambiguous_candidate",
        "same_name_same_birth_date_record", "birth_date_conflict", "overlap_contradicted")  # fmt: skip
TEAM_WORDS = ("davis cup", "united cup", "laver cup")
# ISO 3166 alpha-3 codes that differ from the IOC codes the archive uses (an alias is not a conflict).
ISO_TO_IOC = {"SVN": "SLO", "PRT": "POR", "URY": "URU", "DEU": "GER", "NLD": "NED", "CHE": "SUI", "HRV": "CRO",
              "GRC": "GRE", "ZAF": "RSA", "DNK": "DEN", "BGR": "BUL", "PRY": "PAR", "CHL": "CHI", "LVA": "LAT",
              "MYS": "MAS", "IDN": "INA", "PHL": "PHI", "SAU": "KSA", "ARE": "UAE", "TWN": "TPE", "MCO": "MON",
              "VNM": "VIE", "IRN": "IRI", "NGA": "NGR", "DZA": "ALG", "KWT": "KUW", "SLV": "ESA", "GTM": "GUA",
              "CRI": "CRC", "BRB": "BAR", "BHS": "BAH", "ZWE": "ZIM", "TGO": "TOG", "MDG": "MAD", "LBN": "LIB"}  # fmt: skip


def same_country(a: str | None, b: str | None) -> bool | None:
    if not a or not b:
        return None
    return ISO_TO_IOC.get(a, a) == ISO_TO_IOC.get(b, b)


# ---------------------------------------------------------------------------------------------------------- checks
def history_table(hist: pl.DataFrame) -> dict[int, dict]:
    both = pl.concat([hist.select(pl.col("player_a_id").alias("pid"), "tournament_date"),
                      hist.select(pl.col("player_b_id").alias("pid"), "tournament_date")])  # fmt: skip
    g = both.group_by("pid").agg(pl.len().alias("n"), pl.col("tournament_date").min().alias("first"),
                                 pl.col("tournament_date").max().alias("last"))  # fmt: skip
    return {
        r["pid"]: {"archive_matches": r["n"], "first": r["first"], "last": r["last"]} for r in g.iter_rows(named=True)
    }


def _age(dob: date | None, on: date | None) -> float | None:
    return round((on - dob).days / 365.25, 1) if dob and on else None


def check_mappings(proposals: list[dict], archive: dict[int, dict], history: dict[int, dict],
                   provider: dict[str, dict], first_gap: dict[str, date], existing: dict[tuple, int],
                   ambiguous_candidates: set[int]) -> list[dict]:  # fmt: skip
    """Machine-checkable tests for each proposed mapping. ``archive``: id -> {name, birth_date, country}."""
    by_name = defaultdict(list)
    for pid, a in archive.items():
        by_name[norm(a["name"])].append(pid)
    can_count = Counter(p["canonical_player_id"] for p in proposals)
    prov_count = Counter((p["provider"], p["provider_player_id"]) for p in proposals)
    existing_by_canon = defaultdict(set)
    for k, v in existing.items():
        existing_by_canon[v].add(k)
    out = []
    for p in proposals:
        cid, key = p["canonical_player_id"], (p["provider"], p["provider_player_id"])
        a, pr = archive.get(cid, {}), provider.get(p["provider_player_id"], {})
        h = history.get(cid, {"archive_matches": 0, "first": None, "last": None})
        same_name = [x for x in by_name.get(norm(a.get("name")), []) if x != cid]
        same_name_same_dob = [
            x for x in same_name if a.get("birth_date") and archive[x]["birth_date"] == a["birth_date"]
        ]
        fg = first_gap.get(p["provider_player_id"])
        flags = []
        if can_count[cid] > 1 or prov_count[key] > 1:
            flags.append("not_one_to_one")
        if (key in existing and existing[key] != cid) or (existing_by_canon.get(cid, set()) - {key}):
            flags.append("conflicts_existing_override")
        if cid in ambiguous_candidates:
            flags.append("canonical_is_ambiguous_candidate")
        if same_name_same_dob:
            flags.append("same_name_same_birth_date_record")
        if pr.get("birth_date") is None or a.get("birth_date") is None or pr["birth_date"] != a["birth_date"]:
            flags.append("birth_date_conflict")
        if p.get("overlap_matches_contradicting"):
            flags.append("overlap_contradicted")
        if same_name and not same_name_same_dob:
            flags.append("same_name_other_birth_date")
        if same_country(pr.get("country"), a.get("country")) is False:
            flags.append("country_differs")
        age = _age(a.get("birth_date"), fg)
        if age is not None and not 15 <= age <= 42:
            flags.append("unusual_age_at_gap_match")
        if h["archive_matches"] == 0:
            flags.append("no_archive_history")
        elif fg and h["last"] and (fg - h["last"]).days > 3 * 365:
            flags.append("archive_inactive_over_3_years")
        first_age = _age(a.get("birth_date"), h["first"])
        if first_age is not None and first_age < 14:
            flags.append("archive_history_before_age_14")
        out.append({"provider": p["provider"], "provider_player_id": p["provider_player_id"],
                    "canonical_player_id": cid, "method": p.get("method"), "archive_name": a.get("name"),
                    "birth_date": str(a["birth_date"]) if a.get("birth_date") else None,
                    "archive_country": a.get("country"), "provider_country": pr.get("country"),
                    "evidence_class": "overlap_corroborated" if p.get("overlap_matches_confirming") and not
                    p.get("overlap_matches_contradicting") else "name_and_birth_date_only",
                    "overlap_matches_confirming": p.get("overlap_matches_confirming", 0),
                    "overlap_matches_contradicting": p.get("overlap_matches_contradicting", 0),
                    "archive_matches": h["archive_matches"], "archive_first": str(h["first"]) if h["first"] else None,
                    "archive_last": str(h["last"]) if h["last"] else None, "first_gap_match": str(fg) if fg else None,
                    "age_at_first_gap_match": age, "same_name_other_records": sorted(same_name),
                    "flags": flags, "blocking": [f for f in flags if f in HARD],
                    "review_decision": "PENDING"})  # fmt: skip
    return out


def _name_compatible(a: str, b: str) -> bool:
    """Same tokens, one a subset of the other ('Daniel Merida' / 'Daniel Merida Aguilar'), or equal without spaces
    ('Soonwoo Kwon' / 'Soon Woo Kwon'). A different spelling ('Abdullah' / 'Abedallah') is not compatible."""
    ta, tb = set(norm(a).split()), set(norm(b).split())
    joined = norm(a).replace(" ", "") == norm(b).replace(" ", "")
    return bool(ta and tb) and (ta <= tb or tb <= ta or joined)


def propose_ambiguous(entry: dict, history: dict[int, dict], affected: int) -> dict:
    """One review row for an ambiguous or new provider player. Proposes a candidate only when it is defensible."""
    dob, ctry, name = entry.get("birth_date"), entry.get("country"), entry.get("name") or ""
    cands = []
    for c in entry.get("candidates", []):
        h = history.get(c["id"], {"archive_matches": 0, "last": None})
        cands.append({"id": c["id"], "name": c["name"], "birth_date": c.get("birth_date"), "country": c.get("country"),
                      "same_birth_date": bool(dob and c.get("birth_date") == dob),
                      "same_country": bool(same_country(ctry, c.get("country"))),
                      "name_compatible": _name_compatible(name, c["name"]),
                      "archive_matches": h["archive_matches"], "archive_last": str(h["last"]) if h["last"] else None})  # fmt: skip
    strong = [c for c in cands if c["same_birth_date"] and c["same_country"] and c["name_compatible"]]
    proposal, reason = None, None
    if entry["category"] == "unmatched":
        reason = "no archive candidate: a new canonical ID would be assigned after approval"
    elif not dob:
        reason = "provider record has no birth date"
    elif not strong:
        spelled = [c for c in cands if c["same_birth_date"] and c["same_country"]]
        undated = [c for c in cands if c["birth_date"] is None and c["name_compatible"] and c["same_country"]]
        if spelled:
            reason = (f"{spelled[0]['id']} has the same birth date and country but a different spelling "
                      f"({spelled[0]['name']}): owner check needed")  # fmt: skip
        elif undated:
            reason = f"name and country match {undated[0]['id']}, but the archive record has no birth date"
        else:
            reason = "no candidate with the same birth date, country and a compatible name"
    elif len(strong) == 1:
        proposal, reason = strong[0]["id"], "single candidate with the same birth date, country and compatible name"
        others = [c["id"] for c in cands if c is not strong[0] and c["birth_date"] is None and c["name_compatible"]]
        if others:
            reason += f"; archive also has an undated record with the same name ({', '.join(map(str, others))})"
    else:
        active = [c for c in strong if c["archive_matches"] > 0]
        if len(active) == 1:
            proposal = active[0]["id"]
            reason = (f"{len(strong)} archive records share name and birth date (duplicate archive IDs); only "
                      f"{proposal} has archive matches")  # fmt: skip
        else:
            reason = (f"{len(strong)} archive records share name and birth date and "
                      f"{'several' if active else 'none'} have archive matches")  # fmt: skip
    return {"provider_player_id": entry["provider_player_id"], "name": name, "birth_date": dob, "country": ctry,
            "method": entry["method"], "category": entry["category"], "affected_matches": affected,
            "proposed_canonical_id": proposal, "grade": "defensible, owner review required" if proposal else "UNRESOLVED",
            "reason": reason, "candidates": cands, "review_decision": "PENDING"}  # fmt: skip


def review_events(events: list[dict]) -> list[dict]:
    """Flags for each new event's release under the frozen batching. Each event: tournament, level, draw_size,
    weeks, start, last_match, batch, release_batch, enters_state, played_on_or_after."""
    out = []
    for e in events:
        start, last, enters = (date.fromisoformat(e[k]) for k in ("start", "last_match", "enters_state"))
        span = (last - start).days + 1
        flags = []
        if e["played_on_or_after"]:
            flags.append("results_released_on_or_before_match_day")
        if span > 7 * e["weeks"]:
            flags.append("event_longer_than_frozen_weeks")
        if start.weekday() not in (5, 6, 0):
            flags.append("starts_mid_week")
        if any(w in e["tournament"].lower() for w in TEAM_WORDS):
            flags.append("team_event")
        if e["level"] == "M" and (e["draw_size"] or 0) < 96:
            flags.append("masters_below_96_draw_one_week_rule")
        out.append({**e, "span_days": span, "overrun_days": max(0, (last - enters).days + 1), "flags": flags})
    return out


# ------------------------------------------------------------------------------------------------------- assembly
def _events(gap, d: Path) -> list[dict]:
    sim = d / "simulation" / "backfill"
    src = d / "workspace" / "tml" / gap.tml["commit"]
    own = chronology_audit.tml_match_dates({n: (src / n).read_bytes() for n in gap.tml["files"]})
    led = pl.read_parquet(sim / "ledger.parquet")
    mt = pl.read_parquet(sim / "matches.parquet").select("match_id", "tournament_name", "tournament_level", "draw_size")
    g = led.join(mt, on="match_id").with_columns(
        pl.col("provider_match_id").replace_strict(own, default=None, return_dtype=pl.Date).alias("match_date"),
        pl.col("state_eligible_at").dt.date().alias("enters"))  # fmt: skip
    ev = []
    for (name, lv, draw), x in g.group_by("tournament_name", "tournament_level", "draw_size"):
        ev.append({"tournament": name, "level": lv, "draw_size": draw, "weeks": int(x["event_weeks"].max()),
                   "start": str(x["tournament_date"].min()), "last_match": str(x["match_date"].max()),
                   "batch": str(x["prediction_batch"].min()), "release_batch": str(x["release_batch"].max()),
                   "enters_state": str(x["enters"].min()), "matches": x.height,
                   "played_on_or_after": x.filter(pl.col("match_date") >= pl.col("enters")).height})  # fmt: skip
    return sorted(review_events(ev), key=lambda e: (e["start"], e["tournament"])), g


FUTURE_CASES = [
    ("United Cup (late December / January)", "level A, draw 18, starts on a Friday and runs about 10 days; the frozen "
     "rule gives one week, so finals fall after the release; dated archive and backfill events disagree by a batch"),
    ("Davis Cup Finals (November)", "level D, a week-long knockout; one-week rule; not in this export"),
    ("ATP Finals (November)", "Sunday to Sunday, eight days; level mapping and one-week rule not exercised here"),
    ("Masters with draw below 96 (Monte Carlo, Paris)", "one-week rule; finals on Sunday after the release Saturday"),
    ("Olympic tennis", "level O, two-week rule; no 2026 event"),
    ("Next Gen ATP Finals", "round robin; level mapping not exercised here"),
    ("Events moved by weather or scheduling (finals on Monday)", "only per-match dates reveal the overrun"),
]  # fmt: skip


def build(cfg, label: str, as_of: date) -> dict:
    d = exports.export_dir(cfg, label)
    full, gap = exports.export_config(cfg, label), exports.export_config(cfg, label, gap_only=True)
    ws = gap.workspace_prospective()
    man = json.loads((d / "manifest.json").read_text())
    dry = json.loads((d / "dry_run.json").read_text())
    pol = availability.load_policy()

    # Chronology: per-event review and the safeguard's effect.
    events, g = _events(gap, d)
    t0 = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
    times = [t0 + timedelta(days=k) for k in (0, 1, 2)]
    safeguard = availability.evaluate(g, pol, times)
    # The match-day rule alone: max(frozen release, match date + buffer), whatever the ingestion time.
    late = availability.annotate(g.filter(pl.col("match_date") >= pl.col("enters")), pol)
    alone = [max(r["plausible_end_at"], r["state_eligible_at"]) for r in late.iter_rows(named=True)]
    safeguard["the_121"] = {
        "rows": late.height,
        "available_after_match_day_buffer": sum(1 for x, r in zip(alone, late.iter_rows(named=True), strict=True)
                                                if x >= r["plausible_end_at"]),
        "delay_days_vs_frozen": {str(k): n for k, n in sorted(Counter(
            int((x - r["state_eligible_at"]).total_seconds() // 86400)
            for x, r in zip(alone, late.iter_rows(named=True), strict=True)).items())}}  # fmt: skip
    no_ingest = availability.annotate(g.with_columns(pl.lit(None, dtype=availability.UTC_TS).alias("ingested_at")), pol)
    safeguard["without_ingestion_time"] = dict(Counter(no_ingest["availability_status"]))

    # Identities.
    gprop = backfill_plan.identity_proposal(gap)
    corr = {r["provider_player_id"]: r for r in backfill_plan.identity_proposal(full)["proposed"]}
    for r in gprop["proposed"]:
        c = corr.get(r["provider_player_id"])
        if c:
            r.update({k: c[k] for k in ("overlap_matches_confirming", "overlap_matches_contradicting")})
    arch_p = load_archive_players(ws)
    archive = {r["player_id"]: {"name": f"{r['first_name'] or ''} {r['last_name'] or ''}".strip(),
                                "birth_date": r["birth_date"], "country": r["country"]}
               for r in arch_p.iter_rows(named=True)}  # fmt: skip
    hist, _ = load_archive(ws)
    history = history_table(hist)
    conv = json.loads((gap.workspace_dir / "converted.json").read_text())
    provider = {p["id"]: {"country": p.get("country"),
                          "birth_date": date.fromisoformat(str(p["birth_date"])) if p.get("birth_date") else None}
                for p in conv["players"]}  # fmt: skip
    starts = {t["id"]: date.fromisoformat(str(t["start_date"])) for t in conv["tournaments"]}
    first_gap: dict = {}
    per_player = Counter()
    for m in conv["matches"]:
        for pid in (m["player1_id"], m["player2_id"]):
            s = starts[m["tournament_id"]]
            first_gap[pid] = min(first_gap.get(pid, s), s)
            per_player[pid] += 1
    existing = {}
    ov = Path(ws.overrides_path)
    if ov.exists():
        for e in tomllib.loads(ov.read_text()).get("mapping", []):
            existing[(e["provider"], str(e["provider_player_id"]))] = int(e["canonical_player_id"])
    queue = dry["review_queue"]
    amb_cands = {c["id"] for q in queue for c in q.get("candidates", [])}
    mappings = check_mappings(gprop["proposed"], archive, history, provider, first_gap, existing, amb_cands)

    # Ambiguous and new players: affected matches are the backfill records involving them.
    ambiguous = [propose_ambiguous(q, history, per_player[q["provider_player_id"]]) for q in queue]
    held = pl.read_parquet(d / "simulation" / "backfill" / "reconciliation_quarantine.parquet")

    # Simulation verification: tables, a fresh no-write re-run, and the recorded reconstructions.
    sim = d / "simulation" / "backfill"
    led = pl.read_parquet(sim / "ledger.parquet")
    summ = json.loads((sim / "backfill_summary.json").read_text())
    sws = replace(ws, prospective_dir=d / "simulation", reports_dir=d / "simulation" / "reports",
                  overrides_path=d / "identity_overrides.proposed.toml")  # fmt: skip
    rerun = run_backfill(sws, write=False)["summary"]
    recon = json.loads((d / "simulation_state_reconstruction.json").read_text())
    verification = {
        "provider_records": summ["provider_records"], "validated": led.height,
        "validated_by_level": dict(sorted(led.join(pl.read_parquet(sim / "matches.parquet").select(
            "match_id", "tournament_level"), on="match_id").group_by("tournament_level").len().iter_rows())),
        "elo_updates": int(led["is_update"].sum()),
        "not_elo_eligible": dict(sorted(led.filter(~pl.col("is_update")).group_by("exclusion_reason").len().iter_rows())),
        "held": held.height, "held_by_reason": dict(held.group_by("reason").len().iter_rows()),
        "records_accounted": led.height + held.height == summ["provider_records"],
        "service_statistics_updates": int(led["stats_eligible"].sum()),
        "rows_with_complete_statistics": int((led["stats_status"] == "complete").sum()),
        "recorded_snapshot": summ["state_snapshot_id"], "rerun_snapshot": rerun["state_snapshot_id"],
        "rerun_identical": json.loads(json.dumps(rerun, default=str)) == summ,
        "reconstruction_hashes": [r["state_table_hash"] for r in recon["runs"]],
        "reconstruction_rows": recon["runs"][0]["rows"], "reconstruction_deterministic": recon["deterministic"],
        "elo_feature_configuration_hashes": [recon["runs"][0]["elo_configuration_hash"],
                                             recon["runs"][0]["feature_configuration_hash"]]}  # fmt: skip

    gates = tomllib.loads((PROJECT_ROOT / "config" / "prospective.toml").read_text())["status"]
    return {"label": label, "as_of": as_of.isoformat(), "availability_policy_hash": pol.policy_hash(),
            "safeguard": safeguard, "events": events, "future_cases": FUTURE_CASES,
            "identity": {"proposed": len(mappings), "mappings": mappings}, "ambiguous": ambiguous,
            "_proposal": gprop,
            "licence": {"files": [{k: f[k] for k in ("original_name", "stored_as", "role", "sha256", "bytes")} | {"licence": f["licence"]}
                                  for f in man["files"]], "hashes_verified": licence.verify_hashes(d, man)},
            "verification": verification, "model": readiness.model_evidence(cfg),
            "production_state": pipeline.baseline_state(cfg), "gates": gates}  # fmt: skip


# --------------------------------------------------------------------------------------------------------- render
def _t(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |"
                                                                       for r in rows]  # fmt: skip


def render(p: dict) -> str:
    sg, v, m = p["safeguard"], p["verification"], p["identity"]["mappings"]
    groups = {
        k: [x for x in m if x["evidence_class"] == k] for k in ("overlap_corroborated", "name_and_birth_date_only")
    }
    flagc = {k: Counter(f for x in g for f in x["flags"]) for k, g in groups.items()}
    blocking = [x for x in m if x["blocking"]]
    ev_flags = Counter(f for e in p["events"] for f in e["flags"])
    amb = p["ambiguous"]
    L = [f"# Backfill authorization package ({p['label']})", "",
         f"As of {p['as_of']}. Generated by `python -m tennis_data.operations.authorization_review`. Read-only: no "
         "identity is approved, no state is changed, prospective predictions stay DISABLED and the backfill "
         "INCOMPLETE. The machine-auditable identity package is record-level and source-derived, so it is kept locally "
         "(`data/operations/exports/<label>/private/identity_review.json`); this report carries aggregates only.", "",
         "## 1. Chronology safeguard (proposed `state_availability_v1`, not active)", "",
         "The frozen weekly batching stays unchanged for the archive, M4A, Elo and every historical evaluation. For "
         "prospective forecasts, `config/availability.toml` (policy hash "
         f"`{p['availability_policy_hash']}`, `enabled = false`) defines: a result enters the state of a forecast "
         "created at *T* only if `max(frozen state_eligible_at, ingested_at, match date + "
         f"{sg['policy']['match_day_buffer_days']} days 00:00 UTC) <= T`; undated results wait until release batch + "
         f"{sg['policy']['undated_after_release_batch_days']} days; rows without an ingestion time are never used; "
         "an ingestion time before the match date excludes the row.", "",
         f"- The {sg['the_121']['rows']} results the frozen policy releases on or before their match day: under the "
         f"safeguard all {sg['the_121']['available_after_match_day_buffer']} wait at least until the match-day buffer "
         "has passed. From the match-day rule alone (whatever the ingestion time), they become available "
         "this many days after the frozen release: " + ", ".join(
             f"{k}: {n}" for k, n in sg["the_121"]["delay_days_vs_frozen"].items()) + " results. For the historical "
         "gap the actual ingestion time dominates: no backfilled result is usable by a forecast created before the "
         "production import.",
         f"- All {sg['rows']:,} rows: status " + ", ".join(f"{k} {v}" for k, v in sg["status"].items())
         + f"; never earlier than the frozen policy: **{sg['never_earlier_than_frozen']}**; rows still available on or "
         f"before their match date: **{sg['safeguard_played_on_or_after_available']}** "
         f"(frozen: {sg['frozen_played_on_or_after_release']}).",
         "- Usable by a forecast created at " + "; ".join(f"{k[:16]}: {n:,}" for k, n in sg["usable_at"].items())
         + f" (the simulation stamps ingestion at {p['as_of']}; the last finals, 2026-10-25, wait until "
         f"{sg['latest_available_at'][:10]}).",
         "- Without an ingestion time no row would be used (" + ", ".join(
             f"{k} {n}" for k, n in sg["without_ingestion_time"].items()) + "): the production import must record "
         "it, which Phase 4A does (`ingested_at`).",
         "- Remaining risks: provider timestamps are not verified against an external clock (the RFC 3161 checkpoint "
         "is not enabled); a wrong match date in a source shifts availability (match dates are cross-checked against "
         "two other sources only for outcomes); the safeguard is a proposal until the owner enables it together with "
         "the prospective path, which is outside Phase 5A.", "",
         "## 2. Release review of every new event (frozen batching, unchanged)", "",
         "Flags: " + ", ".join(f"`{k}` {n}" for k, n in sorted(ev_flags.items())) + ". Overrun = days of the event "
         "on or after the day its results enter state.", ""]  # fmt: skip
    L += _t(["start", "event", "lvl", "draw", "weeks", "batch", "release", "enters state", "last match", "overrun",
             "played on/after", "flags"],
            [[e["start"], e["tournament"], e["level"], e["draw_size"], e["weeks"], e["batch"], e["release_batch"],
              e["enters_state"], e["last_match"], e["overrun_days"], f"{e['played_on_or_after']}/{e['matches']}",
              ", ".join(e["flags"]) or "-"] for e in p["events"]])  # fmt: skip
    multi = [e for e in p["events"] if e["weeks"] > 1]
    dc = [e for e in p["events"] if e["level"] == "D"]
    L += ["", "Two-week events (" + ", ".join(f"{e['tournament']} overrun {e['overrun_days']} d" for e in multi)
          + ") release at the close of their second batch; a mid-week or Friday start still leaves the last days "
          f"after the release. Davis Cup: {len(dc)} one-week level D ties dated by their first rubber; "
          f"{sum(1 for e in dc if e['played_on_or_after'])} Friday-dated ties have Saturday rubbers released on their "
          "match day. No event's level, draw or week count is changed.", "",
          "**Unresolved exceptions** (the 27 matching overlap events do not prove these safe; the archive has no match "
          "dates to check them):", ""]  # fmt: skip
    L += [f"- {a}: {b}." for a, b in p["future_cases"]]
    L += ["", "With the safeguard, dated results of all these cases wait for their match day; undated ones wait 21 "
          "days after their release batch starts.", "",
          "## 3. Identity review (308 proposed high-confidence mappings)", "",
          "Phase 4A methods: " + ", ".join(f"`{k}` {n}" for k, n in sorted(Counter(x["method"] for x in m).items()))
          + " (`name_country`: exact name and country, no birth date on either side). Country codes are compared "
          "after mapping ISO 3166 codes to IOC codes (e.g. `SVN` = `SLO`). Blocking checks: "
          + ", ".join(f"`{h}`" for h in HARD) + f". Mappings with a blocking flag: **{len(blocking)}**.", ""]  # fmt: skip
    rows = []
    for k, g in groups.items():
        rows.append([k, len(g), sum(1 for x in g if x["blocking"]), flagc[k].get("same_name_other_birth_date", 0),
                     flagc[k].get("country_differs", 0), flagc[k].get("no_archive_history", 0),
                     flagc[k].get("archive_inactive_over_3_years", 0), flagc[k].get("unusual_age_at_gap_match", 0),
                     flagc[k].get("archive_history_before_age_14", 0)])  # fmt: skip
    L += _t(
        [
            "evidence class",
            "mappings",
            "blocking",
            "same name, other DOB",
            "country differs",
            "no archive history",
            "inactive > 3 y",
            "unusual age",
            "history before 14",
        ],
        rows,
    )
    soft = [x for x in m if set(x["flags"]) - {"same_name_other_birth_date"} and not x["blocking"]]
    L += ["", f"Mappings with a blocking flag: {len(blocking)} ("
          + ", ".join(f"`{k}` {n}" for k, n in sorted(Counter(f for x in blocking for f in x["blocking"]).items()))
          + f"); mappings needing a closer look (other non-blocking flags): {len(soft)}. Per-mapping records "
          "(evidence, history, every flag, `review_decision = PENDING`) are in the local identity package, as are "
          "the proposed overrides split by class.", "",
          "## 4. Ambiguous and new players", "",
          f"{sum(1 for a in amb if a['proposed_canonical_id'])} of {len(amb)} have a defensible proposal (owner review "
          f"required); {sum(1 for a in amb if not a['proposed_canonical_id'])} stay UNRESOLVED. Affected backfill "
          f"records: {sum(a['affected_matches'] for a in amb)} (the 99 held matches; a match can involve two of these "
          "players).", ""]  # fmt: skip
    L += _t(["method", "players", "defensible proposal", "unresolved", "affected records"],
            [[k, sum(1 for a in amb if a["method"] == k), sum(1 for a in amb if a["method"] == k and a["proposed_canonical_id"]),
              sum(1 for a in amb if a["method"] == k and not a["proposed_canonical_id"]),
              sum(a["affected_matches"] for a in amb if a["method"] == k)] for k in sorted({a["method"] for a in amb})])  # fmt: skip
    lic = p["licence"]
    L += ["", "## 5. Source licence status", ""]
    L += _t(["file", "role", "SHA-256 (verified)", "status", "research", "raw redistribution", "commercial"],
            [[f["original_name"], f["role"], f"`{f['sha256'][:16]}…` ({'ok' if lic['hashes_verified'].get(f['stored_as']) else 'MISMATCH'})",
              f["licence"]["status"], f["licence"].get("research_use", "-"), f["licence"].get("raw_redistribution", "-"),
              f["licence"].get("commercial_use", "-")] for f in lic["files"]])  # fmt: skip
    L += ["", f"All stored files match their recorded hashes: **{all(lic['hashes_verified'].values())}**. Attribution: "
          f"{next((f['licence']['attribution'] for f in lic['files'] if 'attribution' in f['licence']), '-')}. The "
          "cross-check files keep *not verified*; they are never imported.", "",
          "## 6. Expected backfill effects (SIMULATION, verified)", ""]  # fmt: skip
    L += _t(["item", "value"], [
        ["provider records", f"{v['provider_records']:,}"],
        ["validated new main-tour matches", f"**{v['validated']:,}** ("
         + ", ".join(f"{k} {n}" for k, n in v["validated_by_level"].items()) + ")"],
        ["eligible Elo updates", f"**{v['elo_updates']:,}**"],
        ["not Elo-eligible (frozen policy)", ", ".join(f"{k} {n}" for k, n in v["not_elo_eligible"].items())],
        ["held", f"**{v['held']}** (" + ", ".join(f"{k} {n}" for k, n in v["held_by_reason"].items()) + ")"],
        ["all records accounted for", v["records_accounted"]],
        ["compatible service-statistics updates", f"**{v['service_statistics_updates']}** (rows with complete "
         f"statistics {v['rows_with_complete_statistics']})"],
        ["simulated snapshot: recorded / fresh re-run", f"`{v['recorded_snapshot']}` / `{v['rerun_snapshot']}` "
         f"(identical summary: {v['rerun_identical']})"],
        ["full reconstruction (2 runs)", ", ".join(f"`{h}`" for h in v["reconstruction_hashes"])
         + f" over {v['reconstruction_rows']:,} matches; deterministic {v['reconstruction_deterministic']}"],
        ["Elo / feature configuration hashes", " / ".join(f"`{h}`" for h in v["elo_feature_configuration_hashes"])],
    ])  # fmt: skip
    mo, ps = p["model"], p["production_state"]
    L += ["", "## 7. Frozen model integrity and gates", "",
          f"- M4A frozen hash `{mo['frozen_hash']}` (expected `{mo['expected_frozen_hash']}`), configuration hash "
          f"`{mo['configuration_hash']}` (expected `{mo['expected_configuration_hash']}`): match **{mo['hashes_match']}**; "
          f"recorded replay passed {mo['replay_passed']}.",
          f"- Production state unchanged: `{ps['state_snapshot_id']}` (match component `{ps['matches_component']}`).",
          f"- Gates: backfill **{p['gates']['backfill'] if p['gates'] else '?'}**, prospective predictions "
          f"**{p['gates']['prospective_predictions'] if p['gates'] else '?'}**. No external publication is enabled.", ""]  # fmt: skip
    return "\n".join(L)


def main(argv=None) -> int:
    from tennis_data.operations.config import load_operations_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--operations-config")
    a = ap.parse_args(argv)
    cfg = load_operations_config(a.operations_config)
    p = build(cfg, a.label, date.fromisoformat(a.as_of))
    gprop = p.pop("_proposal")
    d = exports.export_dir(cfg, a.label)
    (d / "authorization_review.json").write_text(json.dumps(p, indent=1, sort_keys=True, default=str) + "\n")
    for cls in ("overlap_corroborated", "name_and_birth_date_only"):
        sel = {x["provider_player_id"] for x in p["identity"]["mappings"] if x["evidence_class"] == cls}
        prop = {"proposed": [r for r in gprop["proposed"] if r["provider_player_id"] in sel]}
        backfill_plan.write_proposal(d / f"identity_overrides.proposed.{cls}.toml", prop)
    out = Path(cfg.reports_dir)
    priv = exports.private_dir(cfg, a.label)
    priv.mkdir(parents=True, exist_ok=True)
    pkg = {"label": p["label"], "as_of": p["as_of"], "format": "identity-review-1",
           "note": "review_decision is PENDING for every entry; nothing here is applied",
           "mappings": p["identity"]["mappings"], "ambiguous": p["ambiguous"]}  # fmt: skip
    (priv / "identity_review.json").write_text(json.dumps(pkg, indent=1, sort_keys=True, default=str) + "\n")
    (out / f"backfill_authorization_{a.label}.md").write_text(render(p) + "\n")
    print(json.dumps({"mappings": len(p["identity"]["mappings"]), "ambiguous": len(p["ambiguous"]),
                      "verification": p["verification"]}, indent=1, default=str))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

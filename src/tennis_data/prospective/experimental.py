"""Experimental single-match predictions for personal research. Never prospective evidence.

This module reuses the Phase 4A path unchanged:

- the verified active state snapshot (``load_state_snapshot``, which checks the restored state's content hash against
  the recorded one);
- the hash-checked frozen production artifact (``load_production``);
- an outcome-free pending row whose Elo and serve/return state the unchanged engines freeze at the start of its weekly
  batch (``compute_state``);
- the frozen model matrix and probabilities (``derive``, ``apply_model``);
- the freshness checks (``assess``).

It adds three things:

1. **Name resolution** against the archive player master. A query must match exactly one player with history in the
   active state, otherwise the request is refused.
2. **Cutoff enforcement.** The prediction cutoff is the start of the match's weekly batch. The request is refused if
   the frozen production fit was trained on results released at or after the cutoff, or if the state holds any result
   released at or after it.
3. **Explicit labelling.** Every result is EXPERIMENTAL. It is also STALE DATA when any freshness check fails.

It never reads or changes the status gate, never writes to the ledger and writes no files. ``prediction_mode`` is
``experimental``, which is not a mode the ledger or the prospective evaluation accepts.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from tennis_data.elo.chronology import add_batches
from tennis_data.operations.controlled_backfill import restore, verify_snapshot
from tennis_data.prospective.backfill import load_state_snapshot
from tennis_data.prospective.config import ProspectiveConfig
from tennis_data.prospective.freshness import assess, state_coverage
from tennis_data.prospective.identity import load_archive_players, name_tokens, norm
from tennis_data.prospective.inference import apply_model, derive, load_production
from tennis_data.prospective.state import PENDING_PREFIX, StateSnapshot, UpcomingMatch, compute_state, make_snapshot

MODE = "experimental"
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
NOTICE = ("EXPERIMENTAL: personal research output from historical state. This is not an independently verified "
          "prospective forecast and is not recorded in the prediction ledger.")  # fmt: skip
STALE_NOTICE = "STALE DATA: at least one freshness check fails; the state may not reflect the players' current form."
# Pending rows never update state, so level and round only place the row in the tour-level rating pool; neither is a
# model input. The frozen M4A uses no tournament-format input (best-of-three vs best-of-five), so none is accepted.
LEVEL, ROUND = "A", "R32"
STATE_VERSIONS = ("v0", "v1", "v2")


class ExperimentalRefused(ValueError):
    """The request cannot be answered without fake features or information from after the cutoff."""


def resolve_player(query: str, players: pl.DataFrame, known_ids: set[int]) -> dict:
    """One archive player with history in the state: an ATP player ID, a full name (any order) or a unique surname."""
    pool = players.filter(pl.col("player_id").is_in(list(known_ids)))
    q = query.strip()
    if q.isdigit():
        hit = pool.filter(pl.col("player_id") == int(q))
        if not hit.height:
            raise ExperimentalRefused(f"player ID {q} has no results in the active state")
        return hit.row(0, named=True)
    toks = frozenset(norm(q).split())
    if not toks:
        raise ExperimentalRefused("empty player name")
    rows = list(pool.iter_rows(named=True))
    hits = [r for r in rows if name_tokens(r["first_name"], r["last_name"]) == toks]
    if not hits:
        hits = [r for r in rows if norm(r["last_name"]) == norm(q)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ExperimentalRefused(f"no player with results in the active state matches {query!r}")
    listed = ", ".join(f"{r['first_name']} {r['last_name']} ({r['country'] or '?'}, ID {r['player_id']})"
                       for r in hits[:8])  # fmt: skip
    raise ExperimentalRefused(f"{query!r} is ambiguous: {listed}. Use the full name or the player ID")


def snapshot_store(cfg: ProspectiveConfig) -> Path:
    """The Phase 5A content-addressed snapshot store next to the Phase 4A directory (read only here)."""
    return cfg.prospective_dir.parent / "operations" / "snapshots"


def _state_version(cfg: ProspectiveConfig, snapshot_id: str) -> dict:
    reg = snapshot_store(cfg) / "registry.json"
    if reg.exists():
        for s in json.loads(reg.read_text())["snapshots"]:
            if s["state_snapshot_id"] == snapshot_id:
                return {"version": s["version"], "name": s["name"], "active": s["active_phase4a_directory"],
                        "source": "active Phase 4A directory"}  # fmt: skip
    return {"version": None, "name": "active Phase 4A state (not in a snapshot registry)", "active": True,
            "source": "active Phase 4A directory"}  # fmt: skip


def _manifest(store: Path, name: str) -> dict:
    path = store / "manifests" / f"{name}.json"
    if not path.exists():
        raise ExperimentalRefused(f"snapshot manifest {name} is not available locally")
    man = json.loads(path.read_text())
    bad = verify_snapshot(store, man)  # every stored object's SHA-256 and the manifest's tree hash
    if bad:
        raise ExperimentalRefused(f"snapshot {name} failed verification: {bad[:5]}")
    return man


def load_versioned_state(cfg: ProspectiveConfig, version: str) -> tuple[StateSnapshot, dict, pl.DataFrame | None, int]:
    """Verify a registered snapshot and restore it read-only into a temporary directory.

    Checks, in order:

    1. **Registry entry.** The version is registered, and its name, state ID and tree hash agree with the manifest.
    2. **Manifest.** Every object's SHA-256 matches, and so does the tree hash.
    3. **Parent provenance** (versions after 0). The parent is registered, its manifest verifies, it is the version-0
       baseline, and it is built on the same archive commit as this configuration.
    4. **State ID.** After restoration, ``load_state_snapshot`` recomputes the state's content hash and compares it
       with the recorded one, which must also equal the manifest's and the registry's state ID.

    Nothing in the store or the active directory is written. Returns the snapshot, its provenance, the snapshot's own
    new-player registry (for name resolution) and its count of unresolved identities.
    """
    if version not in STATE_VERSIONS:
        raise ExperimentalRefused(f"state version must be one of {STATE_VERSIONS}")
    store = snapshot_store(cfg)
    reg_path = store / "registry.json"
    if not reg_path.exists():
        raise ExperimentalRefused(f"no snapshot registry at {reg_path}; state {version} is not available")
    reg = {s["version"]: s for s in json.loads(reg_path.read_text())["snapshots"]}
    n = int(version[1:])
    if n not in reg:
        raise ExperimentalRefused(f"state {version} is not registered")
    entry = reg[n]
    man = _manifest(store, entry["name"])
    if (man["name"], man["state_snapshot_id"], man["tree_hash"]) != (entry["name"], entry["state_snapshot_id"],
                                                                    entry["tree_hash"]):  # fmt: skip
        raise ExperimentalRefused(f"registry entry and manifest of {version} disagree")
    provenance = {"version": n, "name": entry["name"], "active": entry["active_phase4a_directory"],
                  "source": "verified snapshot store (restored read-only)", "tree_hash": man["tree_hash"],
                  "role": man.get("role")}  # fmt: skip
    if n > 0:
        parent = man.get("parent")
        if not parent or entry.get("parent") != parent:
            raise ExperimentalRefused(f"{version} has no consistent parent in the registry and manifest")
        pentry = next((s for s in reg.values() if s["name"] == parent), None)
        if pentry is None or pentry["version"] != 0:
            raise ExperimentalRefused(f"parent {parent} of {version} is not the registered version-0 baseline")
        pman = _manifest(store, parent)
        if pman["tree_hash"] != pentry["tree_hash"] or pman["state_snapshot_id"] != pentry["state_snapshot_id"]:
            raise ExperimentalRefused(f"parent {parent} does not match its registry entry")
        if pman.get("state_components", {}).get("archive_commit") != cfg.data.source.commit:
            raise ExperimentalRefused("the parent baseline was built on another archive commit")
        provenance |= {"parent": parent, "parent_state_snapshot_id": pman["state_snapshot_id"]}
    with tempfile.TemporaryDirectory(prefix="tennis-experimental-") as tmp:
        dest = Path(tmp) / "prospective"
        restore(store, man, dest)  # re-checks every restored file against the manifest
        try:
            snap = load_state_snapshot(replace(cfg, prospective_dir=dest))
        except (AssertionError, OSError, ValueError, pl.exceptions.PolarsError) as e:
            raise ExperimentalRefused(f"state {version} could not be reconstructed and verified: {e}") from e
        if snap.snapshot_id != man["state_snapshot_id"]:
            raise ExperimentalRefused(f"restored state {snap.snapshot_id} != recorded {man['state_snapshot_id']}")
        reg_new = dest / "identity" / "new_players.parquet"
        new_players = pl.read_parquet(reg_new) if reg_new.exists() else None
        dec = dest / "identity" / "decisions.parquet"
        unresolved = pl.read_parquet(dec).filter(~pl.col("usable")).height if dec.exists() else 0
    return snap, provenance, new_players, unresolved


def prediction_batch(cfg: ProspectiveConfig, event: date) -> date:
    """The weekly batch of an event starting on ``event`` under the frozen chronology."""
    one = pl.DataFrame({"tournament_date": [event], "tournament_level": [LEVEL], "draw_size": [32]},
                       schema_overrides={"draw_size": pl.Int64})  # fmt: skip
    return add_batches(one, cfg.elo.chronology)["prediction_batch"][0]


def point_in_time(snapshot: StateSnapshot, batch: date, cfg: ProspectiveConfig) -> tuple[StateSnapshot, int]:
    """The snapshot restricted to results released before ``batch``: an earlier point-in-time state.

    The engines already give a pending row only the state released before its batch, so this changes no prediction.
    It makes the guarantee explicit, so that a later, more complete snapshot cannot carry information past the cutoff.
    """
    rel = add_batches(snapshot.matches, cfg.elo.chronology)["release_batch"]
    keep = rel < batch
    removed = int((~keep).sum())
    if not removed:
        return snapshot, 0
    kept = snapshot.matches.filter(keep)
    comp = {"derived_from": snapshot.snapshot_id, "point_in_time_before": str(batch)}
    return make_snapshot(kept, snapshot.quarantine_ids, comp), removed


def predict_match(cfg: ProspectiveConfig, player_a: str, player_b: str, surface: str, match_date: date, *,
                  event_start: date | None = None, state_version: str | None = None, now: datetime | None = None,
                  snapshot=None) -> dict:  # fmt: skip
    """Experimental M4A probability for one match. Raises ``ExperimentalRefused`` instead of guessing.

    ``state_version`` None uses the active Phase 4A directory (version 0). ``v0``/``v1``/``v2`` select a registered
    snapshot, verified and restored read-only; selecting one never activates it.
    """
    if surface not in SURFACES:
        raise ExperimentalRefused(f"surface must be one of {SURFACES}")
    event = event_start or match_date
    if event > match_date:
        raise ExperimentalRefused("the event start date is after the match date")
    if (match_date - event).days > 21:
        raise ExperimentalRefused("the match date is more than three weeks after the event start")
    now = now or datetime.now(UTC)
    art = load_production(cfg)  # checksums and the frozen hash are verified before any prediction
    batch = prediction_batch(cfg, event)
    training_cutoff = date.fromisoformat(art.training_cutoff)
    if batch < training_cutoff:
        raise ExperimentalRefused(
            f"prediction cutoff {batch} precedes the production fit's training cutoff {training_cutoff}: its "
            "coefficients were estimated on results released after the cutoff. Historical matches have walk-forward "
            "predictions in the dashboard's match explorer"
        )
    new_players, unresolved = None, None
    if snapshot is not None:
        provenance = {**_state_version(cfg, snapshot.snapshot_id), "source": "supplied snapshot"}
    elif state_version is None:
        try:
            snapshot = load_state_snapshot(cfg)  # recomputes the state's content hash against the recorded one
        except (AssertionError, OSError, ValueError, pl.exceptions.PolarsError) as e:
            raise ExperimentalRefused(
                f"no verified state is available (build the local research state first): {e}"
            ) from e
        provenance = _state_version(cfg, snapshot.snapshot_id)
    else:
        snapshot, provenance, new_players, unresolved = load_versioned_state(cfg, state_version)
    if unresolved is None:
        dec_path = cfg.prospective_dir / "identity" / "decisions.parquet"
        unresolved = pl.read_parquet(dec_path).filter(~pl.col("usable")).height if dec_path.exists() else 0
    selected_id = snapshot.snapshot_id
    snapshot, removed = point_in_time(snapshot, batch, cfg)
    known = set(snapshot.matches["player_a_id"].to_list()) | set(snapshot.matches["player_b_id"].to_list())
    players = load_archive_players(cfg)
    if new_players is not None and new_players.height:
        players = pl.concat([players, new_players.select(
            pl.col("canonical_player_id").alias("player_id"), "first_name", "last_name", "country", "birth_date")],
            how="vertical_relaxed").unique("player_id", keep="first", maintain_order=True)  # fmt: skip
    a, b = resolve_player(player_a, players, known), resolve_player(player_b, players, known)
    if a["player_id"] == b["player_id"]:
        raise ExperimentalRefused("player A and player B are the same player")
    fx = UpcomingMatch("experimental", a["player_id"], b["player_id"], "experimental", event, LEVEL, surface, ROUND)
    st = compute_state(snapshot, [fx], cfg, keep_history=True)
    pend = st["pending"]
    if pend["prediction_batch"][0] != batch:
        raise ExperimentalRefused("the engine placed the match in another batch than the cutoff check")
    hist = st["frame"].filter(~pl.col("match_id").str.starts_with(PENDING_PREFIX))
    late = hist.filter(pl.col("release_batch") >= batch)
    if late.height:  # cannot happen after point_in_time; kept as a guard
        raise ExperimentalRefused(f"the state holds {late.height} results released at or after the cutoff {batch}")
    cover = state_coverage(hist)
    fresh = assess(pend, cover, cfg, unresolved_ids=set(), unresolved_count=unresolved, now=now)["experimental"]
    x, m = derive(pend, art, cfg)
    pa, pb = apply_model(x, art)
    r = pend.row(0, named=True)
    checks = fresh["checks"]
    elo_status, stats_status = checks["last_eligible_result"]["status"], checks["last_point_statistics"]["status"]
    mixed = elo_status != "fail" and stats_status == "fail"
    labels = ["EXPERIMENTAL"] + (["STALE DATA"] if fresh["status"] == "fail" else [])
    labels += ["MIXED FRESHNESS"] if mixed else []
    labels += [f"INACTIVE STATE v{provenance['version']}"] if provenance.get("active") is False else []
    labels += ["POINT-IN-TIME RECONSTRUCTION"] if removed else []
    warnings = [f"{k}: {v['status']}" for k, v in checks.items() if v["status"] != "ok"]
    if mixed:
        warnings.append(f"mixed freshness: results (Elo) enter the state until {checks['last_eligible_result']['value']}"
                        f", but point statistics only until {checks['last_point_statistics']['value']}; the "
                        "serve/return inputs are older than the Elo input")  # fmt: skip
    if provenance.get("active") is False:
        warnings.append(f"state v{provenance['version']} is an inactive snapshot ({provenance.get('role')}); used "
                        "read-only, not activated. Matches of players with unresolved identities were held out "
                        "before import")  # fmt: skip
    end = cfg.historical_snapshot_end
    post = snapshot.matches.filter(pl.col("tournament_date") > end)
    post_counts = {s: post.filter((pl.col("player_a_id") == p["player_id"]) | (pl.col("player_b_id") == p["player_id"]))
                   .height for s, p in (("a", a), ("b", b))}  # fmt: skip
    if post.height:
        weeks = (post["tournament_date"].max() - post["tournament_date"].min()).days // 7 + 1
        warnings.append(f"backfill after the archive end ({end}) is incomplete: {post.height} results over {weeks} "
                        f"weeks; {a['first_name']} {a['last_name']} has {post_counts['a']}, {b['first_name']} "
                        f"{b['last_name']} {post_counts['b']}. A missing result cannot be told apart from inactivity")  # fmt: skip
    if removed:
        warnings.append(f"{removed} results released at or after the cutoff {batch} were removed: point-in-time "
                        f"state {snapshot.snapshot_id} derived from {selected_id}")  # fmt: skip
    if unresolved:
        warnings.append(f"{unresolved} source identities in this state are unresolved; their matches are not in it")
    filled = [n for n in art.inputs if m[f"filled_{n}"][0]]
    if filled:
        warnings.append(f"inputs set to the frozen neutral value 0: {', '.join(filled)}")

    def player(p, s):
        return {"player_id": p["player_id"], "name": f"{p['first_name']} {p['last_name']}", "country": p["country"],
                "probability": float(pa[0] if s == "a" else pb[0]),
                "surface_elo_probability": float(r["predicted_probability_a"] if s == "a"
                                                 else 1 - r["predicted_probability_a"]),
                "global_elo": r[f"player_{s}_elo_before"], "global_elo_matches": r[f"global_match_count_{s}"],
                "surface_elo_effective": r[f"surface_effective_elo_{s}"],
                "surface_elo_matches": r[f"surface_match_count_{s}"],
                "has_statistics_history": r[f"{s}_has_stats_history"],
                "weeks_since_last_statistics": r[f"{s}_weeks_since_last_stats"],
                "service_points_observed": r[f"{s}_service_points_observed"],
                "service_points_won_52w_decay": r[f"{s}_spw_decay52w"],
                "return_points_won_52w_decay": r[f"{s}_rpw_decay52w"],
                "results_after_archive_end": post_counts[s]}  # fmt: skip

    return {
        "labels": labels, "notice": NOTICE, "stale_notice": STALE_NOTICE if "STALE DATA" in labels else None,
        "prediction_mode": MODE, "surface": surface, "match_date": str(match_date), "event_start": str(event),
        "prediction_cutoff_batch_start": str(batch),
        "data_cutoff": {
            "last_results_entered_state": str(cover["last_update_release_batch"] + timedelta(days=7)),
            "last_point_statistics_entered_state": str(cover["last_stats_release_batch"] + timedelta(days=7)),
            "latest_event_start_in_state": str(snapshot.last_tournament_date),
        },
        "state": {**provenance, "selected_state_snapshot_id": selected_id, "state_snapshot_id": snapshot.snapshot_id,
                  "results_removed_for_cutoff": removed, "components": snapshot.components},
        "elo_freshness": checks["last_eligible_result"], "statistics_freshness": checks["last_point_statistics"],
        "model": {"model_id": art.model_id, "frozen_hash": art.frozen_hash[:16], "artifact": art.source,
                  "artifact_sha256": art.artifact_sha256, "training_cutoff": art.training_cutoff,
                  "calibration": art.calibrator.get("method")},
        "player_a": player(a, "a"), "player_b": player(b, "b"),
        "missing_inputs_set_to_neutral": filled, "freshness": fresh, "warnings": warnings,
        "official_gate": {"prospective_predictions": cfg.prospective_predictions, "backfill": cfg.backfill_status},
        "ledger_written": False, "generated_at": now.isoformat(),
    }  # fmt: skip


def render(res: dict) -> str:
    a, b = res["player_a"], res["player_b"]
    pct = lambda v: f"{100 * v:5.1f}%"  # noqa: E731
    lines = [f"[{' / '.join(res['labels'])}]", res["notice"]]
    if res["stale_notice"]:
        lines.append(res["stale_notice"])
    lines += [
        "",
        f"{a['name']} ({a['country']}) vs {b['name']} ({b['country']}), {res['surface']}, match date "
        f"{res['match_date']} (event start {res['event_start']})",
        "",
        f"{'':28}{'M4A (frozen)':>14}{'surface Elo':>14}",
        f"{a['name']:28}{pct(a['probability']):>14}{pct(a['surface_elo_probability']):>14}",
        f"{b['name']:28}{pct(b['probability']):>14}{pct(b['surface_elo_probability']):>14}",
        "",
        f"prediction cutoff (start of weekly batch): {res['prediction_cutoff_batch_start']}",
        f"data cutoff: last results entered the state {res['data_cutoff']['last_results_entered_state']}, last point "
        f"statistics {res['data_cutoff']['last_point_statistics_entered_state']} (latest event start in state "
        f"{res['data_cutoff']['latest_event_start_in_state']})",
        f"state: version {res['state']['version']} {res['state']['name']} from {res['state']['source']} "
        f"(selected {res['state']['selected_state_snapshot_id']}, used {res['state']['state_snapshot_id']}, "
        f"{'active' if res['state']['active'] else 'INACTIVE, not activated'})",
        f"Elo freshness: {res['elo_freshness']['status']} (results until {res['elo_freshness']['value']}); "
        f"serve/return statistics freshness: {res['statistics_freshness']['status']} "
        f"(until {res['statistics_freshness']['value']})",
        f"model: {res['model']['model_id']} frozen {res['model']['frozen_hash']}, {res['model']['artifact']} fit "
        f"(training cutoff {res['model']['training_cutoff']}, calibration {res['model']['calibration']})",
        "",
        f"freshness: {res['freshness']['status'].upper()}",
    ]  # fmt: skip
    for k, v in res["freshness"]["checks"].items():
        detail = {x: y for x, y in v.items() if x != "status"}
        lines.append(f"  {v['status']:8} {k}: {json.dumps(detail, default=str)}")
    lines.append("")
    for p in (a, b):
        lines.append(f"{p['name']}: global Elo {p['global_elo']:.0f} ({p['global_elo_matches']} matches), "
                     f"{res['surface']} Elo {p['surface_elo_effective']:.0f} ({p['surface_elo_matches']} matches), "
                     f"statistics history {'yes' if p['has_statistics_history'] else 'NO'}, "
                     f"weeks since last statistics {p['weeks_since_last_statistics']}")  # fmt: skip
    miss = res["missing_inputs_set_to_neutral"]
    lines.append(f"missing model inputs set to the frozen neutral value 0: {', '.join(miss) if miss else 'none'}")
    lines += ["", "warnings:"] + [f"  - {w}" for w in res["warnings"]] if res["warnings"] else ["", "warnings: none"]
    lines.append(f"official gate unchanged: prospective predictions {res['official_gate']['prospective_predictions']}, "
                 f"backfill {res['official_gate']['backfill']}; nothing written to the ledger")  # fmt: skip
    return "\n".join(lines)


__all__ = ["ExperimentalRefused", "predict_match", "render", "resolve_player"]

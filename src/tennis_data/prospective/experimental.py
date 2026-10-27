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
from datetime import UTC, date, datetime, timedelta

import polars as pl

from tennis_data.prospective.backfill import load_state_snapshot
from tennis_data.prospective.config import ProspectiveConfig
from tennis_data.prospective.freshness import assess, state_coverage
from tennis_data.prospective.identity import load_archive_players, name_tokens, norm
from tennis_data.prospective.inference import apply_model, derive, load_production
from tennis_data.prospective.state import PENDING_PREFIX, UpcomingMatch, compute_state

MODE = "experimental"
SURFACES = ("Hard", "Clay", "Grass", "Carpet")
NOTICE = ("EXPERIMENTAL: personal research output from historical state. This is not an independently verified "
          "prospective forecast and is not recorded in the prediction ledger.")  # fmt: skip
STALE_NOTICE = "STALE DATA: at least one freshness check fails; the state may not reflect the players' current form."
# Pending rows never update state, so level and round only place the row in the tour-level rating pool; neither is a
# model input. The frozen M4A uses no tournament-format input (best-of-three vs best-of-five), so none is accepted.
LEVEL, ROUND = "A", "R32"


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


def _state_version(cfg: ProspectiveConfig, snapshot_id: str) -> dict:
    reg = cfg.prospective_dir.parent / "operations" / "snapshots" / "registry.json"
    if reg.exists():
        for s in json.loads(reg.read_text())["snapshots"]:
            if s["state_snapshot_id"] == snapshot_id:
                return {"version": s["version"], "name": s["name"], "active": s["active_phase4a_directory"]}
    return {"version": None, "name": "active Phase 4A state (not in a snapshot registry)", "active": True}


def predict_match(cfg: ProspectiveConfig, player_a: str, player_b: str, surface: str, match_date: date, *,
                  event_start: date | None = None, now: datetime | None = None, snapshot=None) -> dict:  # fmt: skip
    """Experimental M4A probability for one match. Raises ``ExperimentalRefused`` instead of guessing."""
    if surface not in SURFACES:
        raise ExperimentalRefused(f"surface must be one of {SURFACES}")
    event = event_start or match_date
    if event > match_date:
        raise ExperimentalRefused("the event start date is after the match date")
    if (match_date - event).days > 21:
        raise ExperimentalRefused("the match date is more than three weeks after the event start")
    now = now or datetime.now(UTC)
    if snapshot is None:
        try:
            snapshot = load_state_snapshot(cfg)  # recomputes the state's content hash against the recorded one
        except (AssertionError, OSError, ValueError, pl.exceptions.PolarsError) as e:
            raise ExperimentalRefused(
                f"no verified state is available (build the local research state first): {e}"
            ) from e
    known = set(snapshot.matches["player_a_id"].to_list()) | set(snapshot.matches["player_b_id"].to_list())
    players = load_archive_players(cfg)
    a, b = resolve_player(player_a, players, known), resolve_player(player_b, players, known)
    if a["player_id"] == b["player_id"]:
        raise ExperimentalRefused("player A and player B are the same player")
    art = load_production(cfg)  # checksums and the frozen hash are verified before any prediction
    fx = UpcomingMatch("experimental", a["player_id"], b["player_id"], "experimental", event, LEVEL, surface, ROUND)
    st = compute_state(snapshot, [fx], cfg, keep_history=True)
    pend = st["pending"]
    batch = pend["prediction_batch"][0]
    training_cutoff = date.fromisoformat(art.training_cutoff)
    if batch < training_cutoff:
        raise ExperimentalRefused(
            f"prediction cutoff {batch} precedes the production fit's training cutoff {training_cutoff}: its "
            "coefficients were estimated on results released after the cutoff. Historical matches have walk-forward "
            "predictions in the dashboard's match explorer"
        )
    hist = st["frame"].filter(~pl.col("match_id").str.starts_with(PENDING_PREFIX))
    late = hist.filter(pl.col("release_batch") >= batch)
    if late.height:
        raise ExperimentalRefused(f"the state holds {late.height} results released at or after the cutoff {batch}")
    cover = state_coverage(hist)
    dec_path = cfg.prospective_dir / "identity" / "decisions.parquet"
    unresolved = pl.read_parquet(dec_path).filter(~pl.col("usable")).height if dec_path.exists() else 0
    fresh = assess(pend, cover, cfg, unresolved_ids=set(), unresolved_count=unresolved, now=now)["experimental"]
    x, m = derive(pend, art, cfg)
    pa, pb = apply_model(x, art)
    r = pend.row(0, named=True)
    labels = ["EXPERIMENTAL"] + (["STALE DATA"] if fresh["status"] == "fail" else [])

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
                "return_points_won_52w_decay": r[f"{s}_rpw_decay52w"]}  # fmt: skip

    return {
        "labels": labels, "notice": NOTICE, "stale_notice": STALE_NOTICE if "STALE DATA" in labels else None,
        "prediction_mode": MODE, "surface": surface, "match_date": str(match_date), "event_start": str(event),
        "prediction_cutoff_batch_start": str(batch),
        "data_cutoff": {
            "last_results_entered_state": str(cover["last_update_release_batch"] + timedelta(days=7)),
            "last_point_statistics_entered_state": str(cover["last_stats_release_batch"] + timedelta(days=7)),
            "latest_event_start_in_state": str(snapshot.last_tournament_date),
        },
        "state": {"state_snapshot_id": snapshot.snapshot_id, **_state_version(cfg, snapshot.snapshot_id),
                  "components": snapshot.components},
        "model": {"model_id": art.model_id, "frozen_hash": art.frozen_hash[:16], "artifact": art.source,
                  "artifact_sha256": art.artifact_sha256, "training_cutoff": art.training_cutoff,
                  "calibration": art.calibrator.get("method")},
        "player_a": player(a, "a"), "player_b": player(b, "b"),
        "missing_inputs_set_to_neutral": [n for n in art.inputs if m[f"filled_{n}"][0]],
        "freshness": fresh,
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
        f"state: {res['state']['name']} (version {res['state']['version']}, snapshot {res['state']['state_snapshot_id']})",
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
    lines.append(f"official gate unchanged: prospective predictions {res['official_gate']['prospective_predictions']}, "
                 f"backfill {res['official_gate']['backfill']}; nothing written to the ledger")  # fmt: skip
    return "\n".join(lines)


__all__ = ["ExperimentalRefused", "predict_match", "render", "resolve_player"]

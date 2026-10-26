"""Phase 4A run: gap audit -> backfill from the raw store -> state snapshot -> replay -> readiness -> reports."""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.prospective.backfill import load_archive, load_state_snapshot, run_backfill
from tennis_data.prospective.config import SCHEMA_VERSION, STATE_ENGINE_VERSION, ProspectiveConfig
from tennis_data.prospective.freshness import assess, state_coverage
from tennis_data.prospective.gap import audit
from tennis_data.prospective.inference import ArtifactIntegrityError, load_frozen, load_production
from tennis_data.prospective.replay import run_replay
from tennis_data.prospective.state import UpcomingMatch

CODE_FILES = ("config", "schema", "identity", "reconciliation", "backfill", "state", "inference", "freshness",
              "replay", "gap", "pipeline", "report", "providers/raw_store", "providers/http", "providers/fixture",
              "providers/sackmann_csv", "providers/sportradar")  # fmt: skip


def code_hash(cfg: ProspectiveConfig) -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in CODE_FILES:
        p = here / f"{name}.py"
        h.update(name.encode() + b"\0" + (p.read_bytes() if p.exists() else b"") + b"\0")
    h.update(cfg.config_path.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def probe_fixture(cfg: ProspectiveConfig, snapshot) -> UpcomingMatch:
    """A diagnostic match on the first batch after the prospective start between the two players with the most
    recent results. It is used only to measure state freshness; no probability is reported for it."""
    m = snapshot.matches.filter(pl.col("tournament_date") == snapshot.last_tournament_date).sort("match_id")
    r = m.row(0, named=True)
    d = cfg.prospective_start + timedelta(days=1)
    while d.weekday() != 0:  # a Monday: tournaments start early in the week
        d += timedelta(days=1)
    return UpcomingMatch("probe:freshness", r["player_a_id"], r["player_b_id"], "probe", d, "A", "Hard", "R32", 32, 3)


def integrity(cfg: ProspectiveConfig) -> dict:
    try:
        frozen = load_frozen(cfg)
        prod = load_production(cfg)
        return {"passed": True, "frozen_hash": frozen["frozen_hash"], "model_id": prod.model_id,
                "production_artifact_sha256": prod.artifact_sha256, "inputs": list(prod.inputs),
                "training_cutoff": prod.training_cutoff, "calibration": prod.calibrator.get("method"),
                "modeling_configuration_hash": cfg.modeling.configuration_hash()}  # fmt: skip
    except ArtifactIntegrityError as e:
        return {"passed": False, "error": str(e)}


def readiness(cfg, audit_res, backfill, replay, integ, fresh) -> dict:
    blockers = []
    if not integ["passed"]:
        blockers.append(f"frozen artifact integrity: {integ.get('error')}")
    if not replay["passed"]:
        blockers.append("historical replay did not reproduce the published predictions")
    mp = audit_res["missing_period"]
    accepted_gap = sum(r["len"] for r in backfill["by_period"] if r["period"] == "backfill_gap")
    if accepted_gap == 0:
        blockers.append(f"historical gap not filled: {mp['weeks']} weekly batches ({mp['first_missing_batch']} to "
                        f"{mp['last_batch_before_prospective']}) have no results; no authorized source was accessible")  # fmt: skip
    if fresh and fresh["status"] == "fail":
        failing = [k for k, v in fresh["checks"].items() if v["status"] == "fail"]
        blockers.append(f"feature freshness fails for a match after the prospective start: {', '.join(failing)}")
    blockers.append("no verified live provider: schedules for upcoming matches cannot be ingested automatically")
    if cfg.prospective_predictions != "ENABLED":
        blockers.append(f"status gate: backfill {cfg.backfill_status}, prospective predictions "
                        f"{cfg.prospective_predictions} (config/prospective.toml [status])")  # fmt: skip
    components = {
        "frozen model integrity": integ["passed"],
        "historical replay (inference path = published Phase 3B)": replay["passed"],
        "state reconstruction (Phase 2 + 3A outputs reproduced)": replay["state_reconstruction"]["features_identical"]
        and replay["state_reconstruction"]["elo_max_abs_error"] <= cfg.replay_tolerance,
        "historical backfill complete to the prospective start": accepted_gap > 0 and fresh["status"] != "fail",
        "live schedule and result provider verified": False,
        f"status gate: backfill {cfg.backfill_status}, prospective predictions {cfg.prospective_predictions}":
        cfg.prospective_predictions == "ENABLED",
    }  # fmt: skip
    return {"ready_for_prospective_predictions": not blockers, "components": components, "blockers": blockers}


def run_phase4a(cfg: ProspectiveConfig, write_reports: bool = True) -> dict:
    t0 = time.perf_counter()
    timings = {}
    hist, qids = load_archive(cfg)
    qpath = cfg.data.paths.quarantine_dir / "matches.parquet"
    quarantine = pl.read_parquet(qpath) if qpath.exists() else None
    t = time.perf_counter()
    audit_res = audit(cfg, hist, qids, quarantine)
    timings["audit"] = time.perf_counter() - t
    t = time.perf_counter()
    bf = run_backfill(cfg)
    timings["backfill"] = time.perf_counter() - t
    snapshot = load_state_snapshot(cfg)
    integ = integrity(cfg)
    probe = probe_fixture(cfg, snapshot)
    t = time.perf_counter()
    rp = run_replay(snapshot, cfg, extra=[probe])  # one engine run serves the replay and the freshness probe
    timings["replay"] = time.perf_counter() - t
    # Freshness probe (no probability is reported for it).
    t = time.perf_counter()
    st = rp["state"]
    cover = state_coverage(st["frame"].filter(~pl.col("match_id").str.starts_with("pending:")))
    store = cfg.prospective_dir / "raw"
    latest = None
    if store.exists():
        from tennis_data.prospective.providers.raw_store import RawStore

        recs = RawStore(store).records()
        latest = max((datetime.fromisoformat(r.retrieved_at) for r in recs), default=None)
    unresolved = bf["decisions"].filter(~pl.col("usable"))
    fresh_all = assess(st["pending"].filter(pl.col("match_id") == "pending:probe:freshness"), cover, cfg, unresolved_ids=set(), unresolved_count=unresolved.height,
                       source_retrieved_at=latest, now=datetime.combine(probe.tournament_date, datetime.min.time(), UTC))  # fmt: skip
    fresh = fresh_all["probe:freshness"]
    timings["freshness"] = time.perf_counter() - t
    ready = readiness(cfg, audit_res, bf["summary"], rp["summary"], integ, fresh)
    commit = git_commit()
    manifest = {
        "schema_version": SCHEMA_VERSION, "state_engine_version": STATE_ENGINE_VERSION,
        "prospective_configuration_hash": cfg.configuration_hash(), "code_hash": code_hash(cfg),
        "modeling_configuration_hash": cfg.modeling.configuration_hash(),
        "feature_configuration_hash": cfg.features.configuration_hash(),
        "elo_configuration_hash": st["elo_configuration_hash"], "frozen_hash": integ.get("frozen_hash"),
        "archive_commit": cfg.data.source.commit, "raw_snapshot_id": bf["summary"]["raw_snapshot_id"],
        "state_snapshot_id": snapshot.snapshot_id, "state_components": snapshot.components,
        "matches_in_state": snapshot.matches.height, "backfill": bf["summary"], "replay": rp["summary"] | {
            "seconds_state": None, "seconds_total": None},
        "integrity": integ, "freshness_probe": {"fixture": probe.as_dict(), **fresh}, "readiness": ready,
    }  # fmt: skip
    out = cfg.prospective_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase4a_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    rp["predictions"].drop("generated_at").sort("match_id").write_parquet(out / "replay_predictions.parquet")
    timings["total"] = time.perf_counter() - t0
    perf = {"seconds": {k: round(v, 1) for k, v in timings.items()},
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "git_commit": commit, "replay_cases": rp["summary"]["cases"],
            "replay_state_seconds": rp["summary"]["seconds_state"], "run_at": datetime.now(UTC).isoformat()}  # fmt: skip
    (out / "performance.json").write_text(json.dumps(perf, indent=2, sort_keys=True) + "\n")
    res = {"audit": audit_res, "backfill": bf, "replay": rp, "integrity": integ, "freshness": fresh,
           "probe": probe, "readiness": ready, "manifest": manifest, "performance": perf, "cfg": cfg,
           "snapshot": snapshot}  # fmt: skip
    if write_reports:
        from tennis_data.prospective.report import write_reports as _w

        res["reports"] = _w(res)
    return res


__all__ = ["run_phase4a", "date"]


class PredictionBlocked(RuntimeError):
    """The request cannot be served as a prospective prediction (stale state, unresolved identity, timing)."""


def predict_upcoming(cfg: ProspectiveConfig, fixtures: list[UpcomingMatch], mode: str = "prospective",
                     now: datetime | None = None) -> pl.DataFrame:  # fmt: skip
    """Price upcoming matches with the frozen production model.

    ``prospective`` requires: the reviewed status gate ``[status].prospective_predictions = ENABLED`` (DISABLED in
    Phase 4A); every match starts after ``prospective_start``; the prediction is generated before the
    match (verified start time, else 00:00 UTC of the event start date); and freshness is not ``fail``. Otherwise the
    request is refused. Other modes are labelled accordingly and never count as prospective evidence. Phase 4A does
    not persist predictions (the immutable ledger is Phase 4B).
    """
    from tennis_data.prospective.inference import predict_state

    now = now or datetime.now(UTC)
    if mode == "prospective":
        check_enabled(cfg)
    snapshot = load_state_snapshot(cfg)
    if mode == "prospective":
        check_timing(cfg, fixtures, snapshot.matches, now)
    art = load_production(cfg)
    st = compute_state_for(snapshot, fixtures, cfg)
    cover = state_coverage(st["frame"].filter(~pl.col("match_id").str.starts_with("pending:")))
    dec_path = cfg.prospective_dir / "identity" / "decisions.parquet"
    unresolved = pl.read_parquet(dec_path).filter(~pl.col("usable")) if dec_path.exists() else None
    from tennis_data.prospective.providers.raw_store import RawStore

    recs = RawStore(cfg.prospective_dir / "raw").records() if (cfg.prospective_dir / "raw").exists() else []
    latest = max((datetime.fromisoformat(r.retrieved_at) for r in recs), default=None)
    fresh = assess(st["pending"], cover, cfg, unresolved_ids=set(),
                   unresolved_count=0 if unresolved is None else unresolved.height, source_retrieved_at=latest, now=now)  # fmt: skip
    if mode == "prospective":
        check_freshness(fresh)
    return predict_state(st["pending"], art, cfg, prediction_mode=mode, snapshot_id=snapshot.snapshot_id,
                         components=snapshot.components, freshness=fresh, generated_at=now)  # fmt: skip


def check_enabled(cfg: ProspectiveConfig) -> None:
    """The reviewed status gate in config/prospective.toml ([status])."""
    if cfg.prospective_predictions != "ENABLED":
        raise PredictionBlocked(
            f"prospective predictions are {cfg.prospective_predictions} (backfill {cfg.backfill_status}); "
            "see reports/prospective/inference_readiness.md"
        )


def check_timing(cfg: ProspectiveConfig, fixtures: list[UpcomingMatch], history: pl.DataFrame, now: datetime) -> None:
    for f in fixtures:
        start = (datetime.fromisoformat(f.scheduled_start_at) if f.scheduled_start_at
                 else datetime.combine(f.tournament_date, datetime.min.time(), UTC))  # fmt: skip
        if f.tournament_date <= cfg.prospective_start:
            raise PredictionBlocked(f"{f.match_id}: starts before the prospective period")
        if now >= start:
            raise PredictionBlocked(f"{f.match_id}: generated after the match start; not a prospective forecast")
    future = history.filter(pl.col("tournament_date") > min(f.tournament_date for f in fixtures))
    if future.height:
        raise PredictionBlocked("state contains results from after the predicted matches (contamination risk)")


def check_freshness(fresh: dict) -> None:
    bad = sorted(k for k, v in fresh.items() if v["status"] == "fail")
    if bad:
        raise PredictionBlocked(f"freshness fails for {bad}; see the freshness diagnostics")


def compute_state_for(snapshot, fixtures, cfg):
    from tennis_data.prospective.state import compute_state

    return compute_state(snapshot, fixtures, cfg, keep_history=True)

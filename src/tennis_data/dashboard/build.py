"""Build the public dashboard dataset from the local research outputs (run once by a maintainer, not by the app).

Every input is derived from Jeff Sackmann's ATP archive (CC BY-NC-SA 4.0) through the repository's own Phase 1-3B
pipeline. Nothing from owner-supplied or other licensed sources (TennisMyLife exports, the Phase 5A backfill,
private identity approvals or local snapshots) is read. The output is redistributed under the same licence, with
attribution, in ``dashboard/data/``:

- ``predictions.parquet``: every historical M4A prediction of the frozen specification (walk-forward annual refits,
  model id ``M4A+raw``) with the paired Phase 2 surface-Elo probability, the actual result, and match context;
- ``snapshots.parquet``: one pre-match row per player per tour-level match since 1991, point-in-time (the state
  entering the match's weekly batch): global and surface Elo and the serve/return indicators of Phase 3A;
- ``players.parquet``: names and career span of the players in the snapshots;
- ``status.json``: aggregate operational facts (archive period, backfill counts, snapshot versions, freshness);
- ``manifest.json``: sources, hashes, row counts and SHA-256 of every file.

``python -m tennis_data.dashboard.build`` (requires the local Phase 1-4A outputs under ``data/``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from tennis_data.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "dashboard" / "data"
SNAPSHOT_START = 1991
LICENCE = ("CC BY-NC-SA 4.0. Derived from Jeff Sackmann's ATP data (https://github.com/JeffSackmann/tennis_atp) "
           "through this repository's pipeline; changes: validation, Elo ratings, serve/return features and model "
           "predictions computed by the project. Non-commercial use only; share alike.")  # fmt: skip
SNAP_FEATURES = {"has_stats_history": "has_stats_history", "stat_matches": "stat_matches",
                 "weeks_since_last_stats": "weeks_since_last_stats", "service_points_observed": "service_points",
                 "spw_shrunk": "spw_shrunk", "rpw_shrunk": "rpw_shrunk", "spw_prior_weight": "spw_prior_weight",
                 "rpw_prior_weight": "rpw_prior_weight", "service_prior_dominated": "service_prior_dominated",
                 "surface_spw": "surface_spw", "surface_rpw": "surface_rpw", "spw_decay52w": "spw_decay52w",
                 "rpw_decay52w": "rpw_decay52w", "service_ess_decay52w": "service_ess_decay52w"}  # fmt: skip


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _validated(root: Path) -> pl.DataFrame:
    files = sorted((root / "data" / "validated" / "matches").rglob("*.parquet"))
    cols = ["match_id", "tournament_name", "round", "score", "player_a_id", "player_b_id", "player_a_name", "player_b_name", "player_a_ioc",
            "player_b_ioc", "player_a_hand", "player_b_hand", "season", "source_commit"]  # fmt: skip
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def _tree_hash(d: Path) -> dict:
    files = sorted(p for p in d.rglob("*.parquet"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(d).as_posix().encode() + b"\0" + _sha(p).encode() + b"\0")
    return {"files": len(files), "sha256": h.hexdigest()}


def _input_hashes(root: Path) -> dict:
    """Exact inputs (SHA-256), so every published value can be traced to the research outputs that produced it."""
    d = root / "data"
    return {"modeling/predictions/selected.parquet": _sha(d / "modeling" / "predictions" / "selected.parquet"),
            "elo/predictions/model_name=surface_elo/part-0.parquet":
                _sha(d / "elo" / "predictions" / "model_name=surface_elo" / "part-0.parquet"),
            "features/serve_return": _tree_hash(d / "features" / "serve_return"),
            "validated/matches": _tree_hash(d / "validated" / "matches")}  # fmt: skip


def build_predictions(root: Path, matches: pl.DataFrame) -> pl.DataFrame:
    sel = pl.read_parquet(root / "data" / "modeling" / "predictions" / "selected.parquet")
    sel = sel.filter(pl.col("model_id") == "M4A+raw")
    elo = pl.read_parquet(root / "data" / "elo" / "predictions" / "model_name=surface_elo" / "part-0.parquet",
                          columns=["match_id", "tournament_date"])  # fmt: skip
    out = (sel.join(elo, on="match_id", how="left", validate="1:1")
           .join(matches.select("match_id", "season", "tournament_name", "round", "score", "player_a_name",
                                "player_b_name"), on="match_id", how="left", validate="1:1")
           .select("match_id", "season", "tournament_date", "prediction_batch", "release_batch", "tournament_name",
                   "tournament_level", "surface", "round", "player_a_id", "player_a_name", "player_b_id",
                   "player_b_name", "score", pl.col("predicted_probability_a").alias("p_m4a"),
                   pl.col("baseline_elo_probability_a").alias("p_elo"), pl.col("target_a_wins").alias("a_won"),
                   "phase", "training_cutoff")
           .sort("prediction_batch", "match_id"))  # fmt: skip
    if out["player_a_name"].null_count() or out["tournament_date"].null_count():
        raise ValueError("predictions without archive context")
    return out


def build_snapshots(root: Path, matches: pl.DataFrame) -> pl.DataFrame:
    elo = pl.read_parquet(root / "data" / "elo" / "predictions" / "model_name=surface_elo" / "part-0.parquet").filter(
        pl.col("tournament_date").dt.year() >= SNAPSHOT_START
    )
    feats = pl.concat([pl.read_parquet(f) for f in sorted((root / "data" / "features" / "serve_return").rglob(
        "*.parquet"))], how="vertical_relaxed")  # fmt: skip
    fcols = ["match_id"] + [f"{s}_{c}" for s in "ab" for c in SNAP_FEATURES]
    f = feats.select([c for c in fcols if c in feats.columns])
    df = elo.join(f, on="match_id", how="left")
    sides = []
    for me, op in (("a", "b"), ("b", "a")):
        won = pl.col("target_a_wins") if me == "a" else ~pl.col("target_a_wins")
        sides.append(df.select(
            pl.col(f"player_{me}_id").alias("player_id"), pl.col(f"player_{op}_id").alias("opponent_id"),
            "match_id", "tournament_date", "prediction_batch", "tournament_level", "surface", "match_status",
            won.alias("won"), pl.col(f"player_{me}_elo_before" if me == "a" else "player_b_elo_before").alias("elo_global"),
            pl.col(f"surface_elo_{me}_before").alias("elo_surface"),
            pl.col(f"surface_effective_elo_{me}").alias("elo_surface_effective"),
            pl.col(f"global_match_count_{me}").alias("global_matches"),
            pl.col(f"surface_match_count_{me}").alias("surface_matches"),
            *[pl.col(f"{me}_{c}").alias(n) for c, n in SNAP_FEATURES.items() if f"{me}_{c}" in df.columns]))  # fmt: skip
    snap = pl.concat(sides).sort("player_id", "tournament_date", "match_id")
    # Display precision: ratings to 0.1 point, rates to 1e-4; counts stay exact.
    snap = snap.with_columns(
        *[pl.col(c).round(1).cast(pl.Float32) for c in ("elo_global", "elo_surface", "elo_surface_effective")],
        *[pl.col(c).round(4).cast(pl.Float32) for c in ("spw_shrunk", "rpw_shrunk", "spw_prior_weight",
                                                         "rpw_prior_weight", "surface_spw", "surface_rpw",
                                                         "spw_decay52w", "rpw_decay52w", "service_ess_decay52w")
          if c in snap.columns])  # fmt: skip
    return snap.join(matches.select("match_id", "tournament_name", "round"), on="match_id", how="left")


def build_players(matches: pl.DataFrame, snap: pl.DataFrame) -> pl.DataFrame:
    sides = [matches.select(pl.col(f"player_{s}_id").alias("player_id"), pl.col(f"player_{s}_name").alias("name"),
                            pl.col(f"player_{s}_ioc").alias("ioc"), pl.col(f"player_{s}_hand").alias("hand"),
                            "season") for s in "ab"]  # fmt: skip
    names = (
        pl.concat(sides)
        .drop_nulls("name")
        .sort("season")
        .group_by("player_id")
        .agg(pl.col("name").last(), pl.col("ioc").last(), pl.col("hand").last())
    )
    span = snap.group_by("player_id").agg(pl.col("tournament_date").min().alias("first_match"),
                                          pl.col("tournament_date").max().alias("last_match"),
                                          pl.len().alias("matches"))  # fmt: skip
    return span.join(names, on="player_id", how="left").sort("player_id")


def build_status(root: Path) -> dict:
    """Aggregate operational facts only (no record-level or licensed-source data)."""
    ops = root / "data" / "operations"
    status: dict = {"generated_from": "local Phase 4A manifest, snapshot registry and Phase 5A run records (counts)"}
    man = json.loads((root / "data" / "prospective" / "phase4a_manifest.json").read_text())
    fr = man["freshness_probe"]["checks"]
    status["freshness_probe"] = {k: {"status": v["status"], "value": v.get("value")} for k, v in fr.items()}
    status["active_state_snapshot_id"] = man["state_snapshot_id"]
    status["archive_commit"] = man["archive_commit"]
    reg = ops / "snapshots" / "registry.json"
    if reg.exists():
        status["snapshots"] = [{"version": s["version"], "state_snapshot_id": s["state_snapshot_id"],
                                "active": s["active_phase4a_directory"]}
                               for s in json.loads(reg.read_text())["snapshots"]]  # fmt: skip
    runs = []
    for name, version in (("owner-2026-10-26", 1), ("owner-2026-10-26-v2", 2)):
        p = ops / "controlled_backfill" / name / "run_record.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        b = r["phase4a"]["backfill"]
        runs.append({"version": version, "accepted_matches": b["accepted_records"],
                     "elo_updates": b["state_eligible_updates"],
                     "quarantined_matches": len(r["held_before_import"]),
                     "unresolved_identities": len(r["identity"]["pending"]) + len(r["identity"]["rejected"]),
                     "service_statistics_rows": b["with_complete_stats"],
                     "verification_passed": r.get("verification", {}).get("passed"),
                     "staged_readiness_blockers": r["phase4a"]["readiness"]["blockers"],
                     "staged_freshness": {k: v["status"] for k, v in r["phase4a"]["freshness"]["checks"].items()}})  # fmt: skip
    status["backfill_runs"] = runs
    return status


def main(argv=None) -> int:
    root = PROJECT_ROOT
    matches = _validated(root)
    commits = matches["source_commit"].unique().to_list()
    OUT.mkdir(parents=True, exist_ok=True)
    preds = build_predictions(root, matches)
    snap = build_snapshots(root, matches)
    players = build_players(matches, snap)
    files = {"predictions.parquet": preds, "snapshots.parquet": snap, "players.parquet": players}
    for name, df in files.items():
        df.write_parquet(OUT / name, compression="zstd", compression_level=19, statistics=False)
    (OUT / "status.json").write_text(json.dumps(build_status(root), indent=1, sort_keys=True, default=str) + "\n")
    summary = json.loads((root / "reports" / "modeling" / "modeling_summary.json").read_text())
    tests = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=root, capture_output=True,
                           text=True).stdout.strip().splitlines()  # fmt: skip
    # with the project's "-q" default, --collect-only -q prints "path: N" per test file
    n_tests = sum(
        int(x.rsplit(":", 1)[1]) for x in tests if x.startswith("tests/") and x.rsplit(":", 1)[1].strip().isdigit()
    )
    elo_cfg = tomllib.loads((root / "config" / "elo.toml").read_text())
    manifest = {
        "format": "dashboard-data-1", "built_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "licence": LICENCE, "source": {"archive_commits": commits, "phase": "Phase 1-3B outputs (archive only)"},
        "model": {"id": summary["selected_model"], "frozen_hash": summary["frozen_hash"][:16],
                  "configuration_hash": summary["configuration_hash"], "version": summary["model_version"],
                  "historical_predictions": "walk-forward annual refits of the frozen specification (model id "
                                            "M4A+raw); the production fit has no out-of-sample history"},
        "elo_week_start": elo_cfg["chronology"]["week_start"], "snapshot_start_season": SNAPSHOT_START,
        "tests_collected_at_build": n_tests,
        "files": {n: {"rows": (df.height if (df := files.get(n)) is not None else None), "sha256": _sha(OUT / n)}
                  for n in [*files, "status.json"]},
        "coverage": {"predictions": [str(preds["tournament_date"].min()), str(preds["tournament_date"].max())],
                     "snapshots": [str(snap["tournament_date"].min()), str(snap["tournament_date"].max())]},
        "built_from_date": str(date.today()),
        "inputs": _input_hashes(root),
    }  # fmt: skip
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("files", "coverage", "tests_collected_at_build")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

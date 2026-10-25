"""Player-state reconstruction with the unchanged Phase 2 and Phase 3A engines.

A *state snapshot* is a validated match table (pinned archive plus accepted backfill rows) with its quarantine IDs.
The state is a pure function of that table and the frozen configurations, so it is identified by a content hash and
"restoring a checkpoint" means rebuilding from the stored table, which reproduces the state exactly.

Upcoming matches are appended as *pending* rows: targets (``is_target``) that never update Elo (``is_update``) or
serve/return statistics (``stats_eligible``) and carry no outcome. The engines then give each pending row exactly the
pre-match state frozen at the start of its weekly batch, the same mechanism every historical prediction used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date

import polars as pl

from tennis_data.elo.chronology import add_batches
from tennis_data.elo.config import EloParams
from tennis_data.elo.eligibility import in_pool
from tennis_data.elo.engine import run_elo, surface_key, surface_probability
from tennis_data.features.observations import STAT_INPUT_COLUMNS, feature_frame, observations
from tennis_data.features.pipeline import assemble
from tennis_data.features.snapshots import build_snapshots
from tennis_data.normalization.schema import ROUND_ORDER
from tennis_data.prospective.config import STATE_ENGINE_VERSION, ProspectiveConfig

PENDING_PREFIX = "pending:"


class StateError(ValueError):
    pass


def selected_surface_params(cfg: ProspectiveConfig) -> tuple[EloParams, str]:
    """The Phase 2 surface-Elo parameters as published, verified against their configuration hash."""
    summary = json.loads((cfg.elo.reports_dir / "elo_summary.json").read_text())["selected"]["surface_elo"]
    params = EloParams(**summary["params"])
    h = cfg.elo.with_params(params).configuration_hash()
    if h != summary["configuration_hash"]:
        raise StateError(f"Phase 2 surface-Elo hash mismatch: {h} != {summary['configuration_hash']}")
    return params, h


@dataclass(frozen=True)
class StateSnapshot:
    matches: pl.DataFrame  # STAT_INPUT_COLUMNS, validated rows only
    quarantine_ids: pl.Series
    components: dict  # name -> content hash
    snapshot_id: str

    @property
    def last_tournament_date(self) -> date:
        return self.matches["tournament_date"].max()


def table_hash(df: pl.DataFrame) -> str:
    """Order-independent content hash of a table (rows sorted by match_id, then hashed as JSON lines)."""
    h = hashlib.sha256()
    for r in df.sort("match_id").iter_rows():
        h.update(json.dumps(r, default=str).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def make_snapshot(matches: pl.DataFrame, quarantine_ids: pl.Series, components: dict | None = None) -> StateSnapshot:
    matches = matches.select(list(STAT_INPUT_COLUMNS))
    if matches["match_id"].n_unique() != matches.height:
        raise StateError("duplicate match_id in state snapshot")
    comp = dict(sorted({**(components or {}), "matches": table_hash(matches)}.items()))
    sid = hashlib.sha256(json.dumps([STATE_ENGINE_VERSION, comp], sort_keys=True).encode()).hexdigest()[:16]
    return StateSnapshot(matches, quarantine_ids, comp, sid)


@dataclass(frozen=True)
class UpcomingMatch:
    """A normalized upcoming match (canonical player IDs). ``scheduled_start_at`` only if verified by a source."""

    match_id: str
    player_a_id: int
    player_b_id: int
    tournament_id: str
    tournament_date: date  # event start date (the date the frozen chronology uses)
    tournament_level: str
    surface: str | None
    round: str
    draw_size: int | None = None
    best_of: int | None = None
    scheduled_start_at: str | None = None  # ISO 8601 UTC, verified source time only

    def as_dict(self):
        return asdict(self)


def pending_frame(fixtures: list[UpcomingMatch], cfg: ProspectiveConfig, columns: list[str]) -> pl.DataFrame:
    """Pending rows with the same columns as ``feature_frame`` output; batches from the frozen chronology."""
    if not fixtures:
        return pl.DataFrame()
    ids = [f.match_id for f in fixtures]
    if len(set(ids)) != len(ids):
        raise StateError("duplicate upcoming match ids")
    for f in fixtures:
        if f.player_a_id == f.player_b_id:
            raise StateError(f"{f.match_id}: identical players")
        if f.round not in ROUND_ORDER:
            raise StateError(f"{f.match_id}: unknown round {f.round!r}")
        if f.surface not in (None, "Hard", "Clay", "Grass", "Carpet"):
            raise StateError(f"{f.match_id}: unknown surface {f.surface!r}")
    keys = {s: surface_key(s, cfg.elo.params.carpet_policy) for s in ("Hard", "Clay", "Grass", "Carpet")}
    df = pl.DataFrame(
        {
            "match_id": [PENDING_PREFIX + f.match_id for f in fixtures],
            "tournament_id": [f.tournament_id for f in fixtures],
            "tournament_date": [f.tournament_date for f in fixtures],
            "season": [f.tournament_date.year for f in fixtures],
            "surface": [f.surface for f in fixtures],
            "tournament_level": [f.tournament_level for f in fixtures],
            "draw_size": [f.draw_size for f in fixtures],
            "round": [f.round for f in fixtures],
            "round_order": [ROUND_ORDER[f.round] for f in fixtures],
            "player_a_id": [f.player_a_id for f in fixtures],
            "player_b_id": [f.player_b_id for f in fixtures],
            "source_family": [cfg.backfill_family] * len(fixtures),
        },
        schema_overrides={
            "season": pl.Int32,
            "round_order": pl.Int32,
            "draw_size": pl.Int64,
            "player_a_id": pl.Int64,
            "player_b_id": pl.Int64,
        },  # fmt: skip
    )
    df = add_batches(df.with_columns(in_pool(cfg.elo).alias("pool")), cfg.elo.chronology).with_columns(
        pl.col("surface").replace_strict(keys, default=None, return_dtype=pl.String).alias("surface_key"),
        pl.lit(True).alias("is_target"), pl.lit(False).alias("is_update"), pl.lit(False).alias("stats_eligible"),
        pl.lit(None, dtype=pl.Boolean).alias("target_a_wins"), pl.lit("pending").alias("match_status"),
        pl.lit("pending_upcoming_match").alias("stats_exclusion_reason"),
    )  # fmt: skip
    if df.filter(pl.col("pool").is_null()).height:
        bad = df.filter(pl.col("pool").is_null())["match_id"].to_list()
        raise StateError(f"upcoming matches outside the rating pools (level/family): {bad}")
    return df.select([pl.col(c) if c in df.columns else pl.lit(None).alias(c) for c in columns])


def compute_state(snapshot: StateSnapshot, fixtures: list[UpcomingMatch], cfg: ProspectiveConfig,
                  keep_history: bool = False) -> dict:  # fmt: skip
    """Run Phase 2 Elo and Phase 3A features over the snapshot plus pending rows.

    Returns pending-row state (Elo inputs + every feature column) and, with ``keep_history``, the historical targets.
    """
    params, elo_hash = selected_surface_params(cfg)
    fcfg = cfg.features
    hist = feature_frame(snapshot.matches, fcfg, snapshot.quarantine_ids)
    pend = pending_frame(fixtures, cfg, hist.columns)
    frame = pl.concat([hist, pend.cast(hist.schema, strict=False)], how="vertical") if pend.height else hist
    run = run_elo(frame, replace(params))
    elo = surface_probability(run.predictions, params.shrinkage_matches).select(
        "match_id", "player_a_elo_before", "player_b_elo_before", "global_match_count_a", "global_match_count_b",
        "surface_elo_a_before", "surface_elo_b_before", "surface_match_count_a", "surface_match_count_b",
        "surface_effective_elo_a", "surface_effective_elo_b",
        pl.col("surface_probability_a").alias("predicted_probability_a"),
    )  # fmt: skip
    snap = build_snapshots(frame, observations(frame), fcfg)
    feats = assemble(snap.features, fcfg)
    state = feats.join(elo, on="match_id", how="left", validate="1:1")
    meta = frame.select("match_id", "release_batch", "tournament_id")
    state = state.join(meta, on="match_id", how="left", validate="1:1")
    pending = state.filter(pl.col("match_id").str.starts_with(PENDING_PREFIX))
    if pending.height != len(fixtures):
        raise StateError("some upcoming matches received no state")
    out = {"pending": pending.sort("match_id"), "elo_configuration_hash": elo_hash, "fit_log": snap.fit_log,
           "feature_configuration_hash": fcfg.configuration_hash()}  # fmt: skip
    if keep_history:
        out["history"] = state.filter(~pl.col("match_id").str.starts_with(PENDING_PREFIX))
        out["frame"] = frame
    return out

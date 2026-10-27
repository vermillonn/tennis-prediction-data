"""Data access and analysis for the research dashboard (no Streamlit or Plotly imports; tested in CI).

Sources, in order of authority:

1. **Published reports** (``reports/``, tracked): the headline metrics, bootstrap intervals, calibration and subgroup
   tables of Phases 2-3B. The dashboard shows these numbers as published.
2. **Public dashboard dataset** (``dashboard/data/``, tracked, CC BY-NC-SA 4.0): the historical M4A predictions and
   per-player pre-match snapshots built by ``tennis_data.dashboard.build`` from the archive-only research outputs.
   Interactive views recompute metrics on subsets with the project's own scorer (``modeling.evaluation.summarize``);
   on the full published samples the recomputation reproduces the reports (tested).
3. **Configuration** (``config/``): the operational gates, read at run time.

Every loader returns ``None`` when its files are absent, so the application degrades to the published aggregates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.modeling.evaluation import calibration, row_losses, summarize

LEVELS = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 250/500", "F": "Tour Finals", "D": "Davis Cup",
          "O": "Olympics"}  # fmt: skip
PHASES = {"development": "Development 2002–2009 (model selection)",
          "exploratory": "Exploratory 2010–2026 (seen in Phase 2)"}  # fmt: skip
MODELS = {"p_m4a": "M4A", "p_elo": "Surface Elo"}
ALLOWED_FILES = {"predictions.parquet", "snapshots.parquet", "players.parquet", "status.json", "manifest.json",
                 "README.md"}  # fmt: skip


def data_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_DATA_DIR", PROJECT_ROOT / "dashboard" / "data"))


def reports_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_REPORTS_DIR", PROJECT_ROOT / "reports"))


def config_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_CONFIG_DIR", PROJECT_ROOT / "config"))


def calibration_bins() -> int:
    try:
        return int(tomllib.loads((config_dir() / "modeling.toml").read_text())["evaluation"]["calibration_bins"])
    except (OSError, KeyError, ValueError):
        return 10


# ------------------------------------------------------------------------------------------------------- loading
def _read(name: str):
    p = data_dir() / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else pl.read_parquet(p)


def load_manifest() -> dict | None:
    return _read("manifest.json")


def load_predictions() -> pl.DataFrame | None:
    return _read("predictions.parquet")


def load_snapshots() -> pl.DataFrame | None:
    return _read("snapshots.parquet")


def load_players() -> pl.DataFrame | None:
    return _read("players.parquet")


def load_status() -> dict | None:
    return _read("status.json")


def verify_manifest(manifest: dict | None = None) -> dict[str, bool]:
    """SHA-256 of every dataset file against the manifest (False when a file is missing or altered)."""
    manifest = manifest or load_manifest()
    if not manifest:
        return {}
    out = {}
    for name, f in manifest["files"].items():
        p = data_dir() / name
        out[name] = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
    return out


def published() -> dict:
    """The published Phase 1-3B aggregates used as-is (empty sections when a report is missing)."""
    r = reports_dir()

    def js(path):
        p = r / path
        return json.loads(p.read_text()) if p.exists() else None

    def csv(path):
        p = r / path
        return pl.read_csv(p) if p.exists() else None

    return {"modeling": js("modeling/modeling_summary.json"), "build": js("build_summary.json"),
            "features": js("features/feature_summary.json"), "elo": js("elo/elo_summary.json"),
            "calibration": {"development": csv("modeling/tables/calibration_development.csv"),
                            "exploratory": csv("modeling/tables/calibration_exploratory.csv")},
            "subgroups": {"development": csv("modeling/tables/development_subgroups.csv"),
                          "exploratory": csv("modeling/tables/exploratory_subgroups.csv")},
            "periods": csv("modeling/tables/exploratory_periods.csv"),
            "folds": csv("modeling/tables/development_folds.csv")}  # fmt: skip


def headline(pub: dict) -> list[dict]:
    """Published M4A (``M4A+raw``) versus surface Elo (``M0``) per evaluation period."""
    m = pub.get("modeling")
    if not m:
        return []
    out = []
    for phase in ("development", "exploratory"):
        block = m[phase]
        a, b = block["metrics"]["M4A+raw"], block["metrics"]["M0"]
        c = block["comparisons_vs_M0"]["M4A+raw vs M0"]
        out.append({"phase": phase, "matches": a["n"], "weeks": block["weeks"],
                    "m4a": {k: a[k] for k in ("log_loss", "brier", "accuracy", "ece")},
                    "elo": {k: b[k] for k in ("log_loss", "brier", "accuracy", "ece")},
                    "d_log_loss": c["log_loss_diff"], "ll_ci95": c["ll_ci95"], "d_brier": c["brier_diff"],
                    "brier_ci95": c["brier_ci95"], "bootstrap": {"samples": c["samples"], "clusters": c["clusters"]}})  # fmt: skip
    return out


@dataclass(frozen=True)
class Gates:
    backfill: str
    prospective_predictions: str
    availability_policy_enabled: bool
    external_publishers_enabled: list[str]

    @property
    def live_predictions_available(self) -> bool:
        return self.prospective_predictions == "ENABLED" and self.backfill == "COMPLETE"


def gates() -> Gates:
    """Operational gates read from the reviewed configuration (closed when a file is unreadable)."""
    c = config_dir()
    try:
        st = tomllib.loads((c / "prospective.toml").read_text())["status"]
    except (OSError, KeyError):
        st = {}
    try:
        av = bool(tomllib.loads((c / "availability.toml").read_text())["policy"]["enabled"])
    except (OSError, KeyError):
        av = False
    try:
        pubs = tomllib.loads((c / "evaluation.toml").read_text()).get("publishers", {})
        ext = sorted(k for k, v in pubs.items() if k != "local_fixture" and v.get("enabled"))
    except OSError:
        ext = []
    return Gates(st.get("backfill", "UNKNOWN"), st.get("prospective_predictions", "DISABLED"), av, ext)


# --------------------------------------------------------------------------------------------------- predictions
def filter_predictions(df: pl.DataFrame, *, player_ids: list[int] | None = None, opponent_id: int | None = None,
                       surfaces: list[str] | None = None, years: tuple[int, int] | None = None,
                       levels: list[str] | None = None, phases: list[str] | None = None) -> pl.DataFrame:  # fmt: skip
    """Subset of the historical predictions. Players match either side; ``opponent_id`` requires both players."""
    e = pl.lit(True)
    if player_ids:
        in_a, in_b = pl.col("player_a_id").is_in(player_ids), pl.col("player_b_id").is_in(player_ids)
        e &= in_a | in_b
        if opponent_id is not None:
            e &= (in_a & (pl.col("player_b_id") == opponent_id)) | (in_b & (pl.col("player_a_id") == opponent_id))
    if surfaces:
        e &= pl.col("surface").is_in(surfaces)
    if years:
        e &= pl.col("season").is_between(years[0], years[1])
    if levels:
        e &= pl.col("tournament_level").is_in(levels)
    if phases:
        e &= pl.col("phase").is_in(phases)
    return df.filter(e)


def orient(df: pl.DataFrame, player_id: int) -> pl.DataFrame:
    """Rows from the point of view of ``player_id`` (probabilities and result flipped when the player is B).

    The models are symmetric in player order, so P(player wins) = 1 - P(opponent wins) exactly."""
    a = pl.col("player_a_id") == player_id
    d = df.filter(a | (pl.col("player_b_id") == player_id))
    return d.with_columns(
        pl.when(a).then(pl.col("player_a_name")).otherwise(pl.col("player_b_name")).alias("player"),
        pl.when(a).then(pl.col("player_b_name")).otherwise(pl.col("player_a_name")).alias("opponent"),
        pl.when(a).then(pl.col("p_m4a")).otherwise(1 - pl.col("p_m4a")).alias("p_m4a_player"),
        pl.when(a).then(pl.col("p_elo")).otherwise(1 - pl.col("p_elo")).alias("p_elo_player"),
        pl.when(a).then(pl.col("a_won")).otherwise(~pl.col("a_won")).alias("player_won"),
    )


def lookup(df: pl.DataFrame, match_id: str) -> dict | None:
    """One recorded prediction with its context (``None`` if the match has no recorded eligible prediction)."""
    r = df.filter(pl.col("match_id") == match_id)
    if r.height == 0:
        return None
    row = r.row(0, named=True)
    winner = row["player_a_name"] if row["a_won"] else row["player_b_name"]
    fav_m4a = row["player_a_name"] if row["p_m4a"] >= 0.5 else row["player_b_name"]
    fav_elo = row["player_a_name"] if row["p_elo"] >= 0.5 else row["player_b_name"]
    ll = {k: float(row_losses(np.array([row[k]]), np.array([float(row["a_won"])]))[0][0]) for k in MODELS}
    return {**row, "winner": winner, "m4a_favourite": fav_m4a, "elo_favourite": fav_elo,
            "m4a_correct": fav_m4a == winner, "elo_correct": fav_elo == winner, "log_loss": ll}  # fmt: skip


def metrics(df: pl.DataFrame, bins: int | None = None) -> dict[str, dict]:
    """Log loss, Brier, accuracy and ECE for both models with the project's scorer (identical rows)."""
    if df.height == 0:
        return {}
    y = df["a_won"].to_numpy().astype(np.float64)
    b = bins or calibration_bins()
    return {k: summarize(df[k].to_numpy(), y, b) for k in MODELS}


def calibration_curve(df: pl.DataFrame, model: str, bins: int | None = None) -> pl.DataFrame:
    return calibration(df.select(pl.col(model).alias("p"), pl.col("a_won").alias("target_a_wins")), "p",
                       bins or calibration_bins())  # fmt: skip


def paired(df: pl.DataFrame) -> pl.DataFrame:
    """Per-match log-loss and Brier differences, M4A − surface Elo (negative: M4A better)."""
    y = df["a_won"].to_numpy().astype(np.float64)
    ll_m, br_m = row_losses(df["p_m4a"].to_numpy(), y)
    ll_e, br_e = row_losses(df["p_elo"].to_numpy(), y)
    return df.select("match_id", "season", "surface", "tournament_level", "phase").with_columns(
        pl.Series("d_log_loss", ll_m - ll_e), pl.Series("d_brier", br_m - br_e)
    )


def paired_by(df: pl.DataFrame, by: str) -> pl.DataFrame:
    """Mean paired differences per group (point estimates only; the published bootstrap gives the intervals)."""
    p = paired(df)
    return (p.group_by(by).agg(pl.len().alias("n"), pl.col("d_log_loss").mean(), pl.col("d_brier").mean(),
                               (pl.col("d_log_loss") < 0).mean().alias("share_m4a_better"))
            .sort(by))  # fmt: skip


def confidence(df: pl.DataFrame) -> pl.DataFrame:
    """Confidence (probability of the favourite) and correctness per model, long format."""
    frames = []
    for k, name in MODELS.items():
        frames.append(df.select(
            pl.lit(name).alias("model"),
            pl.max_horizontal(pl.col(k), 1 - pl.col(k)).alias("confidence"),
            ((pl.col(k) >= 0.5) == pl.col("a_won")).alias("correct")))  # fmt: skip
    return pl.concat(frames)


# ----------------------------------------------------------------------------------------------------- players
def player_options(players: pl.DataFrame, min_matches: int = 1) -> pl.DataFrame:
    return players.filter(pl.col("matches") >= min_matches).sort("name")


def snapshot_asof(snap: pl.DataFrame, player_id: int, as_of: date, surface: str | None = None) -> dict | None:
    """Pre-match state entering the player's last recorded match on or before ``as_of`` (optionally on a surface).

    Results of that match and anything later are excluded: this is what the frozen pipeline knew at that point."""
    r = snap.filter((pl.col("player_id") == player_id) & (pl.col("tournament_date") <= as_of))
    if surface:
        r = r.filter(pl.col("surface") == surface)
    if r.height == 0:
        return None
    return r.sort("tournament_date", "prediction_batch", "match_id").row(-1, named=True)


def recent_form(snap: pl.DataFrame, player_id: int, as_of: date, n: int = 10) -> pl.DataFrame:
    """Last ``n`` decided tour-level matches strictly before ``as_of`` (completed or retired)."""
    return (snap.filter((pl.col("player_id") == player_id) & (pl.col("tournament_date") < as_of)
                        & pl.col("won").is_not_null())
            .sort("tournament_date", "match_id").tail(n).reverse())  # fmt: skip


def rating_history(snap: pl.DataFrame, player_id: int, as_of: date) -> pl.DataFrame:
    return (snap.filter((pl.col("player_id") == player_id) & (pl.col("tournament_date") <= as_of))
            .sort("tournament_date").select("tournament_date", "surface", "elo_global", "elo_surface"))  # fmt: skip


def comparison(snap: pl.DataFrame, a: int, b: int, as_of: date, surface: str) -> dict:
    """Side-by-side indicators as of ``as_of`` (no win probability: prospective predictions are disabled)."""
    out = {}
    for pid in (a, b):
        last = snapshot_asof(snap, pid, as_of)
        surf = snapshot_asof(snap, pid, as_of, surface)
        out[pid] = {"last": last, "surface": surf}
    return out


def h2h(preds: pl.DataFrame, a: int, b: int, as_of: date) -> pl.DataFrame:
    """Recorded historical predictions for matches between the two players before ``as_of``."""
    d = filter_predictions(preds, player_ids=[a], opponent_id=b).filter(pl.col("tournament_date") <= as_of)
    return orient(d, a).sort("tournament_date", descending=True)


def season_coverage(snap: pl.DataFrame) -> pl.DataFrame:
    """Tour-level matches per season in the snapshot table (each match appears once per player)."""
    return (snap.with_columns(pl.col("tournament_date").dt.year().alias("season"))
            .group_by("season").agg((pl.len() / 2).cast(pl.Int64).alias("matches"))
            .sort("season"))  # fmt: skip


def fmt_ci(ci: list[float], digits: int = 4) -> str:
    return f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]" if ci and all(map(math.isfinite, ci)) else "–"

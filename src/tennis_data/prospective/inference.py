"""Frozen-model inference: pending-row state -> M4A probability, with integrity checks and diagnostics.

The inputs are the Phase 3B derived features (``modeling.dataset.DERIVED``) built exactly as in development:
the same expressions, the same epsilon and the same "missing difference -> 0" fallback, which is the only
missing-data behaviour the frozen pipeline defines. Scaling and coefficients come from the frozen artifact; nothing
is fitted at inference time. Artifacts are checked against the Phase 3B manifest checksums and the frozen hash
before any prediction; a mismatch stops inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from tennis_data.modeling.calibration import Calibrator, fit_calibrator
from tennis_data.modeling.dataset import DERIVED, assert_inputs_allowed
from tennis_data.modeling.experiments import verify_frozen
from tennis_data.modeling.logistic import predict
from tennis_data.modeling.preprocessing import Scaler
from tennis_data.prospective.config import ProspectiveConfig

PREDICTION_MODES = ("historical_reconstruction", "exploratory_retrospective", "prospective")


class ArtifactIntegrityError(RuntimeError):
    """A frozen artifact does not match its published checksum or hash. Prediction must not proceed."""


class InferenceError(ValueError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_version: str
    frozen_hash: str
    inputs: tuple[str, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    calibrator: dict
    epsilon: float
    source: str  # "production" or "walk_forward:<year>"
    artifact_sha256: str
    training_cutoff: str
    extra: dict = field(default_factory=dict)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(cfg: ProspectiveConfig) -> dict:
    return json.loads((cfg.modeling.modeling_dir / "manifest.json").read_text())


def _check(cfg: ProspectiveConfig, rel: str) -> tuple[Path, str]:
    man = _manifest(cfg)
    path = cfg.modeling.modeling_dir / rel
    if rel not in man["artifacts"]:
        raise ArtifactIntegrityError(f"{rel} is not listed in the Phase 3B manifest")
    sha = _sha(path)
    if sha != man["artifacts"][rel]["sha256"]:
        raise ArtifactIntegrityError(f"{rel}: sha256 {sha[:12]} != manifest {man['artifacts'][rel]['sha256'][:12]}")
    return path, sha


def load_frozen(cfg: ProspectiveConfig) -> dict:
    path, _ = _check(cfg, "models/frozen_model.json")
    frozen = json.loads(path.read_text())
    try:
        verify_frozen(frozen)
    except AssertionError as e:
        raise ArtifactIntegrityError(str(e)) from e
    if not frozen["frozen_hash"].startswith(cfg.expected_frozen_hash):
        raise ArtifactIntegrityError(
            f"frozen hash {frozen['frozen_hash'][:16]} is not the expected {cfg.expected_frozen_hash[:16]}"
        )
    if frozen["configuration_hash"] != cfg.modeling.configuration_hash():
        raise ArtifactIntegrityError("modeling configuration changed since the model was frozen")
    if _manifest(cfg)["frozen_hash"] != frozen["frozen_hash"]:
        raise ArtifactIntegrityError("manifest and frozen model disagree")
    return frozen


def load_production(cfg: ProspectiveConfig) -> ModelArtifact:
    frozen = load_frozen(cfg)
    path, sha = _check(cfg, "models/production_model.json")
    prod = json.loads(path.read_text())
    if prod["frozen_hash"] != frozen["frozen_hash"] or prod["specification"] != frozen["specification"]:
        raise ArtifactIntegrityError("production model does not belong to the frozen specification")
    spec, fit = prod["specification"], prod["fit"]
    inputs = tuple(spec["inputs"])
    return ModelArtifact(prod["model_id"], prod["model_version"], prod["frozen_hash"], inputs,
                         tuple(fit["scale"][c] for c in inputs), tuple(fit["coef_scaled"][c] for c in inputs),
                         fit["calibrator"], spec["epsilon"], "production", sha, str(fit["training_cutoff_batch"]))  # fmt: skip


def load_walk_forward(cfg: ProspectiveConfig, year: int) -> ModelArtifact:
    """The frozen specification's annual fit for ``year`` (used for historical replay).

    A calibrated specification also needs that year's calibrator. It is re-derived with the frozen Phase 3B rule
    (``experiments.calibrate``: the same model's stored out-of-fold raw predictions for years y-window..y-1 whose
    label was released before the year's cutoff) from the checksummed ``experiments.parquet``; nothing is tuned.
    """
    frozen = load_frozen(cfg)
    spec = frozen["specification"]
    path, sha = _check(cfg, "models/coefficients.parquet")
    phase = "development" if year <= cfg.modeling.development_years[1] else "exploratory"
    c = pl.read_parquet(path).filter((pl.col("experiment") == spec["experiment"]) & (pl.col("C") == spec["C"])
                                     & (pl.col("year") == year) & (pl.col("phase") == phase))  # fmt: skip
    if c.height != len(spec["inputs"]):
        raise InferenceError(f"no stored {spec['experiment']} fit for {year}")
    by = {r["feature"]: r for r in c.iter_rows(named=True)}
    inputs = tuple(spec["inputs"])
    cutoff = by[inputs[0]]["training_cutoff"]
    cal = {"method": "raw"}
    if spec["calibration"] != "raw":
        epath, _ = _check(cfg, "predictions/experiments.parquet")
        w = spec["calibration_window_years"]
        oof = pl.read_parquet(epath, columns=["model_id", "prediction_batch", "release_batch", "raw_probability_a",
                                              "target_a_wins"]).filter(
            (pl.col("model_id") == spec["experiment"]) & (pl.col("prediction_batch").dt.year() >= year - w)
            & (pl.col("prediction_batch").dt.year() < year) & (pl.col("release_batch") < cutoff)
        )  # fmt: skip
        fit = fit_calibrator(spec["calibration"], oof["raw_probability_a"].to_numpy(),
                             oof["target_a_wins"].cast(pl.Float64).to_numpy(), spec["epsilon"])  # fmt: skip
        cal = {"method": fit.method, "rows": fit.rows, "slope": fit.slope, "knots_x": list(fit.knots_x),
               "knots_y": list(fit.knots_y)}  # fmt: skip
    return ModelArtifact(frozen["model_id"], frozen["model_version"], frozen["frozen_hash"], inputs,
                         tuple(by[x]["scale"] for x in inputs), tuple(by[x]["coef_scaled"] for x in inputs),
                         cal, spec["epsilon"], f"walk_forward:{year}", sha, str(cutoff))  # fmt: skip


def derive(state: pl.DataFrame, art: ModelArtifact, cfg: ProspectiveConfig) -> tuple[np.ndarray, pl.DataFrame]:
    """Model matrix exactly as ``modeling.dataset.build_from_frames`` builds it, plus which inputs were filled."""
    assert_inputs_allowed(art.inputs, cfg.modeling)
    exprs, filled = [], []
    for n in art.inputs:
        d = DERIVED[n]
        e = d.build(art.epsilon)
        exprs.append((e.fill_null(0.0) if d.fill_zero else e).cast(pl.Float64).alias(n))
        filled.append((e.is_null() if d.fill_zero else pl.lit(False)).alias(f"filled_{n}"))
    m = state.select("match_id", *exprs, *filled)
    x = m.select(list(art.inputs)).to_numpy().astype(np.float64)
    if not np.isfinite(x).all():
        raise InferenceError("non-finite model input after the frozen fallback")
    return x, m


def apply_model(x: np.ndarray, art: ModelArtifact) -> tuple[np.ndarray, np.ndarray]:
    xs = Scaler(art.inputs, art.scale).transform(x)
    coef = np.asarray(art.coef)
    pa, pb = predict(coef, xs), predict(coef, -xs)  # B's probability from the swapped (negated) inputs
    cal = art.calibrator or {"method": "raw"}
    if cal["method"] != "raw":
        c = Calibrator(cal["method"], art.epsilon, cal.get("rows", 0), cal.get("slope", 1.0),
                       tuple(cal.get("knots_x", ())), tuple(cal.get("knots_y", ())))  # fmt: skip
        pa, pb = c.apply(pa), c.apply(pb)
    dev = np.abs(pa + pb - 1)
    if dev.size and dev.max() > 1e-12:
        raise InferenceError(f"player-order symmetry violated by {dev.max():.2e}")
    return pa, pb


AVAILABILITY_COLUMNS = (
    "a_has_stats_history", "b_has_stats_history", "a_weeks_since_last_stats", "b_weeks_since_last_stats",
    "a_service_points_observed", "b_service_points_observed", "a_service_ess_decay52w", "b_service_ess_decay52w",
    "adjustment_fit_batch", "global_match_count_a", "global_match_count_b",
)  # fmt: skip


def predict_state(pending: pl.DataFrame, art: ModelArtifact, cfg: ProspectiveConfig, *, prediction_mode: str,
                  snapshot_id: str, components: dict, freshness: dict | None = None,
                  generated_at: datetime | None = None) -> pl.DataFrame:  # fmt: skip
    if prediction_mode not in PREDICTION_MODES:
        raise InferenceError(f"prediction_mode must be one of {PREDICTION_MODES}")
    x, m = derive(pending, art, cfg)
    pa, pb = apply_model(x, art)
    gen = (generated_at or datetime.now(UTC)).isoformat()
    rows = []
    for i, r in enumerate(pending.iter_rows(named=True)):
        avail = {
            c: (str(r[c]) if c == "adjustment_fit_batch" and r[c] is not None else r[c]) for c in AVAILABILITY_COLUMNS
        }
        avail["filled_inputs"] = [n for n in art.inputs if m[f"filled_{n}"][i]]
        state_hash = hashlib.sha256(json.dumps([snapshot_id, art.artifact_sha256, x[i].tolist()]).encode()).hexdigest()[
            :16
        ]
        rows.append({
            "match_id": r["match_id"].removeprefix("pending:"), "player_a_id": r["player_a_id"],
            "player_b_id": r["player_b_id"], "surface": r["surface"], "prediction_batch": r["prediction_batch"],
            "probability_a": float(pa[i]), "probability_b": float(pb[i]),
            "baseline_elo_probability_a": float(r["predicted_probability_a"]),
            "model_id": art.model_id, "model_version": art.model_version, "frozen_hash": art.frozen_hash,
            "artifact": art.source, "artifact_sha256": art.artifact_sha256, "training_cutoff": art.training_cutoff,
            "feature_state_hash": state_hash, "state_snapshot_id": snapshot_id,
            "source_snapshots": json.dumps(components, sort_keys=True),
            "feature_availability": json.dumps(avail, sort_keys=True, default=str),
            "freshness": json.dumps((freshness or {}).get(r["match_id"].removeprefix("pending:"), {}), sort_keys=True,
                                    default=str),
            "prediction_mode": prediction_mode, "generated_at": gen,
        })  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)

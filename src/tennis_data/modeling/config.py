"""Phase 3B configuration (config/modeling.toml)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.features.config import FeatureConfig, load_feature_config

DEFAULT_MODELING_CONFIG = PROJECT_ROOT / "config" / "modeling.toml"
MODELING_ENGINE_VERSION = "1"


@dataclass(frozen=True)
class Experiment:
    id: str
    description: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class ModelingConfig:
    model_version: str
    baseline: str
    epsilon: float
    train_start: date
    first_prediction_year: int
    development_years: tuple[int, int]
    exploratory_years: tuple[int, int]
    folds: tuple[tuple[int, int], ...]
    c_grid: tuple[float, ...]
    max_iterations: int
    tolerance: float
    calibration_window: int
    calibration_methods: tuple[str, ...]
    isotonic_min_rows: int
    candidates: tuple[str, ...]
    simplicity_tolerance: float
    calibration_bins: int
    bootstrap_samples: int
    bootstrap_seed: int
    sample_size_edges: tuple[int, ...]
    experiments: tuple[Experiment, ...]
    features: FeatureConfig
    modeling_dir: Path
    reports_dir: Path
    config_path: Path = field(default=Path())

    @property
    def full_version(self) -> str:
        return f"{self.model_version}+engine{MODELING_ENGINE_VERSION}"

    def experiment(self, exp_id: str) -> Experiment:
        for e in self.experiments:
            if e.id == exp_id:
                return e
        raise KeyError(exp_id)

    def dev_years(self) -> range:
        return range(self.development_years[0], self.development_years[1] + 1)

    def exploratory(self) -> range:
        return range(self.exploratory_years[0], self.exploratory_years[1] + 1)

    def hash_payload(self) -> dict:
        d = asdict(self)
        for k in ("features", "modeling_dir", "reports_dir", "config_path"):
            d.pop(k)
        d["model_version"] = self.full_version
        d["upstream_feature_hash"] = self.features.configuration_hash()
        d["upstream_elo_hash_payload"] = self.features.elo.hash_payload()
        return d

    def configuration_hash(self) -> str:
        blob = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_modeling_config(path: str | Path | None = None, root: str | Path | None = None) -> ModelingConfig:
    config_path = Path(path) if path else DEFAULT_MODELING_CONFIG
    if not config_path.exists():
        raise ConfigError(f"Modeling config not found: {config_path}")
    base = Path(root) if root else PROJECT_ROOT
    with config_path.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    feats = load_feature_config(resolve(d.get("features_config", "config/features.toml")), root=base)
    m, c, lg, cal, sel, ev, out = (d.get(k, {}) for k in ("model", "chronology", "logistic", "calibration",
                                                         "selection", "evaluation", "outputs"))  # fmt: skip
    exps = tuple(Experiment(e["id"], e.get("description", ""), tuple(e["inputs"])) for e in d.get("experiments", []))
    cfg = ModelingConfig(
        model_version=m.get("model_version", "m3b-1"),
        baseline=m.get("baseline", "surface_elo"),
        epsilon=float(m.get("probability_epsilon", 1e-6)),
        train_start=date.fromisoformat(str(c["train_start"])),
        first_prediction_year=int(c["first_prediction_year"]),
        development_years=tuple(int(x) for x in c["development_years"]),
        exploratory_years=tuple(int(x) for x in c["exploratory_years"]),
        folds=tuple(tuple(int(x) for x in f) for f in c["folds"]),
        c_grid=tuple(float(x) for x in lg["c_grid"]),
        max_iterations=int(lg.get("max_iterations", 100)),
        tolerance=float(lg.get("tolerance", 1e-10)),
        calibration_window=int(cal.get("window_years", 3)),
        calibration_methods=tuple(cal.get("methods", ["raw", "temperature"])),
        isotonic_min_rows=int(cal.get("isotonic_min_rows", 5000)),
        candidates=tuple(sel["candidates"]),
        simplicity_tolerance=float(sel.get("simplicity_tolerance", 0.0)),
        calibration_bins=int(ev.get("calibration_bins", 10)),
        bootstrap_samples=int(ev.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(ev.get("bootstrap_seed", 0)),
        sample_size_edges=tuple(int(x) for x in ev.get("sample_size_edges", [1, 1000, 5000])),
        experiments=exps,
        features=feats,
        modeling_dir=resolve(out.get("modeling_dir", "data/modeling")),
        reports_dir=resolve(out.get("reports_dir", "reports/modeling")),
        config_path=config_path.resolve(),
    )
    ids = [e.id for e in exps]
    if len(set(ids)) != len(ids):
        raise ConfigError("experiment ids must be unique")
    if "M0" in ids:
        raise ConfigError("M0 is the Phase 2 baseline and cannot be an experiment id")
    missing = [x for x in cfg.candidates if x not in ids]
    if missing:
        raise ConfigError(f"selection candidates not in the registry: {missing}")
    d0, d1 = cfg.development_years
    e0, e1 = cfg.exploratory_years
    if not (cfg.train_start.year < cfg.first_prediction_year <= d0 - cfg.calibration_window and d1 < e0 <= e1):
        raise ConfigError("need train_start < first_prediction_year <= dev_start - window and dev_end < exploratory")
    if [y for f in cfg.folds for y in range(f[0], f[1] + 1)] != list(range(d0, d1 + 1)):
        raise ConfigError("folds must partition the development years in order")
    bad = set(cfg.calibration_methods) - {"raw", "temperature", "isotonic"}
    if bad or "raw" not in cfg.calibration_methods:
        raise ConfigError("calibration methods: subset of raw/temperature/isotonic including raw")
    return cfg

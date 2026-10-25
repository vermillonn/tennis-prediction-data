"""Feature configuration. Chronology and the target population are taken from the Phase 2 Elo config."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.elo.config import EloConfig, load_elo_config

DEFAULT_FEATURES_CONFIG = PROJECT_ROOT / "config" / "features.toml"
FEATURE_ENGINE_VERSION = "2"  # 2: Kish ESS columns, monotone-safeguarded opponent fit


@dataclass(frozen=True)
class Shrinkage:
    kappa_service: float = 200.0
    kappa_return: float = 300.0
    kappa_surface_service: float = 600.0
    kappa_surface_return: float = 500.0
    min_prior_points: int = 20000


@dataclass(frozen=True)
class Opponent:
    enabled: bool = True
    refit_interval_weeks: int = 4
    half_life_weeks: float = 104.0
    kappa_serve: float = 200.0
    kappa_return: float = 300.0
    kappa_surface: float = 550.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    max_step: float = 1.0
    min_fit_points: int = 20000
    store_fits: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str
    include_retirements: bool
    last_matches: tuple[int, ...]
    last_weeks: tuple[int, ...]
    half_life_weeks: tuple[float, ...]
    shrinkage: Shrinkage
    opponent: Opponent
    elo: EloConfig
    features_dir: Path
    reports_dir: Path

    @property
    def full_version(self) -> str:
        return f"{self.feature_version}+engine{FEATURE_ENGINE_VERSION}"

    def with_(self, **kw) -> FeatureConfig:
        return replace(self, **kw)

    def hash_payload(self) -> dict:
        elo = self.elo.hash_payload()
        return {
            "feature_version": self.full_version,
            "include_retirements": self.include_retirements,
            "last_matches": list(self.last_matches),
            "last_weeks": list(self.last_weeks),
            "half_life_weeks": list(self.half_life_weeks),
            "shrinkage": asdict(self.shrinkage),
            "opponent": {k: v for k, v in asdict(self.opponent).items() if k != "store_fits"},
            # Shared with Phase 2: chronology, pools, targets, source commit (not the Elo parameters).
            "chronology": elo["chronology"],
            "pools": elo["pools"],
            "target_statuses": elo["eligibility"]["target_statuses"],
            "source_commit": elo["source_commit"],
        }

    def configuration_hash(self) -> str:
        blob = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_feature_config(path: str | Path | None = None, root: str | Path | None = None) -> FeatureConfig:
    config_path = Path(path) if path else DEFAULT_FEATURES_CONFIG
    if not config_path.exists():
        raise ConfigError(f"Feature config not found: {config_path}")
    base = Path(root) if root else PROJECT_ROOT
    with config_path.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    elo = load_elo_config(resolve(d.get("elo_config", "config/elo.toml")), root=base)
    f, w, dc = d.get("features", {}), d.get("windows", {}), d.get("decay", {})
    s, o, out = d.get("shrinkage", {}), d.get("opponent", {}), d.get("outputs", {})
    shrink = Shrinkage(**{k: type(getattr(Shrinkage(), k))(v) for k, v in s.items()})
    opp = Opponent(**{k: type(getattr(Opponent(), k))(v) for k, v in o.items()})
    cfg = FeatureConfig(
        feature_version=f.get("feature_version", "sr-1"),
        include_retirements=bool(f.get("include_retirements", False)),
        last_matches=tuple(int(x) for x in w.get("last_matches", [5, 20])),
        last_weeks=tuple(int(x) for x in w.get("last_weeks", [13, 52])),
        half_life_weeks=tuple(float(x) for x in dc.get("half_life_weeks", [52])),
        shrinkage=shrink,
        opponent=opp,
        elo=elo,
        features_dir=resolve(out.get("features_dir", "data/features")),
        reports_dir=resolve(out.get("reports_dir", "reports/features")),
    )
    for name, vals in (("last_matches", cfg.last_matches), ("last_weeks", cfg.last_weeks)):
        if any(v <= 0 for v in vals) or len(set(vals)) != len(vals):
            raise ConfigError(f"[windows].{name} must be distinct positive integers")
    if any(h <= 0 for h in cfg.half_life_weeks):
        raise ConfigError("[decay].half_life_weeks must be positive")
    for k, v in asdict(shrink).items():
        if v < 0:
            raise ConfigError(f"[shrinkage].{k} must be >= 0")
    if opp.refit_interval_weeks < 1 or min(opp.kappa_serve, opp.kappa_return, opp.kappa_surface) <= 0:
        raise ConfigError("[opponent] refit_interval_weeks >= 1 and kappas > 0 are required")
    return cfg

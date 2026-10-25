"""Elo configuration: model parameters, chronology, eligibility, pools, evaluation and tuning grid."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, Config, ConfigError, load_config

DEFAULT_ELO_CONFIG = PROJECT_ROOT / "config" / "elo.toml"
ENGINE_VERSION = "1"  # bump when engine semantics change; part of model_version

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SURFACE_INITS = ("constant", "global")
CARPET_POLICIES = ("separate", "hard", "none")
K_MODES = ("constant", "decay")


@dataclass(frozen=True)
class EloParams:
    """Everything that changes ratings or probabilities. Hashed into configuration_hash."""

    initial_rating: float = 1500.0
    k_mode: str = "constant"
    k_constant: float = 32.0
    k_numerator: float = 250.0
    k_offset: float = 5.0
    k_shape: float = 0.4
    surface_init: str = "constant"
    carpet_policy: str = "separate"
    shrinkage_matches: float = 20.0

    def __post_init__(self):
        if self.k_mode not in K_MODES:
            raise ConfigError(f"k mode must be one of {K_MODES}, got {self.k_mode!r}")
        if self.surface_init not in SURFACE_INITS:
            raise ConfigError(f"surface_init must be one of {SURFACE_INITS}, got {self.surface_init!r}")
        if self.carpet_policy not in CARPET_POLICIES:
            raise ConfigError(f"carpet_policy must be one of {CARPET_POLICIES}, got {self.carpet_policy!r}")
        if self.shrinkage_matches < 0:
            raise ConfigError("shrinkage_matches must be >= 0")


@dataclass(frozen=True)
class MultiWeekRule:
    levels: tuple[str, ...]
    weeks: int
    min_draw_size: int | None = None


@dataclass(frozen=True)
class Chronology:
    week_start: str = "saturday"
    multi_week: tuple[MultiWeekRule, ...] = ()

    @property
    def week_start_index(self) -> int:
        return WEEKDAYS.index(self.week_start)


@dataclass(frozen=True)
class Eligibility:
    update_statuses: tuple[str, ...] = ("completed",)
    target_statuses: tuple[str, ...] = ("completed",)
    sensitivity_retirements: bool = True


@dataclass(frozen=True)
class Pool:
    name: str
    families: tuple[str, ...]
    levels: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    validation_start: date
    test_start: date
    test_end: date
    calibration_bins: int = 10
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class Tuning:
    k_constant: tuple[float, ...] = ()
    k_decay: tuple[tuple[float, float, float], ...] = ()  # (numerator, offset, shape)
    surface_init: tuple[str, ...] = ("constant",)
    shrinkage_matches: tuple[float, ...] = (20.0,)


@dataclass(frozen=True)
class EloConfig:
    params: EloParams
    chronology: Chronology
    eligibility: Eligibility
    pools: tuple[Pool, ...]
    evaluation: Evaluation
    tuning: Tuning
    model_version: str
    data: Config
    elo_dir: Path
    reports_dir: Path
    config_path: Path = field(default=Path())

    def with_params(self, params: EloParams) -> EloConfig:
        return replace(self, params=params)

    def with_eligibility(self, eligibility: Eligibility) -> EloConfig:
        return replace(self, eligibility=eligibility)

    @property
    def full_model_version(self) -> str:
        return f"{self.model_version}+engine{ENGINE_VERSION}"

    def hash_payload(self) -> dict:
        """The settings that determine predictions (not paths, periods or the tuning grid)."""
        return {
            "params": asdict(self.params),
            "chronology": asdict(self.chronology),
            "eligibility": {
                "update_statuses": list(self.eligibility.update_statuses),
                "target_statuses": list(self.eligibility.target_statuses),
            },
            "pools": [asdict(p) for p in self.pools],
            "model_version": self.full_model_version,
            "source_commit": self.data.source.commit,
        }

    def configuration_hash(self) -> str:
        blob = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_elo_config(path: str | Path | None = None, root: str | Path | None = None) -> EloConfig:
    config_path = Path(path) if path else DEFAULT_ELO_CONFIG
    if not config_path.exists():
        raise ConfigError(f"Elo config not found: {config_path}")
    base = Path(root) if root else PROJECT_ROOT
    with config_path.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    data = load_config(resolve(d.get("data_config", "config/default.toml")), root=base)
    m = d.get("model", {})
    k = m.get("k", {})
    params = EloParams(
        initial_rating=float(m.get("initial_rating", 1500.0)),
        k_mode=k.get("mode", "constant"),
        k_constant=float(k.get("constant", 32.0)),
        k_numerator=float(k.get("numerator", 250.0)),
        k_offset=float(k.get("offset", 5.0)),
        k_shape=float(k.get("shape", 0.4)),
        surface_init=m.get("surface_init", "constant"),
        carpet_policy=m.get("carpet_policy", "separate"),
        shrinkage_matches=float(m.get("shrinkage_matches", 20.0)),
    )
    c = d.get("chronology", {})
    if c.get("week_start", "saturday") not in WEEKDAYS:
        raise ConfigError(f"[chronology].week_start must be a weekday name, got {c.get('week_start')!r}")
    chronology = Chronology(
        week_start=c.get("week_start", "saturday"),
        multi_week=tuple(
            MultiWeekRule(tuple(r["levels"]), int(r["weeks"]), r.get("min_draw_size")) for r in c.get("multi_week", [])
        ),
    )
    if any(r.weeks < 1 for r in chronology.multi_week):
        raise ConfigError("[chronology.multi_week].weeks must be >= 1")
    e = d.get("eligibility", {})
    eligibility = Eligibility(
        update_statuses=tuple(e.get("update_statuses", ["completed"])),
        target_statuses=tuple(e.get("target_statuses", ["completed"])),
        sensitivity_retirements=bool(e.get("sensitivity_retirements", True)),
    )
    bad = {"walkover", "default", "unknown", "unfinished", "abandoned"} & (
        set(eligibility.update_statuses) | set(eligibility.target_statuses)
    )
    if bad:
        raise ConfigError(f"Statuses {sorted(bad)} have no trustworthy played result and cannot be Elo-eligible")
    pools = tuple(Pool(p["name"], tuple(p["families"]), tuple(p["levels"])) for p in d.get("pools", []))
    if not pools:
        raise ConfigError("At least one [[pools]] entry is required")
    if len({p.name for p in pools}) != len(pools):
        raise ConfigError("Pool names must be unique")
    ev = d.get("evaluation", {})
    evaluation = Evaluation(
        validation_start=_date(ev["validation_start"]),
        test_start=_date(ev["test_start"]),
        test_end=_date(ev["test_end"]),
        calibration_bins=int(ev.get("calibration_bins", 10)),
        bootstrap_samples=int(ev.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(ev.get("bootstrap_seed", 0)),
    )
    if not evaluation.validation_start < evaluation.test_start <= evaluation.test_end:
        raise ConfigError("Require validation_start < test_start <= test_end")
    t = d.get("tuning", {})
    tuning = Tuning(
        k_constant=tuple(float(x) for x in t.get("k_constant", [])),
        k_decay=tuple((float(x["numerator"]), float(x["offset"]), float(x["shape"])) for x in t.get("k_decay", [])),
        surface_init=tuple(t.get("surface_init", ["constant"])),
        shrinkage_matches=tuple(float(x) for x in t.get("shrinkage_matches", [20.0])),
    )
    out = d.get("outputs", {})
    return EloConfig(
        params=params,
        chronology=chronology,
        eligibility=eligibility,
        pools=pools,
        evaluation=evaluation,
        tuning=tuning,
        model_version=d.get("model", {}).get("model_version", "elo-1"),
        data=data,
        elo_dir=resolve(out.get("elo_dir", "data/elo")),
        reports_dir=resolve(out.get("reports_dir", "reports/elo")),
        config_path=config_path.resolve(),
    )

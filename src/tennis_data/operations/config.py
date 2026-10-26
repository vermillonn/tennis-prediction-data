"""Phase 5A configuration (``config/operations.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.prospective.config import ProspectiveConfig, load_prospective_config

DEFAULT_OPERATIONS_CONFIG = PROJECT_ROOT / "config" / "operations.toml"
PROVIDER_KINDS = ("api", "file", "repo", "website")


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str
    probe: str
    credential_env: str | None
    auth: str | None
    verify: str | None
    offers: dict
    terms: str


@dataclass(frozen=True)
class OperationsConfig:
    prospective: ProspectiveConfig
    evaluation_config: Path
    workspace_dir: Path
    reports_dir: Path
    first_week: date
    archive_last_week: date
    pools: tuple[str, ...]
    calendar_reference_season: int
    overlap_seasons: tuple[int, ...]
    date_tolerance_days: int
    min_matched_observations: int
    min_exact_agreement: float
    required_checks: tuple[str, ...]
    tml: dict
    expected_frozen_hash: str = ""
    expected_configuration_hash: str = ""
    providers: tuple[Provider, ...] = field(default=())
    config_path: Path = field(default=Path())

    def workspace_prospective(self) -> ProspectiveConfig:
        """The Phase 4A configuration pointed at the operations workspace (Phase 4A's own outputs stay untouched)."""
        return replace(self.prospective, prospective_dir=self.workspace_dir / "phase4a_workspace",
                       reports_dir=self.workspace_dir / "phase4a_workspace" / "reports")  # fmt: skip


def load_operations_config(path: str | Path | None = None, root: str | Path | None = None,
                           prospective: ProspectiveConfig | None = None) -> OperationsConfig:  # fmt: skip
    p = Path(path) if path else DEFAULT_OPERATIONS_CONFIG
    if not p.exists():
        raise ConfigError(f"Operations config not found: {p}")
    base = Path(root) if root else PROJECT_ROOT
    with p.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else base / q

    providers = []
    for pr in d.get("providers", []):
        if pr.get("kind") not in PROVIDER_KINDS:
            raise ConfigError(f"provider {pr.get('name')}: kind must be one of {PROVIDER_KINDS}")
        providers.append(Provider(pr["name"], pr["kind"], pr.get("probe", ""), pr.get("credential_env"),
                                  pr.get("auth"), pr.get("verify"), dict(pr.get("offers", {})), pr.get("terms", "")))  # fmt: skip
    if len({x.name for x in providers}) != len(providers):
        raise ConfigError("duplicate provider names")
    o, g, c, r = (d.get(k, {}) for k in ("outputs", "gap", "compatibility", "readiness"))
    cfg = OperationsConfig(
        prospective=prospective or load_prospective_config(resolve(d.get("prospective_config",
                                                                          "config/prospective.toml")), root=base),
        evaluation_config=resolve(d.get("evaluation_config", "config/evaluation.toml")),
        workspace_dir=resolve(o.get("workspace_dir", "data/operations")),
        reports_dir=resolve(o.get("reports_dir", "reports/operations")),
        first_week=date.fromisoformat(str(g.get("first_week", "2026-06-01"))),
        archive_last_week=date.fromisoformat(str(g.get("archive_last_week", "2026-05-25"))),
        pools=tuple(g.get("pools", ["G", "M", "A", "F", "D"])),
        calendar_reference_season=int(g.get("calendar_reference_season", 2025)),
        overlap_seasons=tuple(int(x) for x in c.get("overlap_seasons", [2025, 2026])),
        date_tolerance_days=int(c.get("date_tolerance_days", 7)),
        min_matched_observations=int(c.get("min_matched_observations", 500)),
        min_exact_agreement=float(c.get("min_exact_agreement", 0.98)),
        required_checks=tuple(r.get("required", [])),
        tml=dict(d.get("tml", {})),
        expected_frozen_hash=str(d.get("model", {}).get("expected_frozen_hash", "")),
        expected_configuration_hash=str(d.get("model", {}).get("expected_configuration_hash", "")),
        providers=tuple(providers),
        config_path=p.resolve(),
    )  # fmt: skip
    if cfg.first_week.weekday() != 0 or cfg.archive_last_week.weekday() != 0:
        raise ConfigError("[gap] weeks must be Mondays")
    if cfg.first_week <= cfg.archive_last_week:
        raise ConfigError("[gap].first_week must follow the archive's last week")
    return cfg

"""Pipeline configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"

FAMILIES = ("tour", "qual_chall", "futures")


class ConfigError(ValueError):
    """Raised for missing or inconsistent configuration values."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    repo: str
    commit: str
    mode: str
    base_url: str
    local_path: str
    verify_local_commit: bool
    timeout_seconds: int
    license: str
    attribution: str


@dataclass(frozen=True)
class SelectionConfig:
    families: tuple[str, ...]
    start_year: int
    end_year: int
    include_players: bool
    include_rankings: bool
    # Optional per-family [first, last] season, intersected with start_year..end_year.
    family_years: dict[str, tuple[int, int]] = field(default_factory=dict)

    def years(self, family: str) -> range:
        lo, hi = self.family_years.get(family, (self.start_year, self.end_year))
        return range(max(lo, self.start_year), min(hi, self.end_year) + 1)


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    validated_dir: Path
    quarantine_dir: Path
    reports_dir: Path
    duckdb_path: Path


@dataclass(frozen=True)
class ValidationConfig:
    allowed_surfaces: tuple[str, ...]
    allowed_best_of: tuple[int, ...]
    allowed_levels: tuple[str, ...]
    allowed_rounds: tuple[str, ...]
    multi_meeting_rounds: tuple[str, ...]
    placeholder_player_ids: tuple[int, ...]
    max_service_game_diff: int
    min_stats_coverage: float
    stats_expected_from: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    selection: SelectionConfig
    paths: PathsConfig
    validation: ValidationConfig
    config_path: Path

    @property
    def raw_source_dir(self) -> Path:
        """Directory holding the immutable raw snapshot for the pinned commit."""
        slug = self.source.repo.replace("/", "__")
        return self.paths.raw_dir / slug / self.source.commit


def _require(table: dict, key: str, section: str):
    if key not in table:
        raise ConfigError(f"Missing required config key [{section}].{key}")
    return table[key]


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load a config file. Relative paths resolve against ``root`` (default: project root)."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    base = Path(root) if root else PROJECT_ROOT

    src = data.get("source", {})
    source = SourceConfig(
        name=_require(src, "name", "source"),
        repo=_require(src, "repo", "source"),
        commit=_require(src, "commit", "source"),
        mode=src.get("mode", "download"),
        base_url=src.get("base_url", "https://raw.githubusercontent.com").rstrip("/"),
        local_path=src.get("local_path", ""),
        verify_local_commit=bool(src.get("verify_local_commit", True)),
        timeout_seconds=int(src.get("timeout_seconds", 120)),
        license=src.get("license", ""),
        attribution=src.get("attribution", ""),
    )
    if source.mode not in ("download", "local"):
        raise ConfigError(f"[source].mode must be 'download' or 'local', got {source.mode!r}")
    if source.mode == "local" and not source.local_path:
        raise ConfigError("[source].mode = 'local' requires [source].local_path")
    if len(source.commit) != 40 or any(c not in "0123456789abcdef" for c in source.commit):
        raise ConfigError(f"[source].commit must be a full 40-char lowercase SHA, got {source.commit!r}")

    sel = data.get("selection", {})
    selection = SelectionConfig(
        families=tuple(_require(sel, "families", "selection")),
        start_year=int(_require(sel, "start_year", "selection")),
        end_year=int(_require(sel, "end_year", "selection")),
        include_players=bool(sel.get("include_players", True)),
        include_rankings=bool(sel.get("include_rankings", True)),
        family_years={k: (int(v[0]), int(v[1])) for k, v in sel.get("family_years", {}).items()},
    )
    unknown = (set(selection.families) | set(selection.family_years)) - set(FAMILIES)
    if unknown:
        raise ConfigError(f"Unknown families {sorted(unknown)}; valid: {list(FAMILIES)}")
    if selection.start_year > selection.end_year:
        raise ConfigError("[selection].start_year must be <= end_year")

    p = data.get("paths", {})

    def resolve(key: str) -> Path:
        value = Path(_require(p, key, "paths"))
        return value if value.is_absolute() else base / value

    paths = PathsConfig(
        raw_dir=resolve("raw_dir"),
        validated_dir=resolve("validated_dir"),
        quarantine_dir=resolve("quarantine_dir"),
        reports_dir=resolve("reports_dir"),
        duckdb_path=resolve("duckdb_path"),
    )
    dirs = [paths.raw_dir, paths.validated_dir, paths.quarantine_dir, paths.reports_dir]
    if len({d.resolve() for d in dirs}) != len(dirs):
        raise ConfigError("raw, validated, quarantine and reports directories must be distinct")

    v = data.get("validation", {})
    validation = ValidationConfig(
        allowed_surfaces=tuple(v["allowed_surfaces"]),
        allowed_best_of=tuple(int(x) for x in v["allowed_best_of"]),
        allowed_levels=tuple(str(x) for x in v["allowed_levels"]),
        allowed_rounds=tuple(v["allowed_rounds"]),
        multi_meeting_rounds=tuple(v["multi_meeting_rounds"]),
        placeholder_player_ids=tuple(int(x) for x in v.get("placeholder_player_ids", [])),
        max_service_game_diff=int(v.get("max_service_game_diff", 2)),
        min_stats_coverage=float(v.get("min_stats_coverage", 0.8)),
        stats_expected_from={k: int(x) for k, x in v.get("stats_expected_from", {}).items()},
    )
    return Config(source, selection, paths, validation, config_path.resolve())

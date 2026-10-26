"""Phase 4A configuration (config/prospective.toml)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.modeling.config import ModelingConfig, load_modeling_config

DEFAULT_PROSPECTIVE_CONFIG = PROJECT_ROOT / "config" / "prospective.toml"
SCHEMA_VERSION = "provider-match-1"
STATE_ENGINE_VERSION = "state-1"
CHRONOLOGY_POLICIES = ("weekly_batch_v1",)


@dataclass(frozen=True)
class Freshness:
    warn_missing_weeks: int = 1
    fail_missing_weeks: int = 4
    warn_result_age_days: int = 10
    fail_result_age_days: int = 35
    warn_fit_age_weeks: int = 5
    fail_fit_age_weeks: int = 12
    warn_source_age_days: int = 2
    fail_source_age_days: int = 14
    fail_unresolved_identities: int = 1
    warn_player_inactive_weeks: int = 26


@dataclass(frozen=True)
class HttpProviderConfig:
    name: str
    base_url: str
    access_level: str
    language: str
    api_key_env: str
    requests_per_second: float
    max_retries: int
    backoff_seconds: float


@dataclass(frozen=True)
class ProspectiveConfig:
    modeling: ModelingConfig
    expected_frozen_hash: str
    backfill_status: str
    prospective_predictions: str
    prospective_dir: Path
    reports_dir: Path
    chronology_policy: str
    historical_snapshot_end: date
    prospective_start: date
    backfill_family: str
    identity_policy_version: str
    new_player_id_base: int
    auto_approve_high_confidence: bool
    accept_unmatched_as_new: bool
    overrides_path: Path
    reconciliation_policy_version: str
    date_tolerance_days: int
    precedence: tuple[str, ...]
    freshness: Freshness
    http_providers: dict = field(default_factory=dict)
    replay_tolerance: float = 1e-12
    replay_cases_per_category: int = 25
    replay_seed: int = 0
    config_path: Path = field(default=Path())

    @property
    def features(self):
        return self.modeling.features

    @property
    def elo(self):
        return self.modeling.features.elo

    @property
    def data(self):
        return self.modeling.features.elo.data

    def hash_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_engine_version": STATE_ENGINE_VERSION,
            "chronology_policy": self.chronology_policy,
            "historical_snapshot_end": str(self.historical_snapshot_end),
            "prospective_start": str(self.prospective_start),
            "backfill_family": self.backfill_family,
            "identity": [
                self.identity_policy_version,
                self.new_player_id_base,
                self.auto_approve_high_confidence,
                self.accept_unmatched_as_new,
            ],
            "reconciliation": [self.reconciliation_policy_version, self.date_tolerance_days, list(self.precedence)],
            "freshness": asdict(self.freshness),
            "modeling_configuration_hash": self.modeling.configuration_hash(),
            "expected_frozen_hash": self.expected_frozen_hash,
            "status": [self.backfill_status, self.prospective_predictions],
        }

    def configuration_hash(self) -> str:
        blob = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_prospective_config(path: str | Path | None = None, root: str | Path | None = None) -> ProspectiveConfig:
    config_path = Path(path) if path else DEFAULT_PROSPECTIVE_CONFIG
    if not config_path.exists():
        raise ConfigError(f"Prospective config not found: {config_path}")
    base = Path(root) if root else PROJECT_ROOT
    with config_path.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    modeling = load_modeling_config(resolve(d.get("modeling_config", "config/modeling.toml")), root=base)
    out, ch, idn, rec = (d.get(k, {}) for k in ("outputs", "chronology", "identity", "reconciliation"))
    fr = Freshness(**{k: type(getattr(Freshness(), k))(v) for k, v in d.get("freshness", {}).items()})
    http = {
        name: HttpProviderConfig(
            name=name,
            base_url=p["base_url"],
            access_level=p.get("access_level", "trial"),
            language=p.get("language", "en"),
            api_key_env=p["api_key_env"],
            requests_per_second=float(p.get("requests_per_second", 1.0)),
            max_retries=int(p.get("max_retries", 4)),
            backoff_seconds=float(p.get("backoff_seconds", 2.0)),
        )  # fmt: skip
        for name, p in d.get("providers", {}).items()
    }
    rp = d.get("replay", {})
    cfg = ProspectiveConfig(
        modeling=modeling,
        expected_frozen_hash=str(d.get("model", {}).get("expected_frozen_hash", "")),
        backfill_status=str(d.get("status", {}).get("backfill", "INCOMPLETE")),
        prospective_predictions=str(d.get("status", {}).get("prospective_predictions", "DISABLED")),
        prospective_dir=resolve(out.get("prospective_dir", "data/prospective")),
        reports_dir=resolve(out.get("reports_dir", "reports/prospective")),
        chronology_policy=ch.get("policy", "weekly_batch_v1"),
        historical_snapshot_end=date.fromisoformat(str(ch["historical_snapshot_end"])),
        prospective_start=date.fromisoformat(str(ch["prospective_start"])),
        backfill_family=ch.get("backfill_family", "tour"),
        identity_policy_version=idn.get("policy_version", "id-1"),
        new_player_id_base=int(idn.get("new_player_id_base", 9_000_000)),
        auto_approve_high_confidence=bool(idn.get("auto_approve_high_confidence", False)),
        accept_unmatched_as_new=bool(idn.get("accept_unmatched_as_new", True)),
        overrides_path=resolve(idn.get("overrides", "config/identity_overrides.toml")),
        reconciliation_policy_version=rec.get("policy_version", "rec-1"),
        date_tolerance_days=int(rec.get("date_tolerance_days", 7)),
        precedence=tuple(rec.get("precedence", ["historical_archive"])),
        freshness=fr,
        http_providers=http,
        replay_tolerance=float(rp.get("tolerance", 1e-12)),
        replay_cases_per_category=int(rp.get("cases_per_category", 25)),
        replay_seed=int(rp.get("seed", 0)),
        config_path=config_path.resolve(),
    )
    if len(cfg.expected_frozen_hash) < 16:
        raise ConfigError("[model].expected_frozen_hash must name the frozen model (at least 16 hex characters)")
    if cfg.backfill_status not in ("INCOMPLETE", "COMPLETE"):
        raise ConfigError("[status].backfill must be INCOMPLETE or COMPLETE")
    if cfg.prospective_predictions not in ("DISABLED", "ENABLED"):
        raise ConfigError("[status].prospective_predictions must be DISABLED or ENABLED")
    if cfg.prospective_predictions == "ENABLED" and cfg.backfill_status != "COMPLETE":
        raise ConfigError("prospective predictions cannot be ENABLED while the backfill is INCOMPLETE")
    if cfg.chronology_policy not in CHRONOLOGY_POLICIES:
        raise ConfigError(
            f"[chronology].policy must be one of {CHRONOLOGY_POLICIES}; timestamp policies are not enabled"
        )
    if cfg.precedence[:1] != ("historical_archive",):
        raise ConfigError("[reconciliation].precedence must start with historical_archive (never overwritten)")
    if cfg.new_player_id_base < 1_000_000:
        raise ConfigError("[identity].new_player_id_base must be >= 1,000,000 to stay clear of archive IDs")
    if cfg.prospective_start < cfg.historical_snapshot_end:
        raise ConfigError("prospective_start must not precede the historical snapshot end")
    return cfg

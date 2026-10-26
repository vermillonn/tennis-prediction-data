"""Phase 4C configuration (config/evaluation.toml) and the frozen evaluation protocol.

The protocol is the set of rules fixed before any outcome is examined: cohort mode, approved model identity, forecast
selection, timing evidence, outcome scoring policy, clipping, bins, bootstrap and sample-size thresholds. It is stored
in every audit package as canonical JSON and identified by ``protocol_hash``; a changed rule needs a new version.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from tennis_data.config import PROJECT_ROOT, ConfigError
from tennis_data.ledger.hashing import content_hash
from tennis_data.ledger.storage import MODES
from tennis_data.ledger.writer import LedgerConfig, load_ledger_config

DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "config" / "evaluation.toml"
SELECTION_POLICIES = ("first_admissible_in_slot",)
OUTCOME_POLICIES = ("unanimous_latest_version",)
RETIREMENT_POLICIES = ("separate", "exclude")
SCORABLE_OUTCOMES = ("COMPLETED",)
SUBGROUPS = ("surface", "tournament_level", "best_of", "prior_service_points", "elo_experience", "probability_range",
             "freshness", "period")  # fmt: skip
CLUSTERS = ("week", "tournament")


@dataclass(frozen=True)
class Limits:
    max_export_bytes: int = 512 * 1024 * 1024
    max_line_bytes: int = 1024 * 1024
    max_outcome_records: int = 2_000_000
    max_receipt_bytes: int = 1024 * 1024


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: dict  # canonical, hashable protocol (see ``protocol_from``)
    ledger: LedgerConfig
    snapshot_dir: Path
    package_dir: Path
    reports_dir: Path
    evidence_dir: Path
    trusted_checkpoint_dir: Path
    limits: Limits
    publishers: dict
    config_path: Path

    @property
    def protocol_hash(self) -> str:
        return content_hash(self.protocol)


def protocol_from(pr: dict, ev: dict, ledger: LedgerConfig) -> dict:
    """The frozen protocol. The approved model identity comes from the Phase 4A gate config, not from this file."""
    p = {
        "version": str(pr.get("version", "pe-1")),
        "cohort_mode": str(pr.get("cohort_mode", "PROSPECTIVE")),
        "approved_model_id": str(pr.get("approved_model_id", "M4A+raw")),
        "approved_frozen_hash": ledger.prospective.expected_frozen_hash,
        "approved_configuration_hash": ledger.prospective.modeling.configuration_hash(),
        "primary_forecast_slot": str(pr.get("primary_forecast_slot", "pre_match")),
        "selection_policy": str(pr.get("selection_policy", "first_admissible_in_slot")),
        "require_actual_start_evidence": bool(pr.get("require_actual_start_evidence", True)),
        "scored_outcomes": list(pr.get("scored_outcomes", ["COMPLETED"])),
        "retirement_policy": str(pr.get("retirement_policy", "separate")),
        "outcome_policy": str(pr.get("outcome_policy", "unanimous_latest_version")),
        "probability_clip": float(pr.get("probability_clip", 1e-15)),
        "calibration_bins": int(ev.get("calibration_bins", 10)),
        "bootstrap_samples": int(ev.get("bootstrap_samples", 2000)),
        "bootstrap_seed": int(ev.get("bootstrap_seed", 20261025)),
        "clusters": list(ev.get("clusters", list(CLUSTERS))),
        "min_matches_inferential": int(ev.get("min_matches_inferential", 200)),
        "min_blocks_inferential": int(ev.get("min_blocks_inferential", 10)),
        "min_subgroup_matches": int(ev.get("min_subgroup_matches", 50)),
        "sample_size_edges": [int(x) for x in ev.get("sample_size_edges", [1, 1000, 5000])],
        "elo_experience_edges": [int(x) for x in ev.get("elo_experience_edges", [1, 20, 100])],
        "subgroups": list(ev.get("subgroups", list(SUBGROUPS))),
    }
    validate_protocol(p)
    return p


def validate_protocol(p: dict) -> None:
    def need(cond, msg):
        if not cond:
            raise ConfigError(f"evaluation protocol: {msg}")

    need(p["cohort_mode"] in MODES, f"cohort_mode must be one of {MODES}")
    need(p["selection_policy"] in SELECTION_POLICIES, f"selection_policy must be one of {SELECTION_POLICIES}")
    need(p["outcome_policy"] in OUTCOME_POLICIES, f"outcome_policy must be one of {OUTCOME_POLICIES}")
    need(p["retirement_policy"] in RETIREMENT_POLICIES, f"retirement_policy must be one of {RETIREMENT_POLICIES}")
    need(set(p["scored_outcomes"]) <= set(SCORABLE_OUTCOMES) and p["scored_outcomes"],
         "only COMPLETED outcomes can be scored as match-winner results")  # fmt: skip
    need(set(p["subgroups"]) <= set(SUBGROUPS), f"unknown subgroup (allowed: {SUBGROUPS})")
    need(set(p["clusters"]) <= set(CLUSTERS) and p["clusters"], f"clusters must be a subset of {CLUSTERS}")
    need(0 < p["probability_clip"] < 0.01, "probability_clip must be small and positive")
    need(p["calibration_bins"] >= 2 and p["bootstrap_samples"] >= 100, "too few bins or bootstrap samples")
    need(p["min_matches_inferential"] >= 30 and p["min_blocks_inferential"] >= 5,
         "inferential thresholds below 30 matches or 5 blocks are not allowed")  # fmt: skip
    need(len(p["approved_frozen_hash"]) >= 16, "approved frozen hash missing")


def load_evaluation_config(path: str | Path | None = None, root: str | Path | None = None,
                           ledger: LedgerConfig | None = None) -> EvaluationConfig:  # fmt: skip
    p = Path(path) if path else DEFAULT_EVALUATION_CONFIG
    if not p.exists():
        raise ConfigError(f"Evaluation config not found: {p}")
    base = Path(root) if root else PROJECT_ROOT
    with p.open("rb") as fh:
        d = tomllib.load(fh)

    def resolve(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else base / q

    lcfg = ledger or load_ledger_config(resolve(d.get("ledger_config", "config/ledger.toml")), root=base)
    o = d.get("outputs", {})
    lim = Limits(**{k: int(v) for k, v in d.get("limits", {}).items()})
    return EvaluationConfig(
        protocol=protocol_from(d.get("protocol", {}), d.get("evaluation", {}), lcfg),
        ledger=lcfg,
        snapshot_dir=resolve(o.get("snapshot_dir", "data/evaluation/snapshots")),
        package_dir=resolve(o.get("package_dir", "data/evaluation/packages")),
        reports_dir=resolve(o.get("reports_dir", "reports/evaluation")),
        evidence_dir=resolve(o.get("evidence_dir", "data/ledger/anchors")),
        trusted_checkpoint_dir=resolve(o.get("trusted_checkpoint_dir", "data/ledger/trusted_checkpoints")),
        limits=lim,
        publishers=d.get("publishers", {}),
        config_path=p.resolve(),
    )

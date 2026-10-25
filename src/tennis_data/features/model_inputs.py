"""Schema guard for predictive model matrices (Phase 3B and later).

Only pre-match information may enter a model matrix. Allowed:

- feature-table columns whose registry role is ``feature`` or ``context``
  (see ``dictionary.ROLES``), and
- the Phase 2 pre-match Elo state (``ELO_PREMATCH_FEATURES``), optionally.

Always rejected, even if a caller lists them explicitly (``FORBIDDEN``):

- the label and any outcome of the current match (winner, score, status, duration),
- the current match's own statistics (Phase 1 serve stats, Phase 3A point counts),
- post-match audit fields such as ``stats_eligible`` / ``stats_exclusion_reason``, eligibility flags
  derived from the outcome (``is_target``, ``is_update``) and ``quality_flags``,
- keys, time labels, evaluation-period labels and version metadata.

The audit fields stay in the source feature table; this guard only governs what a model may see.
"""

from __future__ import annotations

import polars as pl

from tennis_data.features.config import FeatureConfig
from tennis_data.features.dictionary import column_roles
from tennis_data.normalization.schema import STAT_COLUMNS

ALLOWED_ROLES = ("feature", "context")

ELO_PREMATCH_FEATURES = (
    "player_a_elo_before", "player_b_elo_before", "global_match_count_a", "global_match_count_b",
    "surface_elo_a_before", "surface_elo_b_before", "surface_match_count_a", "surface_match_count_b",
    "surface_effective_elo_a", "surface_effective_elo_b", "predicted_probability_a",
)  # fmt: skip

# Post-match / label / audit columns from any project table. Never model inputs.
OUTCOME_COLUMNS = (
    "target_a_wins", "player_a_won", "winner_id", "loser_id", "score", "match_status", "minutes",
    "stats_status", "quality_flags", "is_target", "is_update", "exclusion_reason",
    "stats_eligible", "stats_exclusion_reason", "release_batch", "evaluation_period", "period",
    "a_sp_played", "a_sp_won", "b_sp_played", "b_sp_won",
)  # fmt: skip
FORBIDDEN = frozenset(OUTCOME_COLUMNS) | frozenset(STAT_COLUMNS)


class ModelInputError(ValueError):
    """A model matrix would contain a forbidden or unregistered column."""


def allowed_model_inputs(cfg: FeatureConfig, include_elo: bool = True) -> tuple[str, ...]:
    roles = column_roles(cfg)
    cols = [c for c, r in roles.items() if r in ALLOWED_ROLES and c not in FORBIDDEN]
    if include_elo:
        cols += [c for c in ELO_PREMATCH_FEATURES if c not in FORBIDDEN]
    return tuple(cols)


def check_model_inputs(columns, cfg: FeatureConfig, include_elo: bool = True) -> None:
    columns = list(columns)
    forbidden = sorted(c for c in columns if c in FORBIDDEN)
    allowed = set(allowed_model_inputs(cfg, include_elo))
    unknown = sorted(c for c in columns if c not in allowed and c not in FORBIDDEN)
    if forbidden or unknown:
        parts = []
        if forbidden:
            parts.append(f"post-match/label/audit columns are forbidden as model inputs: {forbidden}")
        if unknown:
            parts.append(f"columns not on the model-input allowlist: {unknown}")
        raise ModelInputError("; ".join(parts))


def model_matrix(df: pl.DataFrame, cfg: FeatureConfig, columns=None, include_elo: bool = True) -> pl.DataFrame:
    """Select model inputs from ``df``. Default: every allowlisted column present in ``df``."""
    if columns is None:
        columns = [c for c in allowed_model_inputs(cfg, include_elo) if c in df.columns]
    check_model_inputs(columns, cfg, include_elo)
    return df.select(columns)

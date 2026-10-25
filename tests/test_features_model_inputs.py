"""Model-input allowlist: post-match, label and audit columns can never enter a model matrix."""

from __future__ import annotations

import polars as pl
import pytest

from feature_helpers import cfg, match, run
from tennis_data.features.config import load_feature_config
from tennis_data.features.dictionary import column_roles
from tennis_data.features.model_inputs import (
    ELO_PREMATCH_FEATURES,
    FORBIDDEN,
    ModelInputError,
    allowed_model_inputs,
    check_model_inputs,
    model_matrix,
)
from tennis_data.normalization.schema import CANONICAL_SCHEMA, STAT_COLUMNS

CFG = load_feature_config()

MUST_BE_FORBIDDEN = [
    "stats_eligible", "stats_exclusion_reason", "target_a_wins", "player_a_won", "winner_id", "score",
    "match_status", "minutes", "is_target", "is_update", "quality_flags", "stats_status",
    "a_sp_played", "a_sp_won", "b_sp_played", "b_sp_won", "player_a_svpt", "player_b_bp_faced", "player_a_first_won",
]  # fmt: skip


def test_allowlist_is_exactly_pre_match_features_context_and_elo_state():
    roles = column_roles(CFG)
    allowed = set(allowed_model_inputs(CFG))
    expected = {c for c, r in roles.items() if r in ("feature", "context")} | set(ELO_PREMATCH_FEATURES)
    assert allowed == expected
    assert not allowed & FORBIDDEN
    for audit in ("stats_eligible", "adjustment_fit_batch"):
        assert roles[audit] == "audit" and audit not in allowed
    for key in ("match_id", "prediction_batch", "tournament_date", "season", "player_a_id", "configuration_hash"):
        assert key not in allowed
    assert "target_a_wins" not in allowed_model_inputs(CFG, include_elo=True)


@pytest.mark.parametrize("col", MUST_BE_FORBIDDEN)
def test_forbidden_columns_are_rejected_even_when_listed_explicitly(col):
    with pytest.raises(ModelInputError, match="forbidden"):
        check_model_inputs(["a_spw_shrunk", col], CFG)


def test_unregistered_columns_are_rejected():
    with pytest.raises(ModelInputError, match="allowlist"):
        check_model_inputs(["a_spw_shrunk", "some_new_column"], CFG)
    check_model_inputs(["a_spw_shrunk", "spw_diff", "surface", "predicted_probability_a"], CFG)  # fine
    with pytest.raises(ModelInputError):
        check_model_inputs(["predicted_probability_a"], CFG, include_elo=False)


def test_forbidden_current_match_stats_exist_in_phase1_schema():
    assert set(STAT_COLUMNS) <= set(CANONICAL_SCHEMA)
    assert set(STAT_COLUMNS) <= FORBIDDEN


def test_model_matrix_drops_audit_fields_but_source_table_keeps_them():
    table = run([match("m1", 0, 100, 200), match("m2", 1, 100, 300)], cfg(opponent=True))
    assert "stats_eligible" in table.columns and "adjustment_fit_batch" in table.columns  # audit kept at source
    joined = table.with_columns(
        pl.lit(True).alias("target_a_wins"), pl.lit(0.6).alias("predicted_probability_a"),
        pl.lit("tag").alias("stats_exclusion_reason"),
    )  # fmt: skip
    X = model_matrix(joined, CFG)
    assert "predicted_probability_a" in X.columns
    for col in ("stats_eligible", "stats_exclusion_reason", "target_a_wins", "adjustment_fit_batch", "match_id"):
        assert col not in X.columns
    with pytest.raises(ModelInputError):
        model_matrix(joined, CFG, columns=["a_spw_shrunk", "stats_eligible"])

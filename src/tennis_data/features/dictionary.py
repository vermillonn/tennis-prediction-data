"""Feature registry: every exported column with dtype, formula, units, null policy and dependencies.

The output table is cast to this schema, docs/feature_dictionary.md is generated from it, and a test
fails if any exported column is undocumented or the generated dictionary is stale.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from tennis_data.config import PROJECT_ROOT
from tennis_data.features.config import FeatureConfig

DICTIONARY_PATH = PROJECT_ROOT / "docs" / "feature_dictionary.md"

TIMING = "state frozen at the start of prediction_batch; observations released in earlier batches only"
PROB = "probability (0-1)"
POINTS = "points (count)"


@dataclass(frozen=True)
class Spec:
    name: str
    dtype: pl.DataType
    formula: str
    units: str
    nulls: str
    deps: str
    group: str


def _fmt_h(h: float) -> str:
    return f"{h:g}"


def player_specs(cfg: FeatureConfig) -> list[Spec]:
    """Per-player features; exported once with prefix ``a_`` and once with ``b_``."""
    k = cfg.shrinkage
    no_hist = "null when the player has no released eligible statistics"
    no_prior = f"null until the population prior has >= {k.min_prior_points} released service points"
    specs = [
        Spec(
            "has_stats_history",
            pl.Boolean,
            "service_points_observed > 0",
            "flag",
            "never null",
            "observations",
            "coverage",
        ),
        Spec(
            "stat_matches",
            pl.Int64,
            "count of released stats-eligible matches",
            "matches",
            "0 when none",
            "observations",
            "coverage",
        ),
        Spec(
            "weeks_since_last_stats",
            pl.Int64,
            "(prediction_batch - event start batch of latest observation) / 7",
            "weeks",
            no_hist,
            "observations",
            "coverage",
        ),
        Spec(
            "service_points_observed",
            pl.Int64,
            "sum svpt over released observations",
            POINTS,
            "0 when none",
            "svpt",
            "overall",
        ),
        Spec(
            "service_points_won", pl.Int64, "sum (1stWon + 2ndWon)", POINTS, "0 when none", "1stWon, 2ndWon", "overall"
        ),
        Spec(
            "return_points_observed", pl.Int64, "sum opponent svpt", POINTS, "0 when none", "opponent svpt", "overall"
        ),
        Spec(
            "return_points_won",
            pl.Int64,
            "sum (opponent svpt - opponent 1stWon - opponent 2ndWon)",
            POINTS,
            "0 when none",
            "opponent svpt, 1stWon, 2ndWon",
            "overall",
        ),
        Spec(
            "spw_raw",
            pl.Float64,
            "service_points_won / service_points_observed",
            PROB,
            no_hist,
            "overall counts",
            "overall",
        ),
        Spec(
            "rpw_raw",
            pl.Float64,
            "return_points_won / return_points_observed",
            PROB,
            no_hist,
            "overall counts",
            "overall",
        ),
        Spec(
            "prior_spw",
            pl.Float64,
            "population service-point win rate over released observations (pool)",
            PROB,
            no_prior,
            "all released observations",
            "prior",
        ),
        Spec(
            "prior_rpw",
            pl.Float64,
            "1 - prior_spw (every point is one player's serve and the other's return)",
            PROB,
            no_prior,
            "prior_spw",
            "prior",
        ),
        Spec(
            "spw_shrunk",
            pl.Float64,
            f"(service_points_won + {k.kappa_service:g} * prior_spw) / (service_points_observed + {k.kappa_service:g})",
            PROB,
            f"{no_prior}; equals prior_spw with no history",
            "overall counts, prior_spw",
            "shrunk",
        ),
        Spec(
            "rpw_shrunk",
            pl.Float64,
            f"(return_points_won + {k.kappa_return:g} * prior_rpw) / (return_points_observed + {k.kappa_return:g})",
            PROB,
            f"{no_prior}; equals prior_rpw with no history",
            "overall counts, prior_rpw",
            "shrunk",
        ),
        Spec(
            "spw_prior_weight",
            pl.Float64,
            f"{k.kappa_service:g} / (service_points_observed + {k.kappa_service:g})",
            "share (0-1)",
            "never null",
            "service_points_observed",
            "shrunk",
        ),
        Spec(
            "rpw_prior_weight",
            pl.Float64,
            f"{k.kappa_return:g} / (return_points_observed + {k.kappa_return:g})",
            "share (0-1)",
            "never null",
            "return_points_observed",
            "shrunk",
        ),
        Spec(
            "service_prior_dominated",
            pl.Boolean,
            "spw_prior_weight > 0.5 (fewer observed points than kappa)",
            "flag",
            "never null",
            "spw_prior_weight",
            "shrunk",
        ),
        Spec(
            "return_prior_dominated",
            pl.Boolean,
            "rpw_prior_weight > 0.5",
            "flag",
            "never null",
            "rpw_prior_weight",
            "shrunk",
        ),
        Spec(
            "surface_service_points",
            pl.Int64,
            "service points on the match surface",
            POINTS,
            "null when the match surface is missing",
            "svpt, surface",
            "surface",
        ),
        Spec(
            "surface_return_points",
            pl.Int64,
            "return points on the match surface",
            POINTS,
            "null when the match surface is missing",
            "opponent svpt, surface",
            "surface",
        ),
        Spec(
            "surface_stat_matches",
            pl.Int64,
            "stats-eligible matches on the match surface",
            "matches",
            "null when the match surface is missing",
            "surface",
            "surface",
        ),
        Spec(
            "surface_spw_raw",
            pl.Float64,
            "surface service points won / surface_service_points",
            PROB,
            "null with no surface points or missing surface",
            "surface counts",
            "surface",
        ),
        Spec(
            "surface_rpw_raw",
            pl.Float64,
            "surface return points won / surface_return_points",
            PROB,
            "null with no surface points or missing surface",
            "surface counts",
            "surface",
        ),
        Spec(
            "surface_prior_spw",
            pl.Float64,
            "population service-point win rate on the match surface",
            PROB,
            f"{no_prior} (per surface); null for missing surface",
            "released observations on surface",
            "prior",
        ),
        Spec(
            "surface_spw",
            pl.Float64,
            f"(surface won + {k.kappa_surface_service:g} * base) / (surface points + {k.kappa_surface_service:g}); "
            "base = sigmoid(logit(off-surface shrunk SPW) + logit(surface_prior_spw) - logit(prior_spw)) [heuristic]",
            PROB,
            "null when priors unavailable or surface missing",
            "surface and off-surface counts, priors",
            "surface",
        ),
        Spec(
            "surface_rpw",
            pl.Float64,
            "as surface_spw for return points (priors 1 - SPW priors, kappa_surface_return)",
            PROB,
            "null when priors unavailable or surface missing",
            "surface and off-surface counts, priors",
            "surface",
        ),
        Spec(
            "surface_prior_weight",
            pl.Float64,
            f"{k.kappa_surface_service:g} / (surface_service_points + {k.kappa_surface_service:g})",
            "share (0-1)",
            "null for missing surface",
            "surface_service_points",
            "surface",
        ),
    ]
    for n in cfg.last_matches:
        specs += [
            Spec(
                f"spw_last{n}",
                pl.Float64,
                f"service rate over the {n} most recent eligible matches (point-weighted)",
                PROB,
                no_hist,
                "recent observations",
                "window",
            ),
            Spec(
                f"rpw_last{n}",
                pl.Float64,
                f"return rate over the {n} most recent eligible matches",
                PROB,
                no_hist,
                "recent observations",
                "window",
            ),
            Spec(
                f"service_points_last{n}",
                pl.Int64,
                "service points in those matches",
                POINTS,
                "0 when none",
                "recent observations",
                "window",
            ),
            Spec(
                f"return_points_last{n}",
                pl.Int64,
                "return points in those matches",
                POINTS,
                "0 when none",
                "recent observations",
                "window",
            ),
        ]
    for w in cfg.last_weeks:
        specs += [
            Spec(
                f"spw_last{w}w",
                pl.Float64,
                f"service rate over observations whose event started in the last {w} batches",
                PROB,
                "null when no observation in the window",
                "recent observations",
                "window",
            ),
            Spec(
                f"rpw_last{w}w",
                pl.Float64,
                "return rate over the same window",
                PROB,
                "null when no observation in the window",
                "recent observations",
                "window",
            ),
            Spec(
                f"service_points_last{w}w",
                pl.Int64,
                "service points in the window",
                POINTS,
                "0 when none",
                "recent observations",
                "window",
            ),
            Spec(
                f"return_points_last{w}w",
                pl.Int64,
                "return points in the window",
                POINTS,
                "0 when none",
                "recent observations",
                "window",
            ),
        ]
    for h in cfg.half_life_weeks:
        t = _fmt_h(h)
        specs += [
            Spec(
                f"spw_decay{t}w",
                pl.Float64,
                f"sum(w_i * won_i) / sum(w_i * played_i), w_i = 2^(-age_weeks/{t})",
                PROB,
                no_hist,
                "observations, event start batches",
                "decay",
            ),
            Spec(f"rpw_decay{t}w", pl.Float64, "same for return points", PROB, no_hist, "observations", "decay"),
            Spec(
                f"service_points_decay{t}w",
                pl.Float64,
                "sum(w_i * service points_i): total decay-weighted points. NOT an effective sample size; "
                "see service_ess_decay",
                "weighted points",
                "0 when none",
                "observations",
                "decay",
            ),
            Spec(
                f"return_points_decay{t}w",
                pl.Float64,
                "sum(w_i * return points_i): total decay-weighted points (not an effective sample size)",
                "weighted points",
                "0 when none",
                "observations",
                "decay",
            ),
            Spec(
                f"service_ess_decay{t}w",
                pl.Float64,
                "Kish effective sample size of the decayed service rate: (sum w_i n_i)^2 / sum w_i^2 n_i, "
                "n_i = service points of match i; assumes independent points (an upper bound under "
                "within-match dependence)",
                "effective points",
                "0 when none",
                "observations",
                "decay",
            ),
            Spec(
                f"return_ess_decay{t}w",
                pl.Float64,
                "Kish effective sample size of the decayed return rate (as service_ess_decay, return points)",
                "effective points",
                "0 when none",
                "observations",
                "decay",
            ),
            Spec(
                f"matches_ess_decay{t}w",
                pl.Float64,
                "Kish effective number of matches: (sum w_i)^2 / sum w_i^2 (each match one unit; robust to "
                "within-match dependence)",
                "effective matches",
                "0 when none",
                "observations",
                "decay",
            ),
        ]
    if cfg.opponent.enabled:
        o = cfg.opponent
        nofit = f"null until the first fit (>= {o.min_fit_points} released points)"
        specs += [
            Spec(
                "serve_strength",
                pl.Float64,
                "alpha_i: overall serving strength (logit scale)",
                "logit",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "return_strength",
                pl.Float64,
                "beta_i: overall returning strength (logit scale)",
                "logit",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "serve_adjusted",
                pl.Float64,
                "sigmoid(mu_overall + alpha_i): SPW vs an average returner",
                PROB,
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "return_adjusted",
                pl.Float64,
                "1 - sigmoid(mu_overall - beta_i): RPW vs an average server",
                PROB,
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "surface_serve_adjusted",
                pl.Float64,
                "sigmoid(mu_s + alpha_i + a_is) on the match surface",
                PROB,
                f"{nofit}; null for missing surface",
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "surface_return_adjusted",
                pl.Float64,
                "1 - sigmoid(mu_s - beta_i - b_is) on the match surface",
                PROB,
                f"{nofit}; null for missing surface",
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "adjusted_serve_points_weighted",
                pl.Float64,
                "sum of decay-weighted service points of this player in the fit (not an effective sample size)",
                "weighted points",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "adjusted_return_points_weighted",
                pl.Float64,
                "decay-weighted return points of this player in the fit",
                "weighted points",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "adjusted_serve_ess",
                pl.Float64,
                "Kish effective service points in the fit: (sum w n)^2 / sum w^2 n (independent-points assumption)",
                "effective points",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "adjusted_return_ess",
                pl.Float64,
                "Kish effective return points in the fit",
                "effective points",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
            Spec(
                "adjusted_in_fit",
                pl.Boolean,
                "player has any weighted points in the fit (false: population fallback)",
                "flag",
                nofit,
                "opponent model fit",
                "adjusted",
            ),
        ]
    return specs


def match_specs(cfg: FeatureConfig) -> list[Spec]:
    key = [
        Spec("match_id", pl.String, "Phase 1 match_id", "id", "never null", "validated table", "key"),
        Spec(
            "prediction_batch",
            pl.Date,
            "Phase 2 batch start (Saturday) whose frozen state produced the row; a batch label, not a match time",
            "date",
            "never null",
            "Phase 2 chronology",
            "key",
        ),
        Spec(
            "tournament_date",
            pl.Date,
            "event start date (not the match date)",
            "date",
            "never null",
            "validated table",
            "key",
        ),
        Spec("season", pl.Int32, "source season", "year", "never null", "validated table", "key"),
        Spec("pool", pl.String, "Phase 2 rating pool", "id", "never null", "Phase 2 pools", "key"),
        Spec("tournament_level", pl.String, "Phase 1 tournament_level", "code", "never null", "validated table", "key"),
        Spec("surface", pl.String, "Phase 1 surface", "code", "null when missing in source", "validated table", "key"),
        Spec(
            "player_a_id",
            pl.Int64,
            "Phase 1 player A (winner-independent)",
            "id",
            "never null",
            "validated table",
            "key",
        ),
        Spec("player_b_id", pl.Int64, "Phase 1 player B", "id", "never null", "validated table", "key"),
        Spec(
            "stats_eligible",
            pl.Boolean,
            "this match's own statistics are eligible to enter LATER snapshots (never used in its own row)",
            "flag",
            "never null",
            "observations.stats_eligibility",
            "key",
        ),
    ]
    players = [
        Spec(f"{side}_{s.name}", s.dtype, s.formula, s.units, s.nulls, s.deps, s.group)
        for side in ("a", "b")
        for s in player_specs(cfg)
    ]
    diffs = [
        Spec(
            "spw_diff",
            pl.Float64,
            "a_spw_shrunk - b_spw_shrunk",
            "probability difference",
            "null if either null",
            "shrunk",
            "diff",
        ),
        Spec(
            "rpw_diff",
            pl.Float64,
            "a_rpw_shrunk - b_rpw_shrunk",
            "probability difference",
            "null if either null",
            "shrunk",
            "diff",
        ),
        Spec(
            "surface_spw_diff",
            pl.Float64,
            "a_surface_spw - b_surface_spw",
            "probability difference",
            "null if either null",
            "surface",
            "diff",
        ),
        Spec(
            "surface_rpw_diff",
            pl.Float64,
            "a_surface_rpw - b_surface_rpw",
            "probability difference",
            "null if either null",
            "surface",
            "diff",
        ),
    ]
    if cfg.opponent.enabled:
        diffs += [
            Spec(
                "serve_adjusted_diff",
                pl.Float64,
                "a_serve_adjusted - b_serve_adjusted",
                "probability difference",
                "null if either null",
                "adjusted",
                "diff",
            ),
            Spec(
                "return_adjusted_diff",
                pl.Float64,
                "a_return_adjusted - b_return_adjusted",
                "probability difference",
                "null if either null",
                "adjusted",
                "diff",
            ),
            Spec(
                "a_spw_vs_b",
                pl.Float64,
                "sigmoid(mu_s + alpha_a + a_as - beta_b - b_bs): expected SPW of A against B "
                "(overall baseline and no surface terms when surface is missing)",
                PROB,
                "null before first fit",
                "adjusted",
                "matchup",
            ),
            Spec(
                "b_spw_vs_a",
                pl.Float64,
                "same for B serving against A",
                PROB,
                "null before first fit",
                "adjusted",
                "matchup",
            ),
            Spec(
                "matchup_spw_diff",
                pl.Float64,
                "a_spw_vs_b - b_spw_vs_a",
                "probability difference",
                "null before first fit",
                "adjusted",
                "matchup",
            ),
            Spec(
                "adjustment_fit_batch",
                pl.Date,
                "batch at which the opponent model used for this row was fitted (on observations released before it)",
                "date",
                "null before first fit",
                "opponent model",
                "adjusted",
            ),
        ]
    meta = [
        Spec("feature_version", pl.String, "feature_version + engine version", "id", "never null", "config", "meta"),
        Spec(
            "configuration_hash",
            pl.String,
            "sha256 of feature settings, chronology, pools, targets, source commit",
            "id",
            "never null",
            "config",
            "meta",
        ),
    ]
    return key + players + diffs + meta


# Roles of the non-feature columns. Everything else in the registry is a pre-match "feature".
#   key:     identifiers and time labels (join keys; never model inputs)
#   context: pre-match match context, allowed as model inputs
#   audit:   provenance / post-match metadata kept for auditing; NEVER model inputs
#   meta:    versioning
ROLES = {
    "match_id": "key", "prediction_batch": "key", "tournament_date": "key", "season": "key", "pool": "key",
    "player_a_id": "key", "player_b_id": "key",
    "tournament_level": "context", "surface": "context",
    "stats_eligible": "audit", "adjustment_fit_batch": "audit",
    "feature_version": "meta", "configuration_hash": "meta",
}  # fmt: skip


def role_of(name: str) -> str:
    return ROLES.get(name, "feature")


def column_roles(cfg: FeatureConfig) -> dict[str, str]:
    return {s.name: role_of(s.name) for s in match_specs(cfg)}


def schema(cfg: FeatureConfig) -> dict[str, pl.DataType]:
    return {s.name: s.dtype for s in match_specs(cfg)}


def render_dictionary(cfg: FeatureConfig) -> str:
    lines = [
        "# Feature dictionary (Phase 3A serve/return)",
        "",
        "_Generated by `tennis-data features-docs` from `src/tennis_data/features/dictionary.py` for "
        "`config/features.toml`. Do not edit by hand; a test fails when it is stale._",
        "",
        f"Timestamp policy for every player feature: {TIMING}. Tournament dates are event start dates; no "
        "match dates exist. Rates are point-weighted (sum won / sum played). Missing statistics are never "
        "treated as zero.",
        "",
        "Player features are exported twice, `a_<name>` and `b_<name>` (Phase 1 winner-independent order).",
        "",
        "**Roles.** `feature` and `context` columns are the only ones allowed in a predictive model matrix "
        "(`tennis_data.features.model_inputs`). `key` columns are join keys and time labels. `audit` columns "
        "describe the current match after the fact (e.g. `stats_eligible`) or provenance, and must never be "
        "model inputs. All player features have role `feature`.",
        "",
        "## Match-level columns",
        "",
        "| column | role | type | formula | units | nulls | depends on |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in match_specs(cfg):
        if s.name.startswith(("a_", "b_")) and s.group not in ("matchup",):
            continue
        lines.append(f"| `{s.name}` | {role_of(s.name)} | {s.dtype} | {s.formula} | {s.units} | {s.nulls} | {s.deps} |")
    lines += ["", "## Player features (`a_` / `b_` prefix)", "", "| feature | group | type | formula | units | nulls | depends on |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for s in player_specs(cfg):
        lines.append(f"| `{s.name}` | {s.group} | {s.dtype} | {s.formula} | {s.units} | {s.nulls} | {s.deps} |")
    return "\n".join(lines) + "\n"


def write_dictionary(cfg: FeatureConfig, path=DICTIONARY_PATH):
    path.write_text(render_dictionary(cfg))
    return path

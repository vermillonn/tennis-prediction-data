"""Batch Elo engine: frozen pre-batch predictions, simultaneous order-independent updates.

Per batch b (in date order):

1. Freeze. Every match with prediction_batch == b is predicted from the state left by the
   closes of batches < b. Nothing from batch b is visible.
2. Close. Every update-eligible match with release_batch == b contributes, for player A,
       delta = K_match * (S_a - E_a),   E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400)),
   and -delta for player B, with R and the match counts that K depends on all taken from the
   state frozen at the start of the close. Each player's change is the correctly rounded sum
   (math.fsum) of their deltas, then all changes are applied at once.

Mathematical assumptions of the close: within one batch, a player's strength is constant, so all of
the batch's results are evaluated against the same frozen ratings (a single simultaneous
gradient step on the Elo log-likelihood rather than a chain of sequential steps). Because each
delta depends only on the frozen state and fsum is exact up to a single final rounding, the result
is bit-for-bit independent of the order of the matches. Every match adds +delta and -delta, so the
total rating mass of a pool is conserved (zero-sum) whenever K_match is the same for both players,
which holds in both K modes.

Surface ratings are independent Elo pools per surface key (Hard, Clay, Grass, Carpet), updated
the same way with surface ratings, surface match counts and surface expected scores. They are
blended with the global rating only at prediction time (see ``surface_probability``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import polars as pl

from tennis_data.elo.config import EloParams


def expected_score(r_a: float, r_b: float) -> float:
    """Standard Elo win probability of A against B."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def k_factor(params: EloParams, n: int) -> float:
    if params.k_mode == "constant":
        return params.k_constant
    return params.k_numerator / (n + params.k_offset) ** params.k_shape


def match_k(params: EloParams, n_a: int, n_b: int) -> float:
    """Shared K for one match, so the update is zero-sum."""
    if params.k_mode == "constant":
        return params.k_constant
    return 0.5 * (k_factor(params, n_a) + k_factor(params, n_b))


def surface_key(surface: str | None, carpet_policy: str) -> str | None:
    if surface is None:
        return None
    if surface == "Carpet":
        return {"separate": "Carpet", "hard": "Hard", "none": None}[carpet_policy]
    return surface


@dataclass
class _Pool:
    rating: dict
    count: dict
    s_rating: dict  # (surface, player) -> rating, only once initialized
    s_count: dict  # (surface, player) -> matches

    @classmethod
    def new(cls) -> _Pool:
        return cls({}, defaultdict(int), {}, defaultdict(int))


ENGINE_INPUT = (
    "match_id", "pool", "prediction_batch", "release_batch", "player_a_id", "player_b_id",
    "target_a_wins", "surface_key", "is_target", "is_update",
)  # fmt: skip

PREDICTION_SCHEMA = {
    "match_id": pl.String,
    "player_a_elo_before": pl.Float64,
    "player_b_elo_before": pl.Float64,
    "global_match_count_a": pl.Int64,
    "global_match_count_b": pl.Int64,
    "surface_elo_a_before": pl.Float64,
    "surface_elo_b_before": pl.Float64,
    "surface_match_count_a": pl.Int64,
    "surface_match_count_b": pl.Int64,
    "global_probability_a": pl.Float64,
}

HISTORY_SCHEMA = {
    "batch": pl.Date,
    "pool": pl.String,
    "player_id": pl.Int64,
    "rating_type": pl.String,  # "global" or a surface key
    "rating_after": pl.Float64,
    "match_count_after": pl.Int64,
}


@dataclass
class EloRun:
    predictions: pl.DataFrame  # one row per input row with is_target
    history: pl.DataFrame  # rating state after each close, for the players it changed


def run_elo(
    matches: pl.DataFrame, params: EloParams, record_history: bool = False, canonical_order: bool = True
) -> EloRun:
    """Replay ``matches`` (columns ENGINE_INPUT) batch by batch.

    Results do not depend on input row order. ``canonical_order=False`` processes rows in the order
    given (used by tests to show the arithmetic itself is order-independent); outputs are sorted either way.
    """
    missing = [c for c in ENGINE_INPUT if c not in matches.columns]
    if missing:
        raise ValueError(f"engine input is missing columns {missing}")
    if matches.filter(pl.col("release_batch") < pl.col("prediction_batch")).height:
        raise ValueError("release_batch earlier than prediction_batch")

    selected = matches.select(ENGINE_INPUT)
    rows = (selected.sort("match_id") if canonical_order else selected).to_dicts()
    predict_at: dict = defaultdict(list)
    release_at: dict = defaultdict(list)
    for r in rows:
        if r["is_target"]:
            predict_at[r["prediction_batch"]].append(r)
        if r["is_update"]:
            release_at[r["release_batch"]].append(r)

    init = params.initial_rating
    pools: dict[str, _Pool] = defaultdict(_Pool.new)
    out: list[tuple] = []
    history: list[tuple] = []

    def surface_rating(p: _Pool, s: str, pid: int) -> float:
        if (s, pid) in p.s_rating:
            return p.s_rating[(s, pid)]
        return init if params.surface_init == "constant" else p.rating.get(pid, init)

    for batch in sorted(set(predict_at) | set(release_at)):
        # 1. Freeze and predict.
        for r in predict_at.get(batch, ()):
            p = pools[r["pool"]]
            a, b, s = r["player_a_id"], r["player_b_id"], r["surface_key"]
            ga, gb = p.rating.get(a, init), p.rating.get(b, init)
            if s is None:
                sa = sb = None
                na_s = nb_s = 0
            else:
                sa, sb = surface_rating(p, s, a), surface_rating(p, s, b)
                na_s, nb_s = p.s_count[(s, a)], p.s_count[(s, b)]
            out.append((r["match_id"], ga, gb, p.count[a], p.count[b], sa, sb, na_s, nb_s, expected_score(ga, gb)))

        # 2. Close: simultaneous update from the frozen state.
        closing = release_at.get(batch, ())
        if not closing:
            continue
        g_deltas: dict = defaultdict(list)
        g_games: dict = defaultdict(int)
        s_deltas: dict = defaultdict(list)
        s_games: dict = defaultdict(int)
        s_base: dict = {}
        for r in closing:
            pname, p = r["pool"], pools[r["pool"]]
            a, b = r["player_a_id"], r["player_b_id"]
            score_a = 1.0 if r["target_a_wins"] else 0.0
            ga, gb = p.rating.get(a, init), p.rating.get(b, init)
            d = match_k(params, p.count[a], p.count[b]) * (score_a - expected_score(ga, gb))
            g_deltas[(pname, a)].append(d)
            g_deltas[(pname, b)].append(-d)
            g_games[(pname, a)] += 1
            g_games[(pname, b)] += 1
            s = r["surface_key"]
            if s is not None:
                sa, sb = surface_rating(p, s, a), surface_rating(p, s, b)
                s_base.setdefault((pname, s, a), sa)
                s_base.setdefault((pname, s, b), sb)
                ds = match_k(params, p.s_count[(s, a)], p.s_count[(s, b)]) * (score_a - expected_score(sa, sb))
                s_deltas[(pname, s, a)].append(ds)
                s_deltas[(pname, s, b)].append(-ds)
                s_games[(pname, s, a)] += 1
                s_games[(pname, s, b)] += 1
        for (pname, pid), ds in g_deltas.items():
            p = pools[pname]
            p.rating[pid] = p.rating.get(pid, init) + math.fsum(ds)
            p.count[pid] += g_games[(pname, pid)]
            if record_history:
                history.append((batch, pname, pid, "global", p.rating[pid], p.count[pid]))
        for (pname, s, pid), ds in s_deltas.items():
            p = pools[pname]
            p.s_rating[(s, pid)] = s_base[(pname, s, pid)] + math.fsum(ds)
            p.s_count[(s, pid)] += s_games[(pname, s, pid)]
            if record_history:
                history.append((batch, pname, pid, s, p.s_rating[(s, pid)], p.s_count[(s, pid)]))

    predictions = pl.DataFrame(out, schema=PREDICTION_SCHEMA, orient="row").sort("match_id")
    hist = pl.DataFrame(history, schema=HISTORY_SCHEMA, orient="row").sort("batch", "pool", "player_id", "rating_type")
    return EloRun(predictions, hist)


def surface_weight(count: pl.Expr, shrinkage: float) -> pl.Expr:
    """n / (n + m); 0 when the player has no prior matches on the surface."""
    n = count.cast(pl.Float64)
    return pl.when(n > 0).then(n / (n + shrinkage)).otherwise(0.0)


def surface_probability(df: pl.DataFrame, shrinkage: float) -> pl.DataFrame:
    """Add shrunk effective surface ratings and the surface-model probability.

    effective = w * surface_rating + (1 - w) * global_rating,  w = n_surface / (n_surface + m).
    Matches without a surface key (missing surface, or carpet under carpet_policy = "none") use w = 0,
    i.e. the global rating.
    """
    wa = surface_weight(pl.col("surface_match_count_a"), shrinkage)
    wb = surface_weight(pl.col("surface_match_count_b"), shrinkage)
    ea = (
        pl.when(pl.col("surface_elo_a_before").is_null())
        .then(pl.col("player_a_elo_before"))
        .otherwise(wa * pl.col("surface_elo_a_before") + (1 - wa) * pl.col("player_a_elo_before"))
    )
    eb = (
        pl.when(pl.col("surface_elo_b_before").is_null())
        .then(pl.col("player_b_elo_before"))
        .otherwise(wb * pl.col("surface_elo_b_before") + (1 - wb) * pl.col("player_b_elo_before"))
    )
    return df.with_columns(ea.alias("surface_effective_elo_a"), eb.alias("surface_effective_elo_b")).with_columns(
        (1.0 / (1.0 + 10.0 ** ((pl.col("surface_effective_elo_b") - pl.col("surface_effective_elo_a")) / 400.0))).alias(
            "surface_probability_a"
        )
    )

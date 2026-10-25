"""Rate estimators. All rates are point-weighted (sum of won / sum of played), never averages of
match percentages. Functions return None rather than 0 when there is no evidence.
"""

from __future__ import annotations

import math


def rate(won: float, played: float) -> float | None:
    """Raw rate; None when nothing was played (never a misleading 0%)."""
    return won / played if played > 0 else None


def shrunk(won: float, played: float, prior: float | None, kappa: float) -> float | None:
    """Beta-binomial posterior mean (won + kappa * prior) / (played + kappa).

    Equivalent to a Beta(kappa * prior, kappa * (1 - prior)) prior on the player's point-win
    probability: kappa is the prior's weight in points. Zero evidence returns the prior itself.
    """
    if prior is None:
        return None
    denom = played + kappa
    return (won + kappa * prior) / denom if denom > 0 else None


def prior_weight(played: float, kappa: float) -> float:
    """Share of the shrunk estimate that comes from the prior: kappa / (played + kappa)."""
    return kappa / (played + kappa) if played + kappa > 0 else 1.0


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def surface_estimate(
    won_surface: float,
    played_surface: float,
    won_all: float,
    played_all: float,
    prior_all: float | None,
    prior_surface: float | None,
    kappa_overall: float,
    kappa_surface: float,
) -> float | None:
    """Surface-specific rate shrunk toward the player's off-surface ability (heuristic blend).

    1. Off-surface evidence only: p_off = shrunk(won_all - won_surface, played_all - played_surface,
       prior_all, kappa_overall). Excluding the surface's own points keeps every point counted once.
    2. Transport p_off to this surface with the population surface effect:
       base = sigmoid(logit(p_off) + logit(prior_surface) - logit(prior_all)).
    3. Shrink the surface points toward it: (won_surface + kappa_surface * base) / (played_surface + kappa_surface).

    With no surface points the result is ``base``; with many it converges to the raw surface rate.
    This is an empirical-Bayes style heuristic, not a fitted hierarchical model.
    """
    if prior_all is None or prior_surface is None:
        return None
    p_off = shrunk(won_all - won_surface, played_all - played_surface, prior_all, kappa_overall)
    base = sigmoid(logit(p_off) + logit(prior_surface) - logit(prior_all))
    return shrunk(won_surface, played_surface, base, kappa_surface)


def method_of_moments_kappa(won: list[float], played: list[float]) -> dict:
    """Beta-binomial prior strength from per-player totals (diagnostic; used to set config defaults).

    tau^2 = Var(observed rates) - mean binomial variance; kappa = mu (1 - mu) / tau^2 - 1.
    """
    rates = [w / n for w, n in zip(won, played, strict=True) if n > 0]
    ns = [n for n in played if n > 0]
    k = len(rates)
    if k < 2:
        return {"players": k, "mu": None, "tau": None, "kappa": None}
    mu = math.fsum(won) / math.fsum(played)
    var_obs = math.fsum((r - mu) ** 2 for r in rates) / (k - 1)
    var_binom = math.fsum(mu * (1 - mu) / n for n in ns) / k
    tau2 = var_obs - var_binom
    kappa = mu * (1 - mu) / tau2 - 1 if tau2 > 0 else None
    return {"players": k, "mu": mu, "tau": math.sqrt(tau2) if tau2 > 0 else None, "kappa": kappa}

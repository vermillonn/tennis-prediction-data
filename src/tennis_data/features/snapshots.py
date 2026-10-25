"""Point-in-time player serve/return snapshots, batch by batch (same chronology as Phase 2 Elo).

Per batch b, in date order:

1. Refit (if due) the opponent model on observations with release_batch < b.
2. Freeze: every target match with prediction_batch == b gets snapshots of both players built from
   the state left by earlier batches. Nothing from batch b is visible, including the match itself.
3. Release: observations with release_batch == b enter the state (multi-week events release later,
   exactly as their Elo results do).

Determinism and order independence: input rows are sorted by a canonical content key
(release_batch, prediction_batch, tournament_date, round_order, match_id) before processing; point
counts are integers (exact); decay sums add each batch's terms with math.fsum.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl

from tennis_data.features import opponent
from tennis_data.features.config import FeatureConfig
from tennis_data.features.dictionary import player_specs
from tennis_data.features.estimates import prior_weight, rate, shrunk, sigmoid, surface_estimate

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
CANONICAL = ("release_batch", "prediction_batch", "tournament_date", "round_order", "match_id")


@dataclass
class _Player:
    sp_n: int = 0
    sp_w: int = 0
    rp_n: int = 0
    rp_w: int = 0
    matches: int = 0
    last_event: object = None
    surf: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    recent: list = field(default_factory=list)  # (event_date, sp_n, sp_w, rp_n, rp_w) in canonical order
    # half_life -> [S_n, S_w, R_n, R_w, Q_sn, Q_rn, M, M2, t_ref] where S/R = sum w*x, Q = sum w^2 * points,
    # M = sum w and M2 = sum w^2 over observations (matches); all referenced to batch t_ref.
    decay: dict = field(default_factory=dict)


@dataclass
class _Population:
    sp_n: int = 0
    sp_w: int = 0
    surf: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))


def _days(a, b) -> int:
    return (a - b).days


def _sum_tail(items, n):
    sp_n = sp_w = rp_n = rp_w = 0
    for _, a, b, c, d in items[-n:] if n else ():
        sp_n += a
        sp_w += b
        rp_n += c
        rp_w += d
    return sp_n, sp_w, rp_n, rp_w


def _sum_since(items, cutoff, slack_days):
    sp_n = sp_w = rp_n = rp_w = 0
    for ev, a, b, c, d in reversed(items):
        if ev >= cutoff:
            sp_n += a
            sp_w += b
            rp_n += c
            rp_w += d
        elif _days(cutoff, ev) > slack_days:
            break  # items are in release order; out-of-order event dates are bounded by slack
    return sp_n, sp_w, rp_n, rp_w


class _Fits:
    """Opponent-model fits per pool, on canonically ordered server-returner observations."""

    def __init__(self, cfg: FeatureConfig, obs: pl.DataFrame):
        self.cfg = cfg
        self.current: dict[str, opponent.OpponentFit] = {}
        self.log: list[dict] = []
        self.param_rows: list[tuple] = []
        self.data = {}
        obs = obs.filter(pl.col("surface").is_not_null())
        for pool in obs["pool"].unique().sort().to_list():
            o = obs.filter(pl.col("pool") == pool)
            ids = np.unique(np.concatenate([o["server_id"].to_numpy(), o["returner_id"].to_numpy()]))
            index = {int(p): i for i, p in enumerate(ids)}
            surf_index = {s: i for i, s in enumerate(SURFACES)}
            self.data[pool] = {
                "ids": ids,
                "index": index,
                "release": o["release_batch"].to_numpy(),
                "event": o["prediction_batch"].to_numpy(),
                "server": np.searchsorted(ids, o["server_id"].to_numpy()),
                "returner": np.searchsorted(ids, o["returner_id"].to_numpy()),
                "surface": np.array([surf_index[s] for s in o["surface"].to_list()], dtype=np.int64),
                "points": o["points"].to_numpy().astype(np.float64),
                "won": o["won"].to_numpy().astype(np.float64),
            }

    def maybe_refit(self, batch) -> None:
        o = self.cfg.opponent
        for pool, d in self.data.items():
            prev = self.current.get(pool)
            if prev is not None and _days(batch, prev.fit_batch) < 7 * o.refit_interval_weeks:
                continue
            m = int(np.searchsorted(d["release"], np.datetime64(batch, "D"), side="left"))
            if m == 0 or d["points"][:m].sum() < o.min_fit_points:
                continue
            age_weeks = (np.datetime64(batch, "D") - d["event"][:m]).astype("timedelta64[D]").astype(np.float64) / 7.0
            weight = np.exp2(-age_weeks / o.half_life_weeks)
            f = opponent.fit(
                d["server"][:m], d["returner"][:m], d["surface"][:m], d["points"][:m], d["won"][:m], weight,
                len(d["ids"]), SURFACES, o.kappa_serve, o.kappa_return, o.kappa_surface, o.max_iterations, o.tolerance,
                o.max_step,
                fit_batch=batch, init=prev,
            )  # fmt: skip
            self.current[pool] = f
            self.log.append(
                {"fit_batch": batch, "pool": pool, "observations": m, "iterations": f.iterations, "converged": f.converged,
                 "final_max_step": f.max_step, "seconds": f.seconds, "mu_overall": f.mu_overall,
                 **{f"mu_{s}": float(f.mu[i]) for i, s in enumerate(SURFACES)}}
            )  # fmt: skip
            if o.store_fits:
                seen = np.nonzero((f.serve_weight + f.return_weight) > 0)[0]
                for i in seen:
                    self.param_rows.append(
                        (batch, pool, int(d["ids"][i]), float(f.alpha[i]), float(f.beta[i]),
                         float(f.serve_weight[i]), float(f.return_weight[i]))
                    )  # fmt: skip

    def player(self, pool: str, pid: int, surface: str | None) -> dict:
        f = self.current.get(pool)
        if f is None:
            return {}
        d = self.data[pool]
        i = d["index"].get(pid)
        alpha = beta = 0.0
        a_s = b_s = 0.0
        sw = rw = s_ess = r_ess = 0.0
        s = SURFACES.index(surface) if surface in SURFACES else None
        if i is not None:
            alpha, beta, sw, rw = (
                float(f.alpha[i]),
                float(f.beta[i]),
                float(f.serve_weight[i]),
                float(f.return_weight[i]),
            )
            s_ess, r_ess = float(f.serve_ess[i]), float(f.return_ess[i])
            if s is not None:
                a_s, b_s = float(f.a_surf[i, s]), float(f.b_surf[i, s])
        mu_all = f.mu_overall
        out = {
            "serve_strength": alpha,
            "return_strength": beta,
            "serve_adjusted": sigmoid(mu_all + alpha),
            "return_adjusted": 1.0 - sigmoid(mu_all - beta),
            "surface_serve_adjusted": sigmoid(float(f.mu[s]) + alpha + a_s) if s is not None else None,
            "surface_return_adjusted": 1.0 - sigmoid(float(f.mu[s]) - beta - b_s) if s is not None else None,
            "adjusted_serve_points_weighted": sw,
            "adjusted_return_points_weighted": rw,
            "adjusted_serve_ess": s_ess,
            "adjusted_return_ess": r_ess,
            "adjusted_in_fit": (sw + rw) > 0,
            "_serve_logit": (float(f.mu[s]) if s is not None else mu_all) + alpha + a_s,
            "_return_logit": beta + b_s,
            "_fit_batch": f.fit_batch,
        }
        return out


@dataclass
class SnapshotResult:
    features: pl.DataFrame
    fit_log: pl.DataFrame
    fit_params: pl.DataFrame
    timings: dict


def build_snapshots(frame: pl.DataFrame, obs: pl.DataFrame, cfg: FeatureConfig) -> SnapshotResult:
    """``frame``: feature_frame() output (all validated rows). ``obs``: observations(frame)."""
    t0 = time.perf_counter()
    k = cfg.shrinkage
    slack_days = 7 * max([r.weeks for r in cfg.elo.chronology.multi_week] + [1])
    halves = cfg.half_life_weeks

    targets = frame.filter("is_target").sort("prediction_batch", "match_id")
    released = frame.filter("stats_eligible").sort(list(CANONICAL), nulls_last=True)
    rel_rows = released.select(
        "pool", "prediction_batch", "release_batch", "surface", "player_a_id", "player_b_id",
        "a_sp_played", "a_sp_won", "b_sp_played", "b_sp_won",
    ).to_dicts()  # fmt: skip
    release_at = defaultdict(list)
    for r in rel_rows:
        release_at[r["release_batch"]].append(r)
    target_rows = targets.select(
        "match_id", "prediction_batch", "tournament_date", "season", "pool", "tournament_level", "surface",
        "player_a_id", "player_b_id", "stats_eligible",
    ).to_dicts()  # fmt: skip
    predict_at = defaultdict(list)
    for r in target_rows:
        predict_at[r["prediction_batch"]].append(r)

    fits = _Fits(cfg, obs) if cfg.opponent.enabled else None
    players: dict = defaultdict(_Player)
    pops: dict = defaultdict(_Population)
    columns: dict[str, list] = defaultdict(list)
    n_rows = 0
    t_fit = 0.0

    def priors(pool, surface):
        p = pops[pool]
        mu = p.sp_w / p.sp_n if p.sp_n > 0 and p.sp_n >= k.min_prior_points else None
        mu_s = None
        if surface is not None:
            ps = p.surf[surface]
            mu_s = ps[1] / ps[0] if ps[0] > 0 and ps[0] >= k.min_prior_points else None
        return mu, mu_s

    def snapshot(pool, pid, surface, batch) -> dict:
        st = players[(pool, pid)]
        mu, mu_s = priors(pool, surface)
        mu_r = None if mu is None else 1.0 - mu
        mu_rs = None if mu_s is None else 1.0 - mu_s
        d = {
            "has_stats_history": st.sp_n > 0,
            "stat_matches": st.matches,
            "weeks_since_last_stats": _days(batch, st.last_event) // 7 if st.last_event is not None else None,
            "service_points_observed": st.sp_n,
            "service_points_won": st.sp_w,
            "return_points_observed": st.rp_n,
            "return_points_won": st.rp_w,
            "spw_raw": rate(st.sp_w, st.sp_n),
            "rpw_raw": rate(st.rp_w, st.rp_n),
            "prior_spw": mu,
            "prior_rpw": mu_r,
            "spw_shrunk": shrunk(st.sp_w, st.sp_n, mu, k.kappa_service),
            "rpw_shrunk": shrunk(st.rp_w, st.rp_n, mu_r, k.kappa_return),
            "spw_prior_weight": prior_weight(st.sp_n, k.kappa_service),
            "rpw_prior_weight": prior_weight(st.rp_n, k.kappa_return),
        }
        d["service_prior_dominated"] = d["spw_prior_weight"] > 0.5
        d["return_prior_dominated"] = d["rpw_prior_weight"] > 0.5
        if surface is None:
            d.update(surface_service_points=None, surface_return_points=None, surface_stat_matches=None,
                     surface_spw_raw=None, surface_rpw_raw=None, surface_prior_spw=None, surface_spw=None,
                     surface_rpw=None, surface_prior_weight=None)  # fmt: skip
        else:
            s_sp_n, s_sp_w, s_rp_n, s_rp_w, s_m = st.surf[surface] if surface in st.surf else (0, 0, 0, 0, 0)
            d.update(
                surface_service_points=s_sp_n, surface_return_points=s_rp_n, surface_stat_matches=s_m,
                surface_spw_raw=rate(s_sp_w, s_sp_n), surface_rpw_raw=rate(s_rp_w, s_rp_n), surface_prior_spw=mu_s,
                surface_spw=surface_estimate(s_sp_w, s_sp_n, st.sp_w, st.sp_n, mu, mu_s, k.kappa_service, k.kappa_surface_service),
                surface_rpw=surface_estimate(s_rp_w, s_rp_n, st.rp_w, st.rp_n, mu_r, mu_rs, k.kappa_return, k.kappa_surface_return),
                surface_prior_weight=prior_weight(s_sp_n, k.kappa_surface_service),
            )  # fmt: skip
        for n in cfg.last_matches:
            a, b, c, e = _sum_tail(st.recent, n)
            d[f"spw_last{n}"], d[f"rpw_last{n}"] = rate(b, a), rate(e, c)
            d[f"service_points_last{n}"], d[f"return_points_last{n}"] = a, c
        for w in cfg.last_weeks:
            cutoff = batch - timedelta(weeks=w)  # events starting in the last w batches
            a, b, c, e = _sum_since(st.recent, cutoff, slack_days)
            d[f"spw_last{w}w"], d[f"rpw_last{w}w"] = rate(b, a), rate(e, c)
            d[f"service_points_last{w}w"], d[f"return_points_last{w}w"] = a, c
        for h in halves:
            tag = f"{h:g}"
            dec = st.decay.get(h)
            if dec is None:
                sn = sw = rn = rw = qs = qr = m1 = m2 = 0.0
            else:
                f = 2.0 ** (-_days(batch, dec[8]) / 7.0 / h)
                sn, sw, rn, rw, m1 = dec[0] * f, dec[1] * f, dec[2] * f, dec[3] * f, dec[6] * f
                qs, qr, m2 = dec[4] * f * f, dec[5] * f * f, dec[7] * f * f
            d[f"spw_decay{tag}w"], d[f"rpw_decay{tag}w"] = rate(sw, sn), rate(rw, rn)
            d[f"service_points_decay{tag}w"], d[f"return_points_decay{tag}w"] = sn, rn
            # Kish effective sample sizes: (sum w x)^2 / (sum w^2 x). See docs/features.md for assumptions.
            d[f"service_ess_decay{tag}w"] = sn * sn / qs if qs > 0 else 0.0
            d[f"return_ess_decay{tag}w"] = rn * rn / qr if qr > 0 else 0.0
            d[f"matches_ess_decay{tag}w"] = m1 * m1 / m2 if m2 > 0 else 0.0
        if fits is not None:
            d.update(fits.player(pool, pid, surface))
        return d

    def release(batch, rows) -> None:
        per_player = defaultdict(list)
        for r in rows:
            pool, s, ev = r["pool"], r["surface"], r["prediction_batch"]
            a_obs = (r["a_sp_played"], r["a_sp_won"], r["b_sp_played"], r["b_sp_played"] - r["b_sp_won"])
            b_obs = (r["b_sp_played"], r["b_sp_won"], r["a_sp_played"], r["a_sp_played"] - r["a_sp_won"])
            per_player[(pool, r["player_a_id"])].append((ev, s, a_obs))
            per_player[(pool, r["player_b_id"])].append((ev, s, b_obs))
            p = pops[pool]
            p.sp_n += r["a_sp_played"] + r["b_sp_played"]
            p.sp_w += r["a_sp_won"] + r["b_sp_won"]
            if s is not None:
                p.surf[s][0] += r["a_sp_played"] + r["b_sp_played"]
                p.surf[s][1] += r["a_sp_won"] + r["b_sp_won"]
        for key, items in per_player.items():
            st = players[key]
            for ev, s, (spn, spw, rpn, rpw) in items:
                st.sp_n += spn
                st.sp_w += spw
                st.rp_n += rpn
                st.rp_w += rpw
                st.matches += 1
                if s is not None:
                    v = st.surf[s]
                    v[0] += spn
                    v[1] += spw
                    v[2] += rpn
                    v[3] += rpw
                    v[4] += 1
                st.recent.append((ev, spn, spw, rpn, rpw))
                st.last_event = ev if st.last_event is None or ev > st.last_event else st.last_event
            for h in halves:
                dec = st.decay.get(h)
                f = 1.0 if dec is None else 2.0 ** (-_days(batch, dec[8]) / 7.0 / h)
                if dec is None:
                    base = [0.0] * 8
                else:
                    # linear sums decay by f, squared-weight sums by f^2
                    base = [dec[j] * (f * f if j in (4, 5, 7) else f) for j in range(8)]
                terms = [[] for _ in range(8)]
                for ev, _, (spn, spw, rpn, rpw) in items:
                    wgt = 2.0 ** (-_days(batch, ev) / 7.0 / h)
                    w2 = wgt * wgt
                    for j, v in enumerate((wgt * spn, wgt * spw, wgt * rpn, wgt * rpw, w2 * spn, w2 * rpn, wgt, w2)):
                        terms[j].append(v)
                st.decay[h] = [base[j] + math.fsum(terms[j]) for j in range(8)] + [batch]

    names = [s.name for s in player_specs(cfg)]
    for batch in sorted(set(predict_at) | set(release_at)):
        if fits is not None:
            tf = time.perf_counter()
            fits.maybe_refit(batch)
            t_fit += time.perf_counter() - tf
        cache: dict = {}
        for r in predict_at.get(batch, ()):
            row = dict(r)
            if fits is not None:
                row.update(a_spw_vs_b=None, b_spw_vs_a=None, adjustment_fit_batch=None)
            snaps = {}
            for side in ("a", "b"):
                key = (r["pool"], r[f"player_{side}_id"], r["surface"])
                if key not in cache:
                    cache[key] = snapshot(r["pool"], r[f"player_{side}_id"], r["surface"], batch)
                snaps[side] = cache[key]
                for n in names:
                    row[f"{side}_{n}"] = snaps[side].get(n)
            if fits is not None:
                sa, sb = snaps["a"], snaps["b"]
                if "_serve_logit" in sa:
                    row["a_spw_vs_b"] = sigmoid(sa["_serve_logit"] - sb["_return_logit"])
                    row["b_spw_vs_a"] = sigmoid(sb["_serve_logit"] - sa["_return_logit"])
                    row["adjustment_fit_batch"] = sa["_fit_batch"]
            for key, value in row.items():
                columns[key].append(value)
            n_rows += 1
        if batch in release_at:
            release(batch, release_at[batch])

    timings = {"snapshot_loop_seconds": time.perf_counter() - t0 - t_fit, "opponent_fit_seconds": t_fit}
    fit_log = pl.DataFrame(fits.log if fits else [], infer_schema_length=None)
    fit_params = pl.DataFrame(
        fits.param_rows if fits else [],
        schema={"fit_batch": pl.Date, "pool": pl.String, "player_id": pl.Int64, "alpha": pl.Float64, "beta": pl.Float64,
                "serve_points_weighted": pl.Float64, "return_points_weighted": pl.Float64},
        orient="row",
    )  # fmt: skip
    features = pl.DataFrame(dict(columns), nan_to_null=False, strict=False) if n_rows else pl.DataFrame()
    return SnapshotResult(features, fit_log, fit_params, timings)

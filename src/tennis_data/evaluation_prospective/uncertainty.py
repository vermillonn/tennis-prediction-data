"""Paired uncertainty for Delta LL = LL_M4A - LL_Elo with temporal and tournament block bootstraps.

Reuses ``modeling.evaluation.paired_bootstrap`` (whole clusters resampled with replacement, fsum per cluster,
deterministic seed). Clusters:

- ``week``: the ISO week (Monday) of the actual start;
- ``tournament``: the tournament id, only when every scored match has one (otherwise the grouping is unreliable and
  no tournament interval is reported).

Neither grouping captures dependence through players who recur across weeks and tournaments, so intervals are, if
anything, too narrow. Below ``min_matches_inferential`` matches or ``min_blocks_inferential`` blocks no interval is
reported: the result is descriptive and marked "insufficient evidence". Season-level figures are descriptive only.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np

from tennis_data.ledger.hashing import parse_ts
from tennis_data.modeling.evaluation import paired_bootstrap, row_losses, verdict


def cluster_ids(rows: list[dict], kind: str) -> list[str] | None:
    if kind == "week":
        ids = []
        for r in rows:
            t = parse_ts(r["actual_start_at"])
            ids.append((t.date() - timedelta(days=t.weekday())).isoformat())
        return ids
    if kind == "tournament":
        ids = [r["tournament_id"] for r in rows]
        return None if any(not x for x in ids) else ids
    raise ValueError(kind)


def paired_uncertainty(rows: list[dict], protocol: dict) -> dict:
    rows = sorted(rows, key=lambda r: r["match_id"])
    n = len(rows)
    out = {"n": n, "min_matches_inferential": protocol["min_matches_inferential"],
           "min_blocks_inferential": protocol["min_blocks_inferential"], "intervals": {}, "seasons": []}  # fmt: skip
    if n == 0:
        out["status"] = "no scorable matches"
        return out
    p_new = np.array([r["probability_a"] for r in rows], dtype=np.float64)
    p_base = np.array([r["baseline_elo_probability_a"] for r in rows], dtype=np.float64)
    y = np.array([r["y"] for r in rows], dtype=np.float64)
    for kind in protocol["clusters"]:
        ids = cluster_ids(rows, kind)
        if ids is None:
            out["intervals"][kind] = {"status": "grouping unreliable (missing tournament ids): not reported"}
            continue
        k = len(set(ids))
        entry = {"blocks": k, "matches": n}
        if n < protocol["min_matches_inferential"] or k < protocol["min_blocks_inferential"]:
            entry["status"] = (f"insufficient evidence: {n} matches in {k} blocks (needs >= "
                               f"{protocol['min_matches_inferential']} matches and >= "
                               f"{protocol['min_blocks_inferential']} blocks); no interval reported")  # fmt: skip
        else:
            b = paired_bootstrap(p_new, p_base, y, np.array(ids), protocol["bootstrap_samples"],
                                 protocol["bootstrap_seed"])  # fmt: skip
            entry |= {"status": "reported", "log_loss_diff": b["log_loss_diff"], "ll_ci95": list(b["ll_ci95"]),
                      "brier_diff": b["brier_diff"], "brier_ci95": list(b["brier_ci95"]),
                      "share_draws_below_zero": b["share_ll_below_zero"], "samples": b["samples"], "seed": b["seed"],
                      "verdict": verdict(*b["ll_ci95"])}  # fmt: skip
        out["intervals"][kind] = entry
    ll_m, _ = row_losses(p_new, y)
    ll_e, _ = row_losses(p_base, y)
    d = ll_m - ll_e
    seasons: dict = {}
    for r, di in zip(rows, d.tolist(), strict=True):
        seasons.setdefault(r["actual_start_at"][:4], []).append(di)
    out["seasons"] = [{"season": s, "n": len(v), "d_ll": math.fsum(v) / len(v), "status": "descriptive only"}
                      for s, v in sorted(seasons.items())]  # fmt: skip
    reported = [v for v in out["intervals"].values() if v.get("status") == "reported"]
    out["status"] = "inferential" if reported else "insufficient evidence (descriptive only)"
    return out

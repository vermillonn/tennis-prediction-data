"""Calibration tables and pre-declared subgroups. Nothing is fitted or recalibrated.

Calibration bins come from the shared scorer (``elo.evaluation.calibration``: equal-width bins on P(A wins)).
Every pre-declared subgroup is always reported; groups below ``min_subgroup_matches`` show counts only and are
marked inconclusive, so results cannot be selected after looking at them.
"""

from __future__ import annotations

import math

from tennis_data.evaluation_prospective.eligibility import subgroup_values
from tennis_data.evaluation_prospective.metrics import MODELS, frame, score
from tennis_data.modeling.evaluation import calibration


def calibration_tables(rows: list[dict], bins: int) -> dict:
    if not rows:
        return {name: [] for name in MODELS}
    df = frame(sorted(rows, key=lambda r: r["match_id"]))
    out = {}
    for name, col in MODELS.items():
        out[name] = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
                     for r in calibration(df, col, bins).iter_rows(named=True)]  # fmt: skip
    return out


def subgroup_tables(rows: list[dict], protocol: dict) -> dict:
    out = {}
    min_n = protocol["min_subgroup_matches"]
    labelled = [(subgroup_values(r, protocol), r) for r in rows]
    for g in protocol["subgroups"]:
        groups: dict = {}
        for labels, r in labelled:
            groups.setdefault(labels[g], []).append(r)
        table = []
        for label in sorted(groups):
            members = groups[label]
            entry = {"group": label, "n": len(members), "status": "reported" if len(members) >= min_n else
                     f"inconclusive (n < {min_n}): counts only"}  # fmt: skip
            if len(members) >= min_n:
                s = score(members, protocol)
                entry |= {"ll_m4a": s["models"]["M4A"]["log_loss"], "ll_elo": s["models"]["Elo"]["log_loss"],
                          "d_ll": s["paired"]["log_loss_diff"], "brier_m4a": s["models"]["M4A"]["brier"],
                          "brier_elo": s["models"]["Elo"]["brier"]}  # fmt: skip
            table.append(entry)
        out[g] = table
    return out

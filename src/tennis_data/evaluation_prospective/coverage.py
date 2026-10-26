"""Coverage and exclusion accounting: every committed prediction gets exactly one final category.

Categories are mutually exclusive (``eligibility.CATEGORIES``); flags overlap and are counted separately. Every
percentage states its denominator.
"""

from __future__ import annotations

from collections import Counter

from tennis_data.evaluation_prospective.eligibility import CATEGORIES


def pct(n: int, d: int) -> dict:
    return {"n": n, "denominator": d, "pct": (100.0 * n / d) if d else None}


def coverage(rows: list[dict], protocol: dict) -> dict:
    total = len(rows)
    mode = protocol["cohort_mode"]
    cats = Counter(r["category"] for r in rows)
    flags = Counter(f.split(":")[0] if f.startswith(("readiness", "voided")) else f for r in rows for f in r["flags"])
    target = [r for r in rows if r["mode"] == mode]
    selected = [r for r in target if r["selected"]]
    scorable = [r for r in rows if r["category"] == "SCORABLE"]
    out = {
        "total_predictions": total,
        "by_mode": {m: pct(n, total) for m, n in sorted(Counter(r["mode"] for r in rows).items())},
        "by_category": {c: pct(cats.get(c, 0), total) for c in CATEGORIES},
        "flags": {k: pct(v, total) for k, v in sorted(flags.items())},
        "by_evidence_level": {k: pct(v, total) for k, v in sorted(Counter(r["evidence_level"] for r in rows).items())},
        "cohort_mode": mode,
        "cohort_mode_predictions": len(target),
        "cohort_matches": len({r["match_id"] for r in target}),
        "selected_forecasts": pct(len(selected), len(target)),
        "admissible_forecasts": pct(sum(r["admissible"] for r in target), len(target)),
        "scorable": pct(len(scorable), len(selected)),
        "primary_cohort": pct(sum(r["cohort"] == "primary" for r in rows), len(selected)),
        "operational_cohort": pct(sum(r["cohort"] in ("primary", "operational") for r in rows), len(selected)),
        "retirement_secondary": pct(sum(r["cohort"] == "retirement_secondary" for r in rows), len(selected)),
    }
    flow = [("all prediction commitments", total), (f"{mode} records", len(target)),
            ("approved model, baseline and readiness", sum(1 for r in target if r["category"] not in (
                "MODEL_NOT_APPROVED", "MISSING_BASELINE", "NOT_OPERATIONALLY_READY"))),
            ("selected (first admissible in primary slot)", len(selected)),
            ("not voided, cancelled or superseded", sum(1 for r in selected if r["category"] not in (
                "VOIDED", "MATCH_CANCELLED", "MATCH_SUPERSEDED"))),
            ("committed before verified actual start", sum(1 for r in selected if r["timing"] == "PRE_START")),
            ("outcome resolved and scorable (COMPLETED)", len(scorable)),
            ("independently timestamped before start (primary)", out["primary_cohort"]["n"])]  # fmt: skip
    out["flow"] = [{"step": s, "n": n} for s, n in flow]
    return out


def exclusions(rows: list[dict], mode: str) -> list[dict]:
    """Every non-scorable prediction of the cohort mode, with its category and flags (voids included)."""
    return [{"seq": r["seq"], "event_id": r["event_id"], "match_id": r["match_id"], "category": r["category"],
             "flags": r["flags"], "outcome": r["outcome"], "timing": r["timing"],
             "void_reason": (r["voided"] or {}).get("reason_code"), "void_actor": (r["voided"] or {}).get("actor")}
            for r in rows if r["mode"] == mode and r["category"] != "SCORABLE"]  # fmt: skip


def voids(rows: list[dict]) -> list[dict]:
    return [{"seq": r["seq"], "event_id": r["event_id"], "match_id": r["match_id"], "mode": r["mode"],
             "reason_code": r["voided"]["reason_code"], "actor": r["voided"]["actor"],
             "void_event_id": r["voided"]["event_id"], "category": r["category"]}
            for r in rows if r["voided"]]  # fmt: skip


def unresolved(outcomes: dict, rows: list[dict], mode: str) -> list[dict]:
    mids = {r["match_id"] for r in rows if r["mode"] == mode}
    return [{"match_id": m, "outcome": o["outcome"], "flags": o["flags"], "actual_start_at": o["actual_start_at"],
             "sources": [a["provider"] for a in o["assertions"]]}
            for m, o in sorted(outcomes.items()) if m in mids and o["outcome"] in ("PENDING", "DISPUTED",
                                                                                   "IDENTITY_MISMATCH", "UNKNOWN")]  # fmt: skip

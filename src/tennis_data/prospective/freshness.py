"""Operational reliability of a prediction's inputs (never changes the model probability).

For each upcoming match: how stale the state is (latest state-eligible result, latest usable point statistics,
age of the opponent-model fit), how many weeks between the last ingested results and the match have no tour-level
results at all, how many identities are unresolved, how old the newest source snapshot is, and how recently each
player has statistics. Each check is ``ok``, ``warning`` or ``fail`` against ``[freshness]`` thresholds; the overall
status is the worst one. A ``fail`` blocks prospective predictions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from tennis_data.prospective.config import ProspectiveConfig

LEVELS = ("ok", "warning", "fail")


def _grade(value, warn, fail) -> str:
    if value is None:
        return "fail"
    return "fail" if value >= fail else "warning" if value >= warn else "ok"


def state_coverage(frame: pl.DataFrame) -> dict:
    """Latest batches that actually changed state (Elo updates and statistics), and the weeks that have results."""
    upd = frame.filter(pl.col("is_update"))
    stats = frame.filter(pl.col("stats_eligible"))
    return {
        "last_update_release_batch": upd["release_batch"].max(),
        "last_stats_release_batch": stats["release_batch"].max(),
        "result_weeks": set(upd["prediction_batch"].unique().to_list()),
    }


def missing_weeks(cover: dict, batch: date) -> int:
    """Weeks from the last week with results to the prediction batch with no tour-level result (off-weeks count)."""
    last = max(cover["result_weeks"]) if cover["result_weeks"] else None
    if last is None:
        return 10**6
    n, b = 0, last + timedelta(days=7)
    while b < batch:
        if b not in cover["result_weeks"]:
            n += 1
        b += timedelta(days=7)
    return n


def assess(pending: pl.DataFrame, cover: dict, cfg: ProspectiveConfig, *, unresolved_ids: set[int] | None = None,
           unresolved_count: int = 0, source_retrieved_at: datetime | None = None,
           now: datetime | None = None) -> dict[str, dict]:  # fmt: skip
    f = cfg.freshness
    now = now or datetime.now(UTC)
    out = {}
    for r in pending.iter_rows(named=True):
        b = r["prediction_batch"]
        res_close = (
            cover["last_update_release_batch"] + timedelta(days=7) if cover["last_update_release_batch"] else None
        )
        st_close = cover["last_stats_release_batch"] + timedelta(days=7) if cover["last_stats_release_batch"] else None
        result_age = (b - res_close).days if res_close else None
        fit = r.get("adjustment_fit_batch")
        fit_age = (b - fit).days // 7 if fit else None
        mw = missing_weeks(cover, b)
        src_age = (now - source_retrieved_at).days if source_retrieved_at else None
        inactive = max((r[f"{s}_weeks_since_last_stats"] or 10**6) if r[f"{s}_has_stats_history"] else 10**6
                       for s in ("a", "b"))  # fmt: skip
        ids_bad = len({r["player_a_id"], r["player_b_id"]} & (unresolved_ids or set()))
        checks = {
            "last_eligible_result": {"value": str(res_close) if res_close else None, "age_days": result_age,
                                     "status": _grade(result_age, f.warn_result_age_days, f.fail_result_age_days)},
            "last_point_statistics": {"value": str(st_close) if st_close else None,
                                      "age_days": (b - st_close).days if st_close else None,
                                      "status": _grade((b - st_close).days if st_close else None,
                                                       f.warn_result_age_days, f.fail_result_age_days)},
            "opponent_fit_age_weeks": {"value": fit_age, "status": _grade(fit_age, f.warn_fit_age_weeks, f.fail_fit_age_weeks)},
            "missing_backfill_weeks": {"value": mw, "status": _grade(mw, f.warn_missing_weeks, f.fail_missing_weeks)},
            "unresolved_identities": {"value": unresolved_count, "players_in_match": ids_bad,
                                      "status": "fail" if ids_bad >= f.fail_unresolved_identities
                                      else "warning" if unresolved_count else "ok"},
            "source_age_days": {"value": src_age, "status": "warning" if src_age is None else
                                _grade(src_age, f.warn_source_age_days, f.fail_source_age_days)},
            "player_recent_history": {"max_weeks_since_last_stats": None if inactive >= 10**6 else inactive,
                                      "status": "warning" if inactive >= f.warn_player_inactive_weeks else "ok"},
        }  # fmt: skip
        status = max((c["status"] for c in checks.values()), key=LEVELS.index)
        out[r["match_id"].removeprefix("pending:")] = {"status": status, "checks": checks}
    return out

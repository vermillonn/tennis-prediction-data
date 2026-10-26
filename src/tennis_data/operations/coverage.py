"""Week-by-week audit of the historical gap (from the week after the archive's last week to the last completed week).

Nothing is inferred to be complete because a provider returned *some* results. For every week the audit reports:

- **expected tournaments (inferred)**: tour-level events of the frozen rating pools whose start falls in the same
  ISO week of ``calendar_reference_season`` in the archive. This is a calendar estimate from source metadata, not a
  verified schedule, and is labelled so;
- observed tournaments and matches retrieved from candidate sources, validated (accepted by Phase 4A), quarantined,
  share with complete serve statistics, unresolved player identities, and the freshness of the newest source record;
- a status: ``NO_DATA`` (nothing retrieved), ``PARTIAL`` (fewer observed than expected events, or unresolved
  identities or quarantined records), ``OBSERVED_UNVERIFIED`` (every expected event observed, but completeness cannot
  be verified without an authoritative calendar).

A week is never reported as complete: verifying completeness needs an authoritative calendar and draw sizes.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def weeks(first: date, as_of: date) -> list[date]:
    """Mondays from ``first`` to the last Monday whose week ended before ``as_of``."""
    last = as_of - timedelta(days=as_of.weekday() + 7)
    out, w = [], first
    while w <= last:
        out.append(w)
        w += timedelta(days=7)
    return out


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def expected_calendar(hist: pl.DataFrame, season: int, pools: tuple[str, ...]) -> dict[int, list[dict]]:
    """ISO week number → events of the reference season (inferred expectation)."""
    ev = (hist.filter((pl.col("season") == season) & pl.col("tournament_level").is_in(list(pools)))
          .group_by("tournament_id").agg(pl.col("tournament_name").first(), pl.col("tournament_level").first(),
                                         pl.col("tournament_date").min(), pl.len().alias("matches"))
          .sort("tournament_date", "tournament_id"))  # fmt: skip
    out: dict = {}
    for r in ev.iter_rows(named=True):
        out.setdefault(r["tournament_date"].isocalendar()[1], []).append(
            {"name": r["tournament_name"], "level": r["tournament_level"], "reference_matches": r["matches"]}
        )
    return out


def audit(cfg, as_of: date, hist: pl.DataFrame, candidates: pl.DataFrame, accepted: pl.DataFrame,
          quarantine: pl.DataFrame, decisions: pl.DataFrame) -> dict:  # fmt: skip
    """``candidates``: MATCH_SCHEMA records of all candidate sources; ``accepted``: Phase 1-validated backfill rows;
    ``quarantine``: Phase 4A reconciliation quarantine (provider, provider_match_id, reason)."""
    cal = expected_calendar(hist, cfg.calendar_reference_season, cfg.pools)
    unresolved = {r["provider_player_id"] for r in decisions.iter_rows(named=True) if not r["usable"]}
    cand = candidates.with_columns(pl.col("event_start_date").map_elements(_monday, return_dtype=pl.Date).alias("wk"))
    acc = accepted.with_columns(pl.col("tournament_date").map_elements(_monday, return_dtype=pl.Date).alias("wk")) \
        if accepted.height else None  # fmt: skip
    qkeys = {(r["provider"], r["provider_match_id"]) for r in quarantine.iter_rows(named=True)} \
        if quarantine.height and "provider_match_id" in quarantine.columns else set()  # fmt: skip
    newest = candidates["event_start_date"].max() if candidates.height else None
    rows = []
    for w in weeks(cfg.first_week, as_of):
        exp = cal.get(w.isocalendar()[1], [])
        c = cand.filter(pl.col("wk") == w)
        a = acc.filter(pl.col("wk") == w) if acc is not None else None
        n_acc = a.height if a is not None else 0
        q = sum(1 for r in c.iter_rows(named=True) if (r["provider"], r["provider_match_id"]) in qkeys)
        ids = set(c["winner_provider_id"].to_list()) | set(c["loser_provider_id"].to_list())
        unres = len(ids & unresolved)
        observed = sorted(set(c["tournament_name"].to_list()))
        stats = (a.filter(pl.col("stats_status") == "complete").height / n_acc) if n_acc else None
        status = ("NO_DATA" if c.height == 0 else
                  "PARTIAL" if len(observed) < len(exp) or unres or q or n_acc < c.height else "OBSERVED_UNVERIFIED")  # fmt: skip
        rows.append({"week": w.isoformat(), "expected_events_inferred": [e["name"] for e in exp],
                     "expected_reference_matches": sum(e["reference_matches"] for e in exp),
                     "observed_events": observed, "retrieved": c.height, "validated": n_acc, "quarantined": q,
                     "complete_stats_share": stats, "unresolved_identities": unres, "status": status})  # fmt: skip
    return {"as_of": as_of.isoformat(), "first_week": cfg.first_week.isoformat(),
            "archive_last_tournament_date": str(hist["tournament_date"].max()) if hist.height else None,
            "weeks": rows, "weeks_total": len(rows),
            "weeks_with_data": sum(1 for r in rows if r["retrieved"]),
            "weeks_no_data": sum(1 for r in rows if r["status"] == "NO_DATA"),
            "expected_events_inferred_total": sum(len(r["expected_events_inferred"]) for r in rows),
            "expected_reference_matches_total": sum(r["expected_reference_matches"] for r in rows),
            "retrieved_total": sum(r["retrieved"] for r in rows), "validated_total": sum(r["validated"] for r in rows),
            "newest_candidate_record_date": str(newest) if newest else None,
            "calendar_basis": f"inferred from the archive's {cfg.calendar_reference_season} season (not verified)"}  # fmt: skip

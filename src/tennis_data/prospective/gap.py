"""Historical gap audit: what the pinned archive actually covers near its end, and what is missing up to the
prospective start. Counts come from the validated and quarantine tables, not from downloaded row totals.

Expected volumes for the missing weeks are *estimates* from the same calendar weeks one year earlier; the archive
has no 2026 calendar for that period, so no claim is made about which 2026 events took place.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from tennis_data.elo.chronology import add_batches
from tennis_data.features.observations import feature_frame
from tennis_data.prospective.config import ProspectiveConfig


def classify_rows(ff: pl.DataFrame) -> pl.Expr:
    """1 complete observation / 2 outcome without usable statistics / 4 excluded by frozen eligibility."""
    return (
        pl.when(pl.col("is_target") & pl.col("stats_eligible")).then(pl.lit("1_complete_observation"))
        .when(pl.col("is_target")).then(pl.lit("2_outcome_without_statistics"))
        .otherwise(pl.lit("4_excluded_by_eligibility"))
    )  # fmt: skip


def audit(cfg: ProspectiveConfig, hist: pl.DataFrame, qids: pl.Series, quarantine: pl.DataFrame | None) -> dict:
    end, start = cfg.historical_snapshot_end, cfg.prospective_start
    year = end.year
    ff = feature_frame(hist.select([c for c in hist.columns]), cfg.features, qids)
    ff = ff.with_columns(classify_rows(ff).alias("category"))
    last_date = hist["tournament_date"].max()
    last_tour = ff.filter(pl.col("pool").is_not_null())
    out: dict = {
        "archive_commit": cfg.data.source.commit,
        "last_tournament_date": last_date,
        "configured_snapshot_end": end,
        "snapshot_end_verified": last_date == end,
        "last_tournament_date_by_level": last_tour.group_by("tournament_level").agg(pl.col("tournament_date").max())
        .sort("tournament_level").to_dicts(),
        "last_update_release_batch": ff.filter("is_update")["release_batch"].max(),
        "last_stats_release_batch": ff.filter("stats_eligible")["release_batch"].max(),
    }  # fmt: skip
    # Final weeks: events and completeness of knockout main draws (expected matches = draw_size - 1).
    tail = ff.filter((pl.col("tournament_date") > end - timedelta(days=35)) & pl.col("pool").is_not_null())
    ev = (tail.group_by("tournament_id", "tournament_name", "tournament_level", "tournament_date", "draw_size")
          .agg(pl.len().alias("rows"), pl.col("stats_eligible").sum().alias("rows_with_usable_stats"),
               pl.col("is_update").sum().alias("rows_updating_elo"), pl.col("round").unique().sort().alias("rounds"),
               pl.col("event_weeks").first(), pl.col("release_batch").first())
          .with_columns((pl.col("draw_size") - 1).alias("expected_main_draw_matches"))
          .with_columns(pl.when(pl.col("tournament_level") == "D").then(pl.lit("not checked (team event)"))
                        .when(pl.col("rows") >= pl.col("expected_main_draw_matches")).then(pl.lit("complete"))
                        .otherwise(pl.lit("incomplete")).alias("completeness"))
          .sort("tournament_date", "tournament_id"))  # fmt: skip
    out["final_weeks_events"] = ev.to_dicts()
    yr = ff.filter(pl.col("season") == year)
    out["year_rows"] = {
        "validated": yr.height,
        "in_rating_pool": yr.filter(pl.col("pool").is_not_null()).height,
        "by_category": yr.filter(pl.col("pool").is_not_null()).group_by("category").len().sort("category").to_dicts(),
        "by_status": yr.group_by("match_status").len().sort("match_status").to_dicts(),
        "stats_status": yr.group_by("stats_status").len().sort("stats_status").to_dicts(),
        "stats_exclusions": yr.filter(~pl.col("stats_eligible")).group_by("stats_exclusion_reason").len()
        .sort("stats_exclusion_reason").to_dicts(),
        "time_precision": hist.filter(pl.col("season") == year).group_by("time_precision").len().to_dicts(),
        "players_not_in_master": hist.filter((pl.col("season") == year)
                                             & pl.col("quality_flags").list.contains("player_not_in_master")).height,
        "multi_week_events": yr.filter(pl.col("event_weeks") > 1).select("tournament_id").n_unique(),
    }  # fmt: skip
    if quarantine is not None and quarantine.height:
        qy = quarantine.filter(pl.col("season") == year)
        out["year_rows"]["quarantined"] = qy.height
        out["year_rows"]["quarantine_reasons"] = (qy.explode("quarantine_reasons").group_by("quarantine_reasons").len()
                                                  .sort("quarantine_reasons").to_dicts())  # fmt: skip
    # Missing period: batches strictly after the last archive batch up to the prospective start.
    batches = add_batches(pl.DataFrame({"tournament_date": [last_date, start], "tournament_level": ["A", "A"],
                                        "draw_size": [32, 32]}), cfg.elo.chronology)["prediction_batch"].to_list()  # fmt: skip
    first_missing, last_needed = batches[0] + timedelta(days=7), batches[1]
    weeks = []
    b = first_missing
    while b <= last_needed:
        weeks.append(b)
        b += timedelta(days=7)
    out["missing_period"] = {"first_missing_batch": first_missing, "last_batch_before_prospective": last_needed,
                             "weeks": len(weeks), "archive_rows_in_period": 0}  # fmt: skip
    # Estimate from the same 22 weekly batches 52 weeks earlier (same weekdays), tour-level pool only.
    yr_ago = timedelta(days=364)
    prev = ff.filter((pl.col("prediction_batch") >= first_missing - yr_ago) & (pl.col("prediction_batch") <= last_needed - yr_ago)
                     & pl.col("pool").is_not_null())  # fmt: skip
    out["estimate_from_previous_year"] = {
        "window_batches": [first_missing - yr_ago, last_needed - yr_ago],
        "events": prev.select("tournament_id").n_unique(),
        "events_by_level": prev.group_by("tournament_level").agg(pl.col("tournament_id").n_unique().alias("events"),
                                                                pl.len().alias("matches")).sort("tournament_level").to_dicts(),
        "matches": prev.height,
        "by_category": prev.group_by("category").len().sort("category").to_dicts(),
        "multi_week_events": prev.filter(pl.col("event_weeks") > 1).select("tournament_name").unique().sort("tournament_name")
        ["tournament_name"].to_list(),
        "grand_slams": prev.filter(pl.col("tournament_level") == "G").select("tournament_name").unique()
        .sort("tournament_name")["tournament_name"].to_list(),
    }  # fmt: skip
    return out

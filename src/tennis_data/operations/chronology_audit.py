"""Chronology audit of an owner-supplied export against the frozen weekly batching (read-only).

The owner's ``2026.csv`` dates every match individually, while the archive and the frozen Phase 2 policy
(``weekly_batch_v1``) use one start date per event. The converter takes the earliest match date of an event as its
start. This module checks the consequences on data already produced by the dry run (no state is built here):

- **mapping**: on the January–May overlap, the frozen batch derived from the TML start is compared with the batch of
  the same archive event (pairs come from the duplicate detection of the validation simulation);
- **release timing**: each match's own date is compared with the time its result enters player state
  (``state_eligible_at``, the Saturday after its release batch). A match dated on or after that day would be used by
  the state before it was played. The same measure is computed for the archive's own dating of the overlap events,
  so a property of the frozen policy is not mistaken for an effect of the mapping;
- **exceptions**: United Cup and Davis Cup events are listed with both datings;
- **Roland Garros 2026**: the archived event (different tournament ID and date convention) must be recognised as
  duplicates and kept out of the backfill set;
- **coverage by week**: records, validated matches, held matches and Elo updates per frozen batch of the gap.

Nothing here changes chronology, eligibility or any state; findings are reported for review.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import polars as pl

from tennis_data.elo.chronology import add_batches
from tennis_data.operations.tml import PLAYER_FILE, decode
from tennis_data.prospective.backfill import load_archive

PREV_EVENTS = ("United Cup", "Cincinnati", "Chengdu", "Hangzhou", "Tokyo", "Beijing", "Shanghai", "Laver Cup")


def tml_match_dates(files: dict[str, bytes]) -> dict[str, date]:
    """Own date of every match, keyed like the converter's match IDs (``tid:match_num`` or the derived key)."""
    out = {}
    for name in sorted(n for n in files if n != PLAYER_FILE):
        for r in csv.DictReader(io.StringIO(decode(files[name])[0])):
            if not r["winner_id"] or not r["loser_id"] or not r["tourney_date"]:
                continue
            tid = r["tourney_id"]
            mid = f"{tid}:{r['match_num']}" if r["match_num"] else \
                f"{tid}:{r['round']}:{'-'.join(sorted((r['winner_id'], r['loser_id'])))}"  # fmt: skip
            out.setdefault(mid, datetime.strptime(r["tourney_date"], "%Y%m%d").date())
    return out


def _batches(df: pl.DataFrame, chron) -> pl.DataFrame:
    return add_batches(df, chron).with_columns((pl.col("release_batch") + pl.duration(days=7)).alias("eligible_date"))


def _lags(own: list[date], eligible: list[date]) -> dict:
    """Days by which matches were played on or after the day their result enters state (0 = that same Saturday)."""
    c = Counter((o - e).days for o, e in zip(own, eligible, strict=True) if o >= e)
    return {"matches": len(own), "played_on_or_after_release": sum(c.values()),
            "by_days_after": {str(k): v for k, v in sorted(c.items())}}  # fmt: skip


def _is_uc(name: str) -> bool:
    return "united cup" in name.lower()


def _is_dc(name: str, level: str) -> bool:
    return level == "D" or "davis cup" in name.lower()


def audit(full, gap, export_dir, as_of: date) -> dict:
    chron = gap.workspace_prospective().features.elo.chronology
    label = gap.tml["commit"]
    src = export_dir / "workspace" / "tml" / label
    own = tml_match_dates({n: (src / n).read_bytes() for n in full.tml["files"]})
    conv_full = json.loads((full.workspace_dir / "converted.json").read_text())
    conv_gap = json.loads((gap.workspace_dir / "converted.json").read_text())
    tours = {t["id"]: t for t in conv_full["tournaments"]}
    rec_tid = {m["id"]: m["tournament_id"] for m in conv_full["matches"]}
    hist, _ = load_archive(gap.workspace_prospective())
    # The archive's own dating of the previous season, for the events whose start day is unusual.
    prev = hist.filter(pl.col("season") == 2025)
    pe = prev.filter(pl.col("tournament_name").str.contains("|".join(PREV_EVENTS)))
    prev_dating = {
        "events": [
            {"tournament": n, "date": str(d), "weekday": d.strftime("%a")}
            for n, d in sorted(
                pe.group_by("tournament_name").agg(pl.col("tournament_date").min()).iter_rows(), key=lambda r: r[1]
            )
        ],
        "davis_cup_rubbers_by_weekday": dict(
            sorted(
                Counter(
                    d.strftime("%a") for d in prev.filter(pl.col("tournament_level") == "D")["tournament_date"]
                ).items()
            )
        ),
    }
    hist = hist.filter(pl.col("season") >= 2026)

    # Overlap: TML event start versus the archive event's own date, through the matches recognised as duplicates.
    notes = pl.read_parquet(export_dir / "validation_simulation" / "backfill" / "reconciliation_notes.parquet")
    dup = notes.filter(pl.col("reason") == "duplicate_of_historical").select(
        pl.col("provider_match_id"), pl.col("detail").alias("match_id")
    )
    dup = dup.with_columns(pl.col("provider_match_id").replace_strict(rec_tid, default=None).alias("tml_tid"))
    arch = hist.select("match_id", "tournament_id", "tournament_name", "tournament_date", "tournament_level",
                       "draw_size")  # fmt: skip
    pairs = dup.join(arch, on="match_id", how="inner")
    pairs = pairs.with_columns(
        pl.col("tml_tid").replace_strict({k: date.fromisoformat(str(t["start_date"])) for k, t in tours.items()},
                                         default=None, return_dtype=pl.Date).alias("tml_start"),
        pl.col("provider_match_id").replace_strict({k: v for k, v in own.items()}, default=None,
                                                   return_dtype=pl.Date).alias("own_date"))  # fmt: skip
    a = _batches(pairs, chron).rename({"prediction_batch": "arch_batch", "release_batch": "arch_release",
                                       "eligible_date": "arch_eligible", "event_weeks": "arch_weeks"})  # fmt: skip
    a = _batches(a.with_columns(pl.col("tml_start").alias("tournament_date")), chron)
    ev = (a.group_by("tml_tid", "tournament_name", "tournament_level").agg(
        pl.col("tml_start").first(), pl.col("tournament_id").unique().sort().alias("archive_tids"),
        pl.col("arch_batch").unique().sort().alias("archive_batches"),
        pl.col("prediction_batch").first().alias("tml_batch"),
        pl.col("arch_eligible").unique().sort().alias("archive_eligible"), pl.col("eligible_date").first().alias("tml_eligible"),
        pl.len().alias("matches"))).sort("tml_start", "tournament_name")  # fmt: skip
    events_overlap = []
    for r in ev.iter_rows(named=True):
        events_overlap.append({"tournament": r["tournament_name"], "level": r["tournament_level"],
                               "tml_tid": r["tml_tid"], "archive_tids": r["archive_tids"], "matches": r["matches"],
                               "tml_start": str(r["tml_start"]), "tml_start_weekday": r["tml_start"].strftime("%a"),
                               "tml_batch": str(r["tml_batch"]), "archive_batches": [str(x) for x in r["archive_batches"]],
                               "same_batch": r["archive_batches"] == [r["tml_batch"]],
                               "same_release": [x for x in r["archive_eligible"]] == [r["tml_eligible"]]})  # fmt: skip
    non_team = [e for e in events_overlap if e["level"] != "D" and not _is_uc(e["tournament"])]
    team = [e for e in events_overlap if e["level"] == "D" or _is_uc(e["tournament"])]
    ok = a.filter(pl.col("own_date").is_not_null())
    overlap_release = {"archive_dating": _lags(ok["own_date"].to_list(), ok["arch_eligible"].to_list()),
                       "tml_earliest_date": _lags(ok["own_date"].to_list(), ok["eligible_date"].to_list())}  # fmt: skip
    by_level_arch = {lv: _lags(g["own_date"].to_list(), g["arch_eligible"].to_list())["played_on_or_after_release"]
                     for (lv,), g in ok.group_by("tournament_level")}  # fmt: skip
    overlap_release["archive_dating_by_level"] = dict(sorted(by_level_arch.items()))

    # Gap (simulated backfill): every validated match's own date versus its release into state.
    sim = export_dir / "simulation" / "backfill"
    led = pl.read_parquet(sim / "ledger.parquet")
    mt = pl.read_parquet(sim / "matches.parquet").select("match_id", "tournament_name", "tournament_level", "draw_size",
                                                         "round")  # fmt: skip
    g = led.join(mt, on="match_id", how="left").with_columns(
        pl.col("provider_match_id").replace_strict(own, default=None, return_dtype=pl.Date).alias("own_date"),
        pl.col("state_eligible_at").dt.date().alias("eligible_date"))  # fmt: skip
    missing_dates = g.filter(pl.col("own_date").is_null()).height
    g = g.filter(pl.col("own_date").is_not_null())
    gap_release = _lags(g["own_date"].to_list(), g["eligible_date"].to_list())
    gap_release["missing_own_date"] = missing_dates
    gap_release["before_prediction_batch"] = g.filter(pl.col("own_date") < pl.col("prediction_batch")).height
    gap_release["by_level"] = {lv: _lags(x["own_date"].to_list(), x["eligible_date"].to_list())
                               ["played_on_or_after_release"] for (lv,), x in g.group_by("tournament_level")}  # fmt: skip
    gap_release["by_level"] = dict(sorted(gap_release["by_level"].items()))
    gap_release["by_round"] = dict(
        sorted(
            Counter(
                r["round"] for r in g.filter(pl.col("own_date") >= pl.col("eligible_date")).iter_rows(named=True)
            ).items()
        )
    )
    gap_release["latest_eligible"] = str(g["eligible_date"].max())
    gap_release["eligible_after_as_of"] = g.filter(pl.col("eligible_date") > as_of).height
    gap_release["played_after_as_of"] = g.filter(pl.col("own_date") > as_of).height
    gap_events = []
    for (_tid, name, lv), x in g.group_by("tournament_id", "tournament_name", "tournament_level"):
        early = x.filter(pl.col("own_date") >= pl.col("eligible_date"))
        gap_events.append({"tournament": name, "level": lv, "start": str(x["tournament_date"].min()),
                           "start_weekday": x["tournament_date"].min().strftime("%a"),
                           "last_match": str(x["own_date"].max()), "batch": str(x["prediction_batch"].min()),
                           "weeks": int(x["event_weeks"].max()), "enters_state": str(x["eligible_date"].min()),
                           "matches": x.height, "played_on_or_after": early.height,
                           "rounds_on_or_after": ", ".join(f"{k} {v}" for k, v in sorted(Counter(early["round"]).items()))})  # fmt: skip
    gap_events.sort(key=lambda e: (e["start"], e["tournament"]))

    # Roland Garros 2026: archived; must be recognised as duplicates and absent from the backfill set.
    rg_tids = sorted(t for t, x in tours.items() if "roland" in x["name"].lower())
    rg_records = [m for m in conv_full["matches"] if m["tournament_id"] in rg_tids]
    rg_ids = {m["id"] for m in rg_records}
    vq = pl.read_parquet(export_dir / "validation_simulation" / "backfill" / "reconciliation_quarantine.parquet")
    vm = pl.read_parquet(export_dir / "validation_simulation" / "backfill" / "matches.parquet")
    arch_rg = hist.filter(pl.col("tournament_name").str.to_lowercase().str.contains("roland"))
    rg = {"tml_tids": rg_tids, "tml_start": [str(tours[t]["start_date"]) for t in rg_tids],
          "tml_records": len(rg_records),
          "archive_tids": sorted(arch_rg["tournament_id"].unique().to_list()),
          "archive_tournament_date": sorted(str(x) for x in arch_rg["tournament_date"].unique().to_list()),
          "archive_matches": arch_rg.height,
          "recognised_duplicates": dup.filter(pl.col("provider_match_id").is_in(list(rg_ids))).height,
          "duplicates_pointing_to_archive_rg": pairs.filter(pl.col("provider_match_id").is_in(list(rg_ids))
                                                            & pl.col("match_id").is_in(arch_rg["match_id"].to_list())).height,
          "quarantined": dict(Counter(r["reason"] for r in vq.filter(pl.col("provider_match_id").is_in(list(rg_ids)))
                                      .iter_rows(named=True))),
          "accepted_as_new_in_validation": vm.filter(pl.col("tournament_name").str.to_lowercase().str.contains("roland")).height,
          "in_backfill_set": sum(1 for m in conv_gap["matches"] if m["tournament_id"] in rg_tids)}  # fmt: skip

    # Coverage by frozen batch over the gap.
    gtours = {t["id"]: date.fromisoformat(str(t["start_date"])) for t in conv_gap["tournaments"]}
    gt = pl.DataFrame({"tournament_date": list(gtours.values()), "tid": list(gtours.keys()),
                       "tournament_level": [None] * len(gtours), "draw_size": [None] * len(gtours)},
                      schema={"tournament_date": pl.Date, "tid": pl.String, "tournament_level": pl.String,
                              "draw_size": pl.Int64})  # fmt: skip
    batch_of = dict(zip(gt["tid"], add_batches(gt, chron)["prediction_batch"], strict=True))
    q = pl.read_parquet(sim / "reconciliation_quarantine.parquet")
    gtid = {m["id"]: m["tournament_id"] for m in conv_gap["matches"]}
    weeks = defaultdict(Counter)
    for b in batch_of.values():
        weeks[b]["events"] += 1
    for m in conv_gap["matches"]:
        weeks[batch_of[m["tournament_id"]]]["records"] += 1
    for r in q.iter_rows(named=True):
        weeks[batch_of[gtid[r["provider_match_id"]]]]["held"] += 1
    for r in led.iter_rows(named=True):
        w = weeks[r["prediction_batch"]]
        w["validated"] += 1
        w["elo_updates"] += int(r["is_update"])
    first = min(weeks) if weeks else None
    b, cov = first, []
    last = as_of - timedelta(days=(as_of.weekday() - 5) % 7)
    while b is not None and b <= last:
        w = weeks.get(b, Counter())
        cov.append({"batch": str(b), "events": w["events"], "records": w["records"], "validated": w["validated"],
                    "held": w["held"], "elo_updates": w["elo_updates"]})  # fmt: skip
        b += timedelta(days=7)
    return {"chronology": {"week_start": chron.week_start,
                           "multi_week": [{"levels": list(r.levels), "weeks": r.weeks, "min_draw_size": r.min_draw_size}
                                          for r in chron.multi_week]},
            "archive_last_tournament_date": str(hist["tournament_date"].max()),
            "overlap_events": events_overlap,
            "overlap_mapping": {"events": len(non_team), "same_batch": sum(e["same_batch"] for e in non_team),
                                "same_release": sum(e["same_release"] for e in non_team),
                                "team_events": len(team), "team_same_batch": sum(e["same_batch"] for e in team)},
            "overlap_release": overlap_release, "gap_release": gap_release, "gap_events": gap_events,
            "roland_garros": rg, "coverage_by_week": cov, "archive_2025_dating": prev_dating}  # fmt: skip

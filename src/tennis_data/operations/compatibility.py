"""Statistical compatibility of a candidate source with the archive, on matches present in both.

Field names are not trusted to mean the same thing: every counting field of the candidate source is compared value
by value with the archive (``sackmann_v1``) on matched matches before it may feed the frozen feature engine.

Matching (analysis only, nothing is written to any state):

- players are mapped with the Phase 4A identity decisions; ``verified`` and ``high_confidence`` mappings are used
  *for this comparison only* (high-confidence mappings still await approval for state use); ambiguous and unmatched
  players exclude their matches, which are counted;
- a candidate record matches an archive match with the same unordered canonical player pair and round whose
  tournament date is within ``date_tolerance_days``; zero or several such archive matches are counted, not guessed.

For every matched pair the winner, status and score are compared, then each serve field per player (aligned through
the winner). Reported per field: comparable pairs, exact agreement, mean and quantiles of the difference
(candidate − archive), missing-on-either-side counts, and the same split by completed versus retired matches.
A field is compatible only with at least ``min_matched_observations`` comparable values and an exact agreement of at
least ``min_exact_agreement``; statistics as a whole are compatible only if every field Phase 3A uses is.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl

FIELDS = {"svpt": "svpt", "1stIn": "first_in", "1stWon": "first_won", "2ndWon": "second_won", "ace": "ace",
          "df": "df", "SvGms": "sv_gms", "bpSaved": "bp_saved", "bpFaced": "bp_faced"}  # fmt: skip
# Fields the Phase 3A serve/return estimators consume (directly or through derived service points won).
REQUIRED_FIELDS = ("svpt", "1stIn", "1stWon", "2ndWon")


def _norm_score(s):
    return " ".join(s.split()) if isinstance(s, str) else None


def match_pairs(cand: pl.DataFrame, decisions: pl.DataFrame, hist: pl.DataFrame, tol_days: int) -> dict:
    """Pair candidate results (MATCH_SCHEMA) with archive matches. Returns the pairs and the accounting."""
    usable = {r["provider_player_id"]: r["canonical_player_id"]
              for r in decisions.iter_rows(named=True)
              if r["category"] in ("verified", "high_confidence") and r["canonical_player_id"] is not None}  # fmt: skip
    idx: dict = {}
    for r in hist.iter_rows(named=True):
        idx.setdefault((frozenset((r["player_a_id"], r["player_b_id"])), r["round"]), []).append(r)
    tol = timedelta(days=tol_days)
    acct = {"candidate_records": cand.height, "unmapped_player": 0, "no_archive_match": 0, "several_archive_matches": 0,
            "matched": 0}  # fmt: skip
    pairs, unmatched = [], []
    used: set = set()
    for c in cand.sort("provider_match_id").iter_rows(named=True):
        w, lo = usable.get(c["winner_provider_id"]), usable.get(c["loser_provider_id"])
        if w is None or lo is None:
            acct["unmapped_player"] += 1
            continue
        cands = [h for h in idx.get((frozenset((w, lo)), c["round"]), ())
                 if abs(h["tournament_date"] - c["event_start_date"]) <= tol]  # fmt: skip
        if not cands:
            acct["no_archive_match"] += 1
            unmatched.append({"provider_match_id": c["provider_match_id"], "level": c["tournament_level"],
                              "tournament": c["tournament_name"], "date": c["event_start_date"]})  # fmt: skip
            continue
        if len(cands) > 1:
            acct["several_archive_matches"] += 1
            continue
        h = cands[0]
        used.add(h["match_id"])
        acct["matched"] += 1
        pairs.append((c, h, w, lo))
    return {"pairs": pairs, "accounting": acct, "unmatched": unmatched, "archive_used": used}


def _field_stats(diffs: list[int], tml_missing: int, arch_missing: int) -> dict:
    n = len(diffs)
    a = np.asarray(diffs, dtype=np.float64)
    out = {"comparable": n, "exact": int((a == 0).sum()) if n else 0, "candidate_missing": tml_missing,
           "archive_missing": arch_missing}  # fmt: skip
    out["exact_agreement"] = out["exact"] / n if n else None
    out["mean_difference"] = float(math.fsum(diffs) / n) if n else None
    out["abs_difference_p50_p90_p99_max"] = ([float(np.quantile(np.abs(a), q)) for q in (0.5, 0.9, 0.99)] +
                                             [float(np.abs(a).max())]) if n else None  # fmt: skip
    out["share_candidate_higher"] = float((a > 0).mean()) if n else None
    out["share_candidate_lower"] = float((a < 0).mean()) if n else None
    return out


def compare(pairs: list, min_matched: int, min_exact: float) -> dict:
    res = {"matched_matches": len(pairs), "winner_agreement": 0, "status_agreement": 0, "score_agreement": 0,
           "status_disagreements": {}, "fields": {}, "fields_by_status": {}}  # fmt: skip
    per: dict = {f: ([], 0, 0) for f in FIELDS}
    per_status: dict = {}
    by_event: dict = {}
    swapped, core_mismatch, winner_dis = 0, 0, []
    for c, h, w, _lo in pairs:
        res["winner_agreement"] += int(h["winner_id"] == w)
        if h["winner_id"] != w:
            winner_dis.append({"tournament": c["tournament_name"], "date": str(c["event_start_date"]),
                               "round": c["round"], "candidate_score": c["score"], "archive_score": h["score"]})  # fmt: skip
        st = c["status"]
        res["status_agreement"] += int(st == h["match_status"])
        if st != h["match_status"]:
            k = f"{st}->{h['match_status']}"
            res["status_disagreements"][k] = res["status_disagreements"].get(k, 0) + 1
        res["score_agreement"] += int(_norm_score(c["score"]) == _norm_score(h["score"]))
        a_is_w = h["player_a_id"] == w

        def arch_val(side, f, h=h, a_is_w=a_is_w):
            return h[f"{'player_a' if (side == 'w') == a_is_w else 'player_b'}_{FIELDS[f]}"]

        core = [(c[f"{sd}_{f}"], arch_val(sd, f)) for sd in "wl" for f in REQUIRED_FIELDS]
        ev = by_event.setdefault(f"{c['tournament_name']} ({c['event_start_date']})", [0, 0])
        ev[0] += 1
        if any(a is not None and b is not None and a != b for a, b in core):
            core_mismatch += 1
            ev[1] += 1
            other = [(c[f"{sd}_{f}"], arch_val("l" if sd == "w" else "w", f)) for sd in "wl" for f in REQUIRED_FIELDS]
            swapped += int(all(a == b for a, b in other))
        for side in ("w", "l"):
            arch = "player_a" if (side == "w") == a_is_w else "player_b"
            for f, af in FIELDS.items():
                cv, av = c[f"{side}_{f}"], h[f"{arch}_{af}"]
                diffs, cm, am = per[f]
                if cv is None and av is None:
                    continue
                if cv is None:
                    per[f] = (diffs, cm + 1, am)
                elif av is None:
                    per[f] = (diffs, cm, am + 1)
                else:
                    diffs.append(int(cv) - int(av))
                    bucket = per_status.setdefault(st, {}).setdefault(f, [])
                    bucket.append(int(cv) - int(av))
    for f, (diffs, cm, am) in per.items():
        s = _field_stats(diffs, cm, am)
        s["compatible"] = bool(s["comparable"] >= min_matched and (s["exact_agreement"] or 0) >= min_exact)
        res["fields"][f] = s
    res["fields_by_status"] = {st: {f: {"comparable": len(v), "exact_agreement": (sum(1 for x in v if x == 0) / len(v))
                                        if v else None} for f, v in d.items()}
                               for st, d in sorted(per_status.items())}  # fmt: skip
    res["core_field_mismatch_matches"] = core_mismatch
    res["mismatches_explained_by_swapped_sides"] = swapped
    res["winner_disagreements"] = winner_dis
    events = [{"event": k, "matches": v[0], "core_mismatches": v[1], "share": v[1] / v[0]}
              for k, v in by_event.items()]  # fmt: skip
    res["events_with_mismatches"] = sorted((e for e in events if e["core_mismatches"]),
                                           key=lambda e: (-e["share"], e["event"]))  # fmt: skip
    res["events_compared"] = len(events)
    res["events_fully_consistent"] = sum(1 for e in events if not e["core_mismatches"])
    n = len(pairs)
    for k in ("winner_agreement", "status_agreement", "score_agreement"):
        res[f"{k}_share"] = res[k] / n if n else None
    res["required_fields"] = list(REQUIRED_FIELDS)
    res["statistics_compatible"] = all(res["fields"][f]["compatible"] for f in REQUIRED_FIELDS)
    return res


def study(cand: pl.DataFrame, decisions: pl.DataFrame, hist: pl.DataFrame, cfg) -> dict:
    """Full study on the overlap window (dates covered by both sources, overlap seasons only)."""
    hist = hist.filter(pl.col("season").is_in(list(cfg.overlap_seasons)))
    if cand.height == 0 or hist.height == 0:
        return {"accounting": {"candidate_records": cand.height}, "comparison": compare([], 1, 1.0),
                "window": None, "archive_matches_in_window": 0, "archive_only": 0}  # fmt: skip
    lo, hi = cand["event_start_date"].min(), cand["event_start_date"].max()
    lo = max(lo, hist["tournament_date"].min())
    hi = min(hi, hist["tournament_date"].max())
    cand = cand.filter(pl.col("event_start_date").is_between(lo, hi))
    hist_w = hist.filter(pl.col("tournament_date").is_between(lo - timedelta(days=cfg.date_tolerance_days),
                                                              hi + timedelta(days=cfg.date_tolerance_days)))  # fmt: skip
    m = match_pairs(cand, decisions, hist_w, cfg.date_tolerance_days)
    comp = compare(m["pairs"], cfg.min_matched_observations, cfg.min_exact_agreement)
    arch_only = hist_w.filter(~pl.col("match_id").is_in(list(m["archive_used"])))
    unmatched_levels: dict = {}
    for u in m["unmatched"]:
        unmatched_levels[u["level"]] = unmatched_levels.get(u["level"], 0) + 1
    return {"window": [str(lo), str(hi)], "accounting": m["accounting"], "comparison": comp,
            "archive_matches_in_window": hist_w.height, "archive_only": arch_only.height,
            "archive_only_by_level": dict(sorted(arch_only.group_by("tournament_level").len().iter_rows())),
            "candidate_unmatched_by_level": dict(sorted(unmatched_levels.items()))}  # fmt: skip

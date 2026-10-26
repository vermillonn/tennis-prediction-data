"""Canonical match reconciliation for newly acquired records.

Deterministic policy (``rec-1``), applied to *all* ingested provider records at once, so re-running on the same
raw snapshots gives the same result whatever the delivery order:

1. Structurally invalid records (``schema.check_matches``) are quarantined.
2. Byte-identical copies are collapsed (first retrieval kept as ``ingested_at``).
3. Versions: per (provider, provider_match_id) the highest ``source_version`` is current; older versions are kept
   as ``superseded``. Two different contents under the same version are a conflict (``version_reissued``).
4. Identity: both players must have a usable identity decision, otherwise the record is held
   (``identity_unresolved``) until a reviewer decides.
5. Tournament identity: records are clustered by normalized name, level, year and start date (within
   ``date_tolerance_days``). A cluster matching an archive tournament uses its ``tournament_id``; otherwise the
   ID is ``<year>-<provider>-<provider_tournament_id>`` of the highest-precedence provider in the cluster.
6. Match key: (tournament_id, round, unordered canonical player pair).
7. A key already in the pinned archive is never overwritten: identical outcome -> ``duplicate_of_historical``
   (dropped); different outcome or score -> ``contradicts_historical`` (quarantined).
8. Several providers for one key: identical winner and status -> the highest-precedence record is accepted and
   statistic differences are logged (``stats_discrepancy``); different outcomes or start dates beyond the tolerance
   -> all are quarantined (``cross_source_conflict``).
9. Serve statistics with a counting convention other than ``sackmann_v1`` are removed (outcome kept) and logged.

Accepted records become Sackmann-format raw rows and pass the unchanged Phase 1 validation (``validate_frame``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import polars as pl

from tennis_data.normalization.schema import SOURCE_MATCH_COLUMNS
from tennis_data.normalization.score import match_status
from tennis_data.prospective.config import ProspectiveConfig
from tennis_data.prospective.identity import norm
from tennis_data.prospective.schema import COMPATIBLE_STAT_DEFINITIONS, MATCH_SCHEMA, STATS, check_matches

CONTENT_COLS = [c for c in MATCH_SCHEMA if c not in ("raw_sha256", "retrieved_at")]
QUARANTINE_SCHEMA = {
    "provider": pl.String, "provider_match_id": pl.String, "source_version": pl.Int64, "reason": pl.String,
    "detail": pl.String, "raw_sha256": pl.String,
}  # fmt: skip


def _content_key(r: dict) -> str:
    return json.dumps({k: r[k] for k in CONTENT_COLS}, sort_keys=True, default=str)


def _q(r: dict, reason: str, detail="") -> dict:
    return {"provider": r["provider"], "provider_match_id": r["provider_match_id"], "source_version": r["source_version"],
            "reason": reason, "detail": str(detail), "raw_sha256": r["raw_sha256"]}  # fmt: skip


def _rank(cfg: ProspectiveConfig, provider: str) -> int:
    return cfg.precedence.index(provider) if provider in cfg.precedence else len(cfg.precedence)


def _score_for(r: dict) -> str | None:
    s, st = r["score"], r["status"]
    if st == "walkover":
        return s if s and match_status(s) == "walkover" else "W/O"
    if st == "retired" and s and match_status(s) != "retired":
        return f"{s} RET"
    if st == "default" and s and match_status(s) != "default":
        return f"{s} DEF"
    return s


def _historical_keys(hist: pl.DataFrame) -> dict:
    out = {}
    for r in hist.select("tournament_id", "round", "player_a_id", "player_b_id", "winner_id", "match_id", "score",
                         "tournament_name", "tournament_level", "tournament_date").iter_rows(named=True):  # fmt: skip
        out[(r["tournament_id"], r["round"], frozenset((r["player_a_id"], r["player_b_id"])))] = r
    return out


def _tournaments(hist: pl.DataFrame) -> list[dict]:
    t = hist.group_by("tournament_id").agg(pl.col("tournament_name").first(), pl.col("tournament_level").first(),
                                           pl.col("tournament_date").min())  # fmt: skip
    return t.sort("tournament_id").to_dicts()


def reconcile(matches: pl.DataFrame, decisions: pl.DataFrame, hist: pl.DataFrame, cfg: ProspectiveConfig) -> dict:
    """matches: MATCH_SCHEMA rows of every ingested snapshot. hist: Phase 1 validated table (pinned archive)."""
    tol = timedelta(days=cfg.date_tolerance_days)
    quarantine, versions, notes = [], [], []
    errs = {(e.provider, e.record_id) for e in check_matches(matches)}
    err_detail = {}
    for e in check_matches(matches):
        err_detail.setdefault((e.provider, e.record_id), []).append(e.code)
    rows = sorted(matches.iter_rows(named=True),
                  key=lambda r: (r["provider"], r["provider_match_id"], r["source_version"] or 0, str(r["retrieved_at"]),
                                 r["raw_sha256"]))  # fmt: skip
    # 1-2: structure and exact copies
    seen, clean = {}, []
    for r in rows:
        if (r["provider"], r["provider_match_id"]) in errs:
            quarantine.append(
                _q(r, "invalid_record", ",".join(sorted(set(err_detail[(r["provider"], r["provider_match_id"])]))))
            )
            continue
        ck = _content_key(r)
        if ck in seen:
            continue
        seen[ck] = r
        clean.append(r)
    # 3: versions
    by_id: dict = {}
    for r in clean:
        by_id.setdefault((r["provider"], r["provider_match_id"]), []).append(r)
    current = []
    for _key, vs in sorted(by_id.items()):
        vnums = [v["source_version"] for v in vs]
        if len(vnums) != len(set(vnums)):
            for v in vs:
                quarantine.append(_q(v, "version_reissued", "same source_version with different content"))
            continue
        vs = sorted(vs, key=lambda v: v["source_version"])
        for v in vs[:-1]:
            versions.append({**{k: v[k] for k in ("provider", "provider_match_id", "source_version", "raw_sha256")},
                             "state": "superseded", "superseded_by": vs[-1]["source_version"]})  # fmt: skip
        versions.append({**{k: vs[-1][k] for k in ("provider", "provider_match_id", "source_version", "raw_sha256")},
                         "state": "current", "superseded_by": None})  # fmt: skip
        current.append(vs[-1])
    # 4: identity
    ident = {(d["provider"], d["provider_player_id"]): d for d in decisions.iter_rows(named=True)}
    resolved = []
    for r in current:
        dw, dl = ident.get((r["provider"], r["winner_provider_id"])), ident.get((r["provider"], r["loser_provider_id"]))
        bad = [d for d in (dw, dl) if d is None or not d["usable"]]
        if bad:
            cats = sorted(d["category"] if d else "no_decision" for d in bad)
            quarantine.append(_q(r, "identity_unresolved", ",".join(cats)))
            continue
        resolved.append({**r, "winner_id": dw["canonical_player_id"], "loser_id": dl["canonical_player_id"]})
    # 5: tournaments
    hist_t = _tournaments(hist)
    clusters: list[dict] = []
    for r in sorted(resolved, key=lambda r: (_rank(cfg, r["provider"]), r["provider"], r["provider_tournament_id"])):
        if r["provider"] == "sackmann_csv":
            r["tournament_id"] = r["provider_tournament_id"]
            continue
        name, lvl, start = norm(r["tournament_name"]), r["tournament_level"], r["event_start_date"]
        hit = None
        for c in clusters:
            if c["name"] == name and c["level"] == lvl and abs(c["start"] - start) <= tol:
                hit = c
                break
        if hit is None:
            hid = next((t["tournament_id"] for t in hist_t if norm(t["tournament_name"]) == name
                        and t["tournament_level"] == lvl and abs(t["tournament_date"] - start) <= tol), None)  # fmt: skip
            hit = {"name": name, "level": lvl, "start": start,
                   "id": hid or f"{start.year}-{r['provider']}-{r['provider_tournament_id']}"}  # fmt: skip
            clusters.append(hit)
        r["tournament_id"] = hit["id"]
    # 6-8: match keys
    hkeys = _historical_keys(hist)
    groups: dict = {}
    for r in resolved:
        key = (r["tournament_id"], r["round"], frozenset((r["winner_id"], r["loser_id"])))
        groups.setdefault(key, []).append(r)
    accepted = []
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], sorted(kv[0][2]))):
        h = hkeys.get(key)
        if h is not None:
            for r in rs:
                same = r["winner_id"] == h["winner_id"] and (r["score"] or "") == (h["score"] or "")
                if same:
                    notes.append({**_q(r, "duplicate_of_historical", h["match_id"])})
                else:
                    quarantine.append(_q(r, "contradicts_historical", h["match_id"]))
            continue
        rs = sorted(rs, key=lambda r: (_rank(cfg, r["provider"]), r["provider"], r["provider_match_id"]))
        outcomes = {(r["winner_id"], r["status"]) for r in rs}
        spread = max(r["event_start_date"] for r in rs) - min(r["event_start_date"] for r in rs)
        if len(outcomes) > 1 or spread > tol:
            for r in rs:
                quarantine.append(
                    _q(r, "cross_source_conflict", f"outcomes={len(outcomes)} date_spread={spread.days}d")
                )
            continue
        best = dict(rs[0])
        for other in rs[1:]:
            diff = [s for s in STATS for side in ("w", "l") if best[f"{side}_{s}"] != other[f"{side}_{s}"]]
            notes.append(_q(other, "secondary_source_agrees" if not diff else "stats_discrepancy",
                            ",".join(sorted(set(diff)))))  # fmt: skip
        if best["stats_definition"] not in COMPATIBLE_STAT_DEFINITIONS and any(
            best[f"{side}_{s}"] is not None for side in ("w", "l") for s in STATS
        ):
            notes.append(_q(best, "stats_definition_incompatible", best["stats_definition"]))
            for side in ("w", "l"):
                for s in STATS:
                    best[f"{side}_{s}"] = None
        score = _score_for(best)
        if score is not None and match_status(score) != best["status"]:
            quarantine.append(_q(best, "status_score_mismatch", f"{best['status']} vs {score!r}"))
            continue
        best["score_out"] = score
        best["n_sources"] = len(rs)
        accepted.append(best)
    return {
        "accepted": accepted,
        "quarantine": pl.DataFrame(quarantine, schema=QUARANTINE_SCHEMA).sort(
            "provider", "provider_match_id", "reason"
        ),
        "notes": pl.DataFrame(notes, schema=QUARANTINE_SCHEMA).sort("provider", "provider_match_id", "reason"),
        "versions": pl.DataFrame(
            versions,
            schema={
                "provider": pl.String,
                "provider_match_id": pl.String,
                "source_version": pl.Int64,
                "raw_sha256": pl.String,
                "state": pl.String,
                "superseded_by": pl.Int64,
            },
        ),  # fmt: skip
    }


def match_num(tournament_id: str, round_: str, a: int, b: int) -> int:
    """Stable match number from the match key (independent of which other matches are known)."""
    lo, hi = sorted((a, b))
    h = hashlib.sha256(f"{tournament_id}|{round_}|{lo}|{hi}".encode()).hexdigest()
    return 100000 + int(h[:7], 16) % 900_000_000


def to_raw_rows(accepted: list[dict], master_names: dict, cfg: ProspectiveConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sackmann-format raw rows (as ``read_raw_file`` returns) plus a provenance table keyed by match_id."""
    raws, prov = [], []
    for i, r in enumerate(sorted(accepted, key=lambda r: (r["tournament_id"], r["round"], r["provider_match_id"]))):
        num = match_num(r["tournament_id"], r["round"], r["winner_id"], r["loser_id"])
        mid = f"{cfg.backfill_family}:{r['tournament_id']}:{num}"
        raw = {c: None for c in SOURCE_MATCH_COLUMNS}

        def s(v):
            return None if v is None else str(v)

        raw.update(tourney_id=r["tournament_id"], tourney_name=r["tournament_name"], surface=r["surface"],
                   draw_size=s(r["draw_size"]), tourney_level=r["tournament_level"],
                   tourney_date=r["event_start_date"].strftime("%Y%m%d"), match_num=str(num),
                   winner_id=str(r["winner_id"]), loser_id=str(r["loser_id"]),
                   winner_name=master_names.get(r["winner_id"]), loser_name=master_names.get(r["loser_id"]),
                   score=r["score_out"], best_of=s(r["best_of"]), round=r["round"], minutes=s(r["minutes"]))  # fmt: skip
        for st in STATS:
            raw[f"w_{st}"], raw[f"l_{st}"] = s(r[f"w_{st}"]), s(r[f"l_{st}"])
        raw.update(source_family=cfg.backfill_family, source_repo=f"prospective:{r['provider']}",
                   source_commit=r["raw_sha256"], source_file=f"{r['provider']}/{r['provider_match_id']}@v{r['source_version']}",
                   source_row_number=i + 1, source_sha256=r["raw_sha256"], season=r["event_start_date"].year)  # fmt: skip
        raws.append(raw)
        prov.append({"match_id": mid, "provider": r["provider"], "provider_match_id": r["provider_match_id"],
                     "source_version": r["source_version"], "raw_sha256": r["raw_sha256"],
                     "event_start_date": r["event_start_date"], "event_start_at": r["event_start_at"],
                     "event_completed_at": r["event_completed_at"], "source_published_at": r["source_published_at"],
                     "ingested_at": r["retrieved_at"], "n_sources": r["n_sources"],
                     "stats_definition": r["stats_definition"]})  # fmt: skip
    schema = {c: pl.String for c in SOURCE_MATCH_COLUMNS} | {
        "source_family": pl.String, "source_repo": pl.String, "source_commit": pl.String, "source_file": pl.String,
        "source_row_number": pl.Int64, "source_sha256": pl.String, "season": pl.Int32,
    }  # fmt: skip
    prov_schema = {"match_id": pl.String, "provider": pl.String, "provider_match_id": pl.String,
                   "source_version": pl.Int64, "raw_sha256": pl.String, "event_start_date": pl.Date,
                   "event_start_at": MATCH_SCHEMA["event_start_at"], "event_completed_at": MATCH_SCHEMA["event_completed_at"],
                   "source_published_at": MATCH_SCHEMA["source_published_at"], "ingested_at": MATCH_SCHEMA["retrieved_at"],
                   "n_sources": pl.Int64, "stats_definition": pl.String}  # fmt: skip
    nums = [r["match_num"] for r in raws]
    if len(set(zip([r["tourney_id"] for r in raws], nums, strict=True))) != len(raws):
        raise AssertionError("match_num collision in backfill rows")
    return pl.DataFrame(raws, schema=schema, orient="row"), pl.DataFrame(prov, schema=prov_schema, orient="row")

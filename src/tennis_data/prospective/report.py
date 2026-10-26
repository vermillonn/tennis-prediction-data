"""Phase 4A generated reports (deterministic; runtimes go to ``runtime.json`` only).

``source_feasibility.md`` is a hand-written research document (it records access attempts and documentation) and is
not generated here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl

from tennis_data.prospective.identity import Master, classify
from tennis_data.prospective.schema import PLAYER_SCHEMA, frame

LICENSE = "Derived from Jeff Sackmann's ATP data (CC BY-NC-SA 4.0); aggregate statistics only, no row-level data."


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _count(rows, key="len"):
    return sum(r[key] for r in rows)


def status_line(cfg) -> str:
    return (f"Status: historical backfill **{cfg.backfill_status}** · prospective predictions "
            f"**{cfg.prospective_predictions}** · real-provider integrations **UNVERIFIED** "
            "(gate: `config/prospective.toml` `[status]`).")  # fmt: skip


def backfill_coverage(res) -> str:
    a, bf, cfg = res["audit"], res["backfill"]["summary"], res["cfg"]
    yr = a["year_rows"]
    est = a["estimate_from_previous_year"]
    mp = a["missing_period"]
    cats = {r["category"]: r["len"] for r in yr["by_category"]}
    ecats = {r["category"]: r["len"] for r in est["by_category"]}
    gap_rows = sum(r["len"] for r in bf["by_period"] if r["period"] == "backfill_gap")
    lines = [
        "# Phase 4A backfill coverage: the historical gap to the prospective start", "", LICENSE, "",
        "## 1. Verdict", "", status_line(cfg), "",
        f"**The gap is not filled.** The pinned archive (commit `{a['archive_commit'][:12]}`) ends with events starting "
        f"on {a['last_tournament_date']}. No result after that is available to this system: {mp['weeks']} weekly "
        f"batches ({mp['first_missing_batch']} to {mp['last_batch_before_prospective']}) contain no newly starting "
        "events (the first of them is only the second week of Roland Garros 2026), and the "
        f"backfill accepted {gap_rows} records for that period. No authorized source was reachable "
        "(`source_feasibility.md`). The player state therefore stops at the archive end.", "",
        "## 2. Final valid coverage of the archive (verified from the validated table)", "",
        f"- Last tournament start date: **{a['last_tournament_date']}** (configured snapshot end "
        f"{a['configured_snapshot_end']}; {'matches' if a['snapshot_end_verified'] else 'DOES NOT match'}). The "
        "end date differs between levels:", "",
        table(["level", "last event start"], [[r["tournament_level"], r["tournament_date"]]
                                               for r in a["last_tournament_date_by_level"]]),
        "",
        f"- Last batch whose close updates Elo: {a['last_update_release_batch']} (the second week of the last "
        f"two-week event). Last batch releasing usable point statistics: **{a['last_stats_release_batch']}**.", "",
        "Events in the final five weeks (knockout completeness = rows against draw size − 1):", "",
        table(["event", "level", "start", "draw", "rows", "expected", "completeness", "usable stats", "update Elo",
               "weeks", "release batch"],
              [[e["tournament_name"], e["tournament_level"], e["tournament_date"], e["draw_size"], e["rows"],
                e["expected_main_draw_matches"], e["completeness"], e["rows_with_usable_stats"], e["rows_updating_elo"],
                e["event_weeks"], e["release_batch"]] for e in a["final_weeks_events"]]),  # fmt: skip
        "",
        "The last event (Roland Garros 2026) is complete as a results record but has **no serve statistics** in the "
        "archive, so it updates Elo but not the serve/return state.", "",
        f"## 3. {cfg.historical_snapshot_end.year} rows in the archive by evidence category", "",
        table(["category", "rows"], [
            ["1. complete observation (target + usable statistics)", cats.get("1_complete_observation", 0)],
            ["2. usable outcome, missing point statistics", cats.get("2_outcome_without_statistics", 0)],
            ["3. incomplete or ambiguous record (quarantined)", yr.get("quarantined", 0)],
            ["4. excluded by the frozen eligibility rules (walkovers, retirements, unknown status)",
             cats.get("4_excluded_by_eligibility", 0)],
        ]),  # fmt: skip
        "",
        table(["match status", "rows"], [[r["match_status"], r["len"]] for r in yr["by_status"]]), "",
        table(["statistics exclusion", "rows"], [[r["stats_exclusion_reason"], r["len"]] for r in yr["stats_exclusions"]]),
        "",
        f"- Multi-week events in the year: {yr['multi_week_events']}. Players not in the player master: "
        f"{yr['players_not_in_master']}. Duplicate or conflicting rows quarantined: {yr.get('quarantined', 0)}.",
        "- Timestamps: every row has `time_precision = tournament_week` (event start date only). No match date or time "
        "exists, so all timing is the conservative weekly-batch inference.", "",
        "## 4. What is missing (estimate)", "",
        f"No 2026 calendar for the missing weeks is available, so the volume is estimated from the same {mp['weeks']} "
        f"weekly batches one year earlier ({est['window_batches'][0]} to {est['window_batches'][1]}). This is an "
        "estimate, not a list of 2026 events:", "",
        table(["level", "events", "matches"], [[r["tournament_level"], r["events"], r["matches"]]
                                              for r in est["events_by_level"]]),
        "",
        f"- Estimated missing tour-level matches: **~{est['matches']}** in ~{est['events']} events, including the Grand "
        f"Slams {', '.join(est['grand_slams'])} and multi-week events {', '.join(est['multi_week_events'])}.",
        f"- In the comparison year, {ecats.get('1_complete_observation', 0)} were complete observations, "
        f"{ecats.get('2_outcome_without_statistics', 0)} outcome-only and {ecats.get('4_excluded_by_eligibility', 0)} "
        "excluded by eligibility; a backfill of similar quality would yield usable statistics for "
        f"~{100 * ecats.get('1_complete_observation', 0) / max(1, est['matches']):.0f}% of matches.", "",
        "## 5. Backfill run on the current raw store", "",
        table(["item", "value"], [
            ["raw snapshot id", bf["raw_snapshot_id"]], ["raw snapshots", len(bf["raw_snapshots"])],
            ["provider match records", bf["provider_records"]], ["accepted after reconciliation", bf["accepted_records"]],
            ["Phase 1 validated", bf["phase1_validated"]], ["Phase 1 quarantined", bf["phase1_quarantined"]],
            ["with complete statistics", bf["with_complete_stats"]], ["state-eligible Elo updates", bf["state_eligible_updates"]],
            ["state-eligible statistics", bf["stats_eligible"]], ["state snapshot id", bf["state_snapshot_id"]],
        ]),  # fmt: skip
        "",
        "Reconciliation outcomes: " + (", ".join(f"{r['reason']} {r['len']}" for r in bf["reconciliation_quarantine"])
                                       or "none") + ".",
        "",
        "## 6. Chronology of backfilled results", "",
        "Each accepted result gets a ledger row (`data/prospective/backfill/ledger.parquet`) with `event_start_date`, "
        "`event_start_at` and `event_completed_at` (only if the provider states them), `source_published_at`, "
        "`ingested_at`, `release_batch` and `state_eligible_at` = 00:00 UTC on the day after the release batch "
        "closes (`eligibility_basis = inferred_weekly_batch`). Multi-week events release one week later, exactly as "
        "in Phase 2. Every backfilled row has `prediction_mode = historical_reconstruction`: a result discovered now "
        "is never presented as information available, or forecast, on its event date.", "",
    ]  # fmt: skip
    return "\n".join(lines)


HARD_CASES = [
    ("accent + hyphen", {"first_name": "Félix", "last_name": "Auger-Aliassime", "country": "CAN"}),
    ("compound surname", {"first_name": "Pablo", "last_name": "Carreno-Busta", "country": "ESP"}),
    ("particle surname", {"first_name": "Juan Martin", "last_name": "Del Potro", "country": "ARG"}),
    ("abbreviated first name", {"first_name": "A.", "last_name": "Zverev"}),
    ("brothers, full name + DOB", {"first_name": "Alexander", "last_name": "Zverev", "birth_date": "1997-04-20"}),
    ("same name, no DOB", {"first_name": "Carlos", "last_name": "Alcaraz", "country": "ESP"}),
    ("same name + DOB", {"first_name": "Carlos", "last_name": "Alcaraz", "birth_date": "2003-05-05"}),
    ("declared ID contradicts name", {"first_name": "Jannik", "last_name": "Sinner", "declared_canonical_id": 207989}),
    ("declared ID consistent", {"first_name": "Jannik", "last_name": "Sinner", "declared_canonical_id": 206173}),
    ("changed nationality, DOB match", {"first_name": "Daniil", "last_name": "Medvedev", "country": "NIL",
                                        "birth_date": "1996-02-11"}),
    ("DOB conflict", {"first_name": "Daniil", "last_name": "Medvedev", "birth_date": "1999-01-01"}),
    ("unseen player", {"first_name": "Qwerty", "last_name": "Zyxwvutsrq"}),
]  # fmt: skip


def identity_demo(cfg, archive: pl.DataFrame) -> list[list]:
    from datetime import date

    master = Master.build(archive, pl.DataFrame(schema={"canonical_player_id": pl.Int64, "first_name": pl.String,
                                                        "last_name": pl.String, "country": pl.String,
                                                        "birth_date": pl.Date}))  # fmt: skip
    rows = []
    for i, (label, p) in enumerate(HARD_CASES):
        rec = {k: None for k in PLAYER_SCHEMA} | {"provider": "demo", "provider_player_id": str(i)} | p
        if isinstance(rec["birth_date"], str):
            rec["birth_date"] = date.fromisoformat(rec["birth_date"])
        d = classify(frame([rec], PLAYER_SCHEMA).row(0, named=True), master, {}, {}, cfg)
        rows.append(
            [
                label,
                f"{p.get('first_name')} {p.get('last_name')}",
                d["category"],
                d["method"],
                d["canonical_player_id"] or "–",
                len(d["candidates"]),
                "yes" if d["usable"] else "no",
            ]
        )
    return rows


def identity_report(res) -> str:
    from tennis_data.prospective.identity import load_archive_players

    cfg, bf = res["cfg"], res["backfill"]
    dec = bf["decisions"]
    counts = dec.group_by("category", "usable").len().sort("category", "usable").to_dicts() if dec.height else []
    lines = [
        "# Phase 4A identity reconciliation", "", LICENSE, "",
        "## 1. Policy (`id-1`)", "",
        "Provider player IDs are mapped to canonical (archive) player IDs by `prospective/identity.py`:", "",
        table(["evidence", "category", "usable in state"], [
            ["reviewed entry in `config/identity_overrides.toml`", "verified (manual_override)", "yes"],
            ["provider ID already registered as a new player", "unmatched (new_player, same ID)", "yes"],
            ["provider-declared canonical ID that exists and whose name agrees (no DOB conflict)", "verified", "yes"],
            ["unique exact name, confirmed by birth date (or nationality when no DOB and no other candidate)",
             "high_confidence", "only after approval" + (" (auto-approve ON)" if cfg.auto_approve_high_confidence else "")],
            ["several candidates, abbreviated/partial name only, declared ID contradicted, DOB conflict",
             "ambiguous", "never (held for review)"],
            ["no canonical player shares the surname", "unmatched -> new canonical ID", "yes" if cfg.accept_unmatched_as_new
             else "no"],
        ]),  # fmt: skip
        "",
        "- Names are compared accent-, case- and punctuation-insensitively; compound surnames match on full or "
        "contained token sets (`Carreno Busta` / `Carreño-Busta`).",
        "- A differing birth date eliminates a candidate. Nationality never does (players change representation); it "
        "is recorded as evidence.",
        f"- New players receive IDs from {cfg.new_player_id_base:,} upward in sorted provider-ID order; the registry "
        "(`data/prospective/identity/new_players.parquet`) is append-only and archive IDs are never changed.",
        "- Every decision is persisted with method, candidates and evidence (`identity/decisions.parquet`).", "",
        "## 2. Decisions in the current backfill", "",
        (table(["category", "usable", "players"], [[r["category"], r["usable"], r["len"]] for r in counts]) if counts
         else "No provider player records were ingested (no authorized source accessible), so there are no live "
              "decisions and **0 unresolved identities**. That is an absence of data, not evidence that matching works "
              "on a real provider."),
        "",
        "## 3. Behaviour on deliberately hard cases against the real player master", "",
        "Hypothetical provider records (not ingested data) classified against the archive's 66,912 players:", "",
        table(["case", "provider name", "category", "method", "canonical id", "candidates", "usable"],
              identity_demo(cfg, load_archive_players(cfg))),
        "",
        "The same behaviours are asserted in `tests/test_prospective_units.py` on a controlled master that contains "
        "identical names, brothers, compound and particle surnames and players without metadata.", "",
        "## 4. Uncertainty", "",
        "- Real providers may spell names in ways not covered here (transliteration of Cyrillic and Chinese names "
        "varies). Such cases become ambiguous, not wrong: they block the match until reviewed.",
        "- Birth dates are the strongest secondary evidence, but the archive lacks them for some players; then only "
        "nationality and the absence of other candidates support a high-confidence match, which still needs approval.",
        "- Matching quality on any real provider is unmeasured until its data can be accessed.", "",
    ]  # fmt: skip
    return "\n".join(lines)


def readiness_report(res) -> str:
    cfg, rp, integ, fresh, ready = (
        res["cfg"],
        res["replay"]["summary"],
        res["integrity"],
        res["freshness"],
        res["readiness"],
    )
    srec = rp["state_reconstruction"]
    lines = [
        "# Phase 4A inference readiness", "", LICENSE, "",
        "## 1. Assessment", "",
        f"**Ready for genuine prospective predictions: {'YES' if ready['ready_for_prospective_predictions'] else 'NO'}.**",
        "",
        status_line(cfg),
        "",
        table(["component", "status"], [[k, "pass" if v else "**not met**"] for k, v in ready["components"].items()]),
        "", "Blockers:", "", *[f"- {b}" for b in ready["blockers"]], "",
        "The inference machinery is verified; the data it needs is not available. M4A must not be used for prospective "
        "forecasts until the backfill reaches the prediction date and a verified provider supplies schedules.", "",
        "## 2. Frozen model integrity", "",
        table(["check", "value"], [
            ["integrity checks passed", integ["passed"]], ["model", integ.get("model_id")],
            ["frozen hash", integ.get("frozen_hash", "–")[:16]], ["expected frozen hash", cfg.expected_frozen_hash[:16]],
            ["modeling configuration hash", integ.get("modeling_configuration_hash")],
            ["production artifact sha256", (integ.get("production_artifact_sha256") or "–")[:16]],
            ["inputs", ", ".join(integ.get("inputs", []))], ["calibration", integ.get("calibration")],
            ["production training cutoff", integ.get("training_cutoff")],
        ]),  # fmt: skip
        "",
        "Before every prediction the inference path verifies the SHA-256 of `frozen_model.json` and "
        "`production_model.json` against the Phase 3B manifest, the frozen hash (and that it equals the configured "
        "one), and that the modeling configuration hash is unchanged. Any mismatch raises `ArtifactIntegrityError` "
        "and no probability is produced. Nothing is fitted at inference time.", "",
        "## 3. Historical replay", "",
        f"{rp['cases']} historical matches were re-submitted as pending upcoming matches (no outcome, no updates) and "
        "priced through the new inference path with the frozen M4A annual fit of their year, then compared with the "
        "published Phase 3B predictions and Phase 2 baselines:", "",
        table(["metric", "value"], [
            ["max abs(p − p_published), M4A", f"{rp['max_abs_error']:.2e}"],
            ["max abs(p_elo − p_published), surface Elo", f"{rp['max_elo_abs_error']:.2e}"],
            ["max abs(P(A) + P(B) − 1)", f"{rp['max_symmetry_error']:.2e}"],
            ["tolerance", f"{rp['tolerance']:.0e}"], ["years covered", f"{rp['years'][0]}–{rp['years'][-1]}"],
            ["passed", rp["passed"]],
        ]),  # fmt: skip
        "",
        table(["category", "cases", "max M4A error", "max Elo error"],
              [[c["category"], c["n"], f"{c['max_abs_error']:.1e}", f"{c['max_elo_abs_error']:.1e}"]
               for c in rp["by_category"]]),
        "",
        "Categories overlap (a match can be both clay and Grand Slam). `multi_week_boundary` = a player has an earlier "
        "match whose result is not yet released at the prediction batch (week two of a two-week event).", "",
        "State reconstruction over the whole archive in the same run:", "",
        table(["check", "value"], [
            ["historical targets with an Elo prediction", f"{srec['elo_rows']:,} (unmatched {srec['elo_unmatched_rows']})"],
            ["max abs(Elo − published Phase 2)", f"{srec['elo_max_abs_error']:.1e}"],
            ["Phase 3A feature table identical", f"{srec['features_identical']} ({srec['feature_rows']:,} rows × "
                                                 f"{srec['feature_columns']} columns)"],
        ]),  # fmt: skip
        "",
        "## 4. Feature freshness for a match after the prospective start", "",
        f"Diagnostic probe (no probability is reported): a tour-level hard-court match on {res['probe'].tournament_date} "
        "between two players from the archive's last event, with the state as it stands.", "",
        f"Overall status: **{fresh['status'].upper()}**", "",
        table(["check", "value", "status"],
              [[k, json.dumps({kk: vv for kk, vv in v.items() if kk != "status"}, default=str), v["status"]]
               for k, v in fresh["checks"].items()]),  # fmt: skip
        "",
        "`missing_backfill_weeks` counts weekly batches after the last week with results and before the probe's batch. "
        "`opponent_fit_age_weeks` measures the refit schedule only: the opponent model refits when due, but on the "
        "same stale observations, so a fresh fit does not mean fresh data. `source_age_days` is null because no "
        "provider snapshot exists.", "",
        "Thresholds (`[freshness]` in `config/prospective.toml`): " + ", ".join(
            f"{k} {v}" for k, v in cfg.freshness.__dict__.items()) + ". A `fail` blocks prospective predictions. "
        "Freshness never changes a probability: model probability and operational reliability are reported "
        "separately.", "",
        "## 5. Inference contract", "",
        "Input: canonical player IDs, surface, tournament level and ID, event start date, round, draw size, best-of, "
        "a verified scheduled start time if any, and the state snapshot. Output per match: P(A), P(B) (= 1 − P(A), "
        "checked to 1e-12), the surface-Elo baseline, model ID, frozen hash, artifact checksum, feature-state hash, "
        "state snapshot and source snapshot IDs, feature availability (which inputs used the frozen 0 fallback), "
        "freshness, `prediction_mode` (`historical_reconstruction`, `exploratory_retrospective`, `prospective`) and "
        "the generation time.", "",
        "## 6. Runtime", "",
        "Runtime and memory of the run that produced this report are in `runtime.json` (not part of the reproducible "
        "report).", "",
    ]  # fmt: skip
    return "\n".join(lines)


def write_reports(res) -> dict:
    cfg = res["cfg"]
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    docs = {"backfill_coverage.md": backfill_coverage(res), "identity_reconciliation.md": identity_report(res),
            "inference_readiness.md": readiness_report(res)}  # fmt: skip
    for name, text in docs.items():
        (out / name).write_text(text)
    summary = {k: res["manifest"][k] for k in ("prospective_configuration_hash", "code_hash", "frozen_hash",
                                                "state_snapshot_id", "raw_snapshot_id", "readiness")}  # fmt: skip
    summary["audit"] = res["audit"]
    summary["replay"] = {k: v for k, v in res["replay"]["summary"].items() if not k.startswith("seconds")}
    (out / "phase4a_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    perf = dict(res["performance"])
    perf["report_written_at"] = datetime.now(UTC).isoformat()
    (out / "runtime.json").write_text(json.dumps(perf, indent=2, sort_keys=True) + "\n")
    return {n: out / n for n in [*docs, "phase4a_summary.json", "runtime.json"]}

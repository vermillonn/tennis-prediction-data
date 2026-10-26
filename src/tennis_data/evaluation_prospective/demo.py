"""Deterministic SYNTHETIC demonstration: a ledger written through the real ``LedgerWriter``, an outcome snapshot
and local-fixture anchors, exercising every lifecycle and outcome case of the evaluation.

Everything here is synthetic and labelled so: SYNTHETIC and HISTORICAL_REPLAY modes only (the writer refuses
PROSPECTIVE while the status gate is DISABLED), ``synthetic:``/``replay:`` match ids, synthetic provider names, an
injected clock (``audit.clock = "injected"``) and fixture anchors that never count as external evidence. Probabilities
come from a seeded generator (true probability, noisier "M4A" and noisier "Elo"), not from the frozen model.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from tennis_data.evaluation_prospective import anchoring
from tennis_data.evaluation_prospective.config import EvaluationConfig
from tennis_data.evaluation_prospective.safeio import write_new
from tennis_data.ledger.checkpoint import make_checkpoint, write_checkpoint
from tennis_data.ledger.hashing import canonical, ts
from tennis_data.ledger.writer import FixedClock, LedgerWriter, PredictionRequest, orient

T_START = datetime(2026, 11, 2, tzinfo=UTC)  # a Monday after the prospective start; synthetic calendar only
LEVELS = ("A", "M", "A", "G", "A", "M")
SURFACES = ("Hard", "Clay", "Grass", "Hard")


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def demo_protocol(cfg: EvaluationConfig) -> dict:
    """The live protocol with the cohort switched to SYNTHETIC (a different protocol hash; never PROSPECTIVE)."""
    return {**cfg.protocol, "version": cfg.protocol["version"] + "+synthetic-demo", "cohort_mode": "SYNTHETIC"}


def _request(cfg, mid, p1, p2, prob1, elo1, start, slot, sched_id, created, level, w, j, rng, mode="SYNTHETIC"):
    o = orient(mid, p1, p2, prob1, elo1)
    sp = [int(x) for x in rng.integers(0, 9000, 2)]
    em = [int(x) for x in rng.integers(0, 400, 2)]
    return PredictionRequest(
        mode=mode, forecast_slot=slot, match_id=mid, tournament_id=f"synthetic-t{w:02d}", surface=SURFACES[w % 4],
        round="R32", best_of=5 if level == "G" else 3, scheduled_start_at=ts(start), scheduled_start_verified=True,
        schedule_snapshot_id=sched_id, schedule_retrieved_at=ts(created - timedelta(hours=1)), model_id="M4A+raw",
        model_version="m3b-1+engine1", frozen_hash=cfg.ledger.prospective.expected_frozen_hash + "0" * 48,
        configuration_hash=cfg.protocol["approved_configuration_hash"], feature_schema_version="sr-1+engine2",
        state_snapshot_id=f"synthetic-state-w{w:02d}", feature_state_hash=f"synthetic-fs-{w:02d}-{j:02d}",
        source_snapshots={"synthetic": "demo"}, prediction_created_at=ts(created), prediction_cutoff=None,
        feature_availability={"a_service_points_observed": sp[0], "b_service_points_observed": sp[1],
                              "global_match_count_a": em[0], "global_match_count_b": em[1], "filled_inputs": []},
        freshness_status="ok", inference_engine_version="synthetic-demo", git_commit=None, **o,
    )  # fmt: skip


def build_demo(cfg: EvaluationConfig, root: Path, weeks: int = 30, per_week: int = 14, seed: int = 20261102,
               replay_matches: int = 40) -> dict:  # fmt: skip
    root = Path(root)
    if (root / "ledger.sqlite").exists():
        raise FileExistsError(root / "ledger.sqlite")
    rng = np.random.default_rng(seed)
    lcfg = replace(cfg.ledger, ledger_path=root / "ledger.sqlite")
    clock = FixedClock(T_START - timedelta(days=3), timedelta(0))
    w = LedgerWriter(lcfg, clock)
    w.init("ledger-synthetic-demo")
    plan, outcome_lines, truth = [], [], {}
    # ------------------------------------------------------------------ plan every event with its commit time
    for wk in range(weeks):
        level = LEVELS[wk % len(LEVELS)]
        for j in range(per_week):
            mid = f"synthetic:demo-w{wk:02d}-m{j:02d}"
            start = T_START + timedelta(weeks=wk, days=j % 6, hours=11 + (j % 5))
            p1, p2 = 100000 + 100 * wk + 2 * j, 100001 + 100 * wk + 2 * j
            z = float(rng.normal(0, 1.1))
            p_true = _sig(z)
            m4a, elo = _sig(z + float(rng.normal(0, 0.35))), _sig(0.8 * z + float(rng.normal(0, 0.5)))
            u = float(rng.random())
            r = float(rng.random())
            win1 = u < p_true
            winner, loser = (p1, p2) if win1 else (p2, p1)
            case = ("pending" if wk == weeks - 1 else "cancelled" if r < 0.01 else "players" if r < 0.015 else
                    "postponed" if r < 0.035 else "late" if r < 0.055 else "no_start" if r < 0.09 else
                    "retired" if r < 0.12 else "walkover" if r < 0.13 else "default" if r < 0.14 else
                    "abandoned" if r < 0.15 else "disputed" if r < 0.165 else "corrected" if r < 0.18 else
                    "duplicate" if r < 0.21 else "completed")  # fmt: skip
            truth[mid] = {"case": case, "p1": p1, "p2": p2, "winner": winner}
            created = start - timedelta(hours=20)
            commit = created + timedelta(minutes=1)
            if case == "late":
                created, commit = start + timedelta(minutes=20), start + timedelta(minutes=21)
            req = _request(
                cfg, mid, p1, p2, m4a, elo, start, "pre_match", f"sched-{mid}-v1", created, level, wk, j, rng
            )
            plan.append((commit, "pred", mid, req))
            if j % 7 == 3 and case not in ("cancelled", "players"):  # another slot: never the primary selection
                plan.append((start - timedelta(hours=1), "pred", mid,
                             _request(cfg, mid, p1, p2, min(0.99, m4a + 0.02), elo, start, "t_minus_1h",
                                      f"sched-{mid}-v1", start - timedelta(hours=1, minutes=1), level, wk, j, rng)))  # fmt: skip
            if case == "duplicate":  # schedule re-issued: a second pre_match forecast; the first stays selected
                plan.append((start - timedelta(hours=10), "pred", mid,
                             _request(cfg, mid, p1, p2, 0.5 + 0.9 * (m4a - 0.5), elo, start, "pre_match",
                                      f"sched-{mid}-v2", start - timedelta(hours=10, minutes=1), level, wk, j, rng)))  # fmt: skip
            src = {"provider": "synthetic_feed_a", "snapshot_id": f"feed-a-w{wk:02d}"}
            if case == "pending":
                continue
            if case == "cancelled":
                plan.append((start - timedelta(hours=5), "cancel", mid, {"reason": "synthetic withdrawal",
                                                                          "source": src}))  # fmt: skip
                continue
            if case == "players":
                plan.append((start - timedelta(hours=5), "players", mid, {
                    "kind": "players", "changes": {"superseded_by_match_id": mid + "-ll", "player_a_id": p1,
                                                   "player_b_id": p2 + 50}, "reason": "lucky loser", "source": src}))  # fmt: skip
                continue
            actual = start + timedelta(minutes=7)
            if case == "postponed":
                actual = start + timedelta(days=1, minutes=7)
                plan.append((start - timedelta(hours=2), "postpone", mid, {
                    "new_scheduled_start_at": ts(start + timedelta(days=1)), "reason": "rain", "source": src}))  # fmt: skip
            if case != "no_start":
                plan.append((actual + timedelta(minutes=3), "start", mid, {
                    "kind": "actual_start", "changes": {"actual_start_at": ts(actual)}, "reason": "match began",
                    "source": src}))  # fmt: skip
            status, score, win = "completed", "6-4 6-4" if level != "G" else "6-4 3-6 6-4 6-4", winner
            if case == "retired":
                status, score = "retired", "6-4 3-1 RET"
            elif case == "walkover":
                status, score = "walkover", "W/O"
            elif case == "default":
                status, score = "default", "6-4 2-1 DEF"
            elif case == "abandoned":
                status, score, win = "abandoned", "6-4 2-2 ABD", None
            res_t = actual + timedelta(hours=3)
            if case == "corrected":
                plan.append((res_t, "result_wrong", mid, {"status": "completed", "winner_id": loser, "score": score,
                                                          "source": src, "supersedes_event_id": None,
                                                          "correction_reason": None}))  # fmt: skip
                plan.append((res_t + timedelta(hours=2), "result_fix", mid, {
                    "status": "completed", "winner_id": winner, "score": score, "source": src,
                    "correction_reason": "winner entered in reverse"}))  # fmt: skip
            else:
                plan.append((res_t, "result", mid, {"status": status, "winner_id": win, "score": score,
                                                     "source": src, "supersedes_event_id": None,
                                                     "correction_reason": None}))  # fmt: skip
            retrieved = ts(res_t + timedelta(hours=6))
            base = {"match_id": mid, "provider_record_id": f"{mid}#1", "source_version": 1, "status": status,
                    "winner_id": win, "player_ids": [p1, p2], "score": score,
                    "actual_start_at": None if case == "no_start" else ts(actual), "tournament_id": f"synthetic-t{wk:02d}",
                    "tournament_level": level, "retrieved_at": retrieved}  # fmt: skip
            outcome_lines.append({**base, "provider": "synthetic_feed_a"})
            if case == "disputed" or j % 3 == 0:
                other = {**base, "provider": "synthetic_feed_b", "provider_record_id": f"b-{mid}"}
                if case == "disputed" and win is not None:
                    other["winner_id"] = loser
                outcome_lines.append(other)
    for k in range(replay_matches):  # historical reconstructions: never part of a SYNTHETIC or PROSPECTIVE cohort
        mid = f"replay:demo-{k:03d}"
        start = T_START + timedelta(days=2 * k, hours=9)
        z = float(rng.normal(0, 1))
        req = _request(cfg, mid, 300000 + 2 * k, 300001 + 2 * k, _sig(z), _sig(0.8 * z), start, "pre_match",
                       f"sched-{mid}", start - timedelta(hours=30), "A", k % weeks, k, rng, mode="HISTORICAL_REPLAY")  # fmt: skip
        plan.append((start - timedelta(hours=29), "pred", mid, req))
    plan.sort(key=lambda x: (x[0], x[1], x[2]))
    ids: dict = {}
    mid_cp = None
    half = len(plan) // 2
    last = clock.t

    def tick(t):  # the ledger refuses clock regressions; planned events at equal times keep their order
        nonlocal last
        last = max(last, t)
        clock.t = last

    for i, (t, kind, mid, payload) in enumerate(plan):
        tick(t)
        if kind == "pred":
            rcp = w.commit_prediction(payload)
            ids.setdefault(("pred", mid), []).append(rcp.event_id)
        elif kind == "cancel":
            ids[("cancel", mid)] = w.record("MATCH_CANCELLED", mid, payload).event_id
            for pid in ids.get(("pred", mid), []):
                tick(t + timedelta(seconds=1))
                w.record("PREDICTION_VOIDED", mid, {"reason_code": "match_cancelled",
                                                    "evidence_event_ids": [ids[("cancel", mid)]],
                                                    "actor": "synthetic-demo-operator", "note": "objective rule"},
                         ref_event_id=pid)  # fmt: skip
        elif kind == "players":
            ev = w.record("MATCH_METADATA_CORRECTED", mid, payload).event_id
            tick(t + timedelta(seconds=1))
            w.record("PREDICTION_VOIDED", mid, {"reason_code": "players_changed", "evidence_event_ids": [ev],
                                                "actor": "synthetic-demo-operator", "note": "objective rule"},
                     ref_event_id=ids[("pred", mid)][0])  # fmt: skip
        elif kind == "postpone":
            w.record("MATCH_POSTPONED", mid, payload)
        elif kind == "start":
            w.record("MATCH_METADATA_CORRECTED", mid, payload)
        elif kind in ("result", "result_wrong"):
            ids[("result", mid)] = w.record("MATCH_RESULT_RECORDED", mid, payload).event_id
        elif kind == "result_fix":
            w.record("MATCH_RESULT_RECORDED", mid, {**payload, "supersedes_event_id": ids[("result", mid)]})
        if i == half:
            mid_cp = make_checkpoint(w.con)
    final_cp = make_checkpoint(w.con, mid_cp)
    w.close()
    # outcome snapshot: a corrected record (feed A, version 2) is issued for corrected matches; v1 had the error
    for line in list(outcome_lines):
        if truth[line["match_id"]]["case"] == "corrected" and line["provider"] == "synthetic_feed_a":
            outcome_lines.append({**line, "source_version": 2})
    for line in outcome_lines:
        if (truth[line["match_id"]]["case"] == "corrected" and line["source_version"] == 1
                and line["provider"] == "synthetic_feed_a"):  # fmt: skip
            line["winner_id"] = [x for x in (truth[line["match_id"]]["p1"], truth[line["match_id"]]["p2"])
                                 if x != truth[line["match_id"]]["winner"]][0]  # fmt: skip
    header = {"format": "tennis-outcome-snapshot-1", "snapshot_id": "synthetic-demo-outcomes",
              "created_at": ts(T_START + timedelta(weeks=weeks)), "description": "SYNTHETIC outcomes for the demo"}  # fmt: skip
    body = b"\n".join(canonical(x) for x in [header, *sorted(outcome_lines, key=lambda d: (
        d["match_id"], d["provider"], d["source_version"]))]) + b"\n"  # fmt: skip
    write_new(root / "outcomes.jsonl", body)
    # anchors: local fixture publications only (never independent evidence) and a retained checkpoint
    fix = anchoring.LocalFixturePublisher(root / "fixture_publications")
    for k, cp in enumerate((mid_cp, final_cp)):
        anchoring.publish(cp, fix, root / "anchors", "synthetic-demo", now=T_START + timedelta(weeks=weeks, hours=k))
    write_checkpoint(mid_cp, root / "trusted" / f"checkpoint-{mid_cp['sequence']:08d}.json",
                     exported_at=T_START + timedelta(weeks=weeks))  # fmt: skip
    cases: dict = {}
    for v in truth.values():
        cases[v["case"]] = cases.get(v["case"], 0) + 1
    return {"ledger": root / "ledger.sqlite", "outcomes": root / "outcomes.jsonl", "evidence": root / "anchors",
            "fixture_root": root / "fixture_publications", "trusted": root / "trusted", "cases": cases,
            "checkpoints": [mid_cp, final_cp]}  # fmt: skip

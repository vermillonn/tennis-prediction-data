"""Phase 4C: outcome reconciliation (statuses, corrections, conflicts) and paired metrics with uncertainty."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from evaluation_helpers import T0, World, outcome_record, scored_match, write_outcomes
from tennis_data.evaluation_prospective.calibration import calibration_tables, subgroup_tables
from tennis_data.evaluation_prospective.config import load_evaluation_config, protocol_from
from tennis_data.evaluation_prospective.metrics import score
from tennis_data.evaluation_prospective.outcomes import load_outcomes, reconcile_match
from tennis_data.evaluation_prospective.pipeline import evaluate
from tennis_data.evaluation_prospective.safeio import UntrustedInputError
from tennis_data.evaluation_prospective.snapshot import snapshot_from_ledger
from tennis_data.evaluation_prospective.uncertainty import paired_uncertainty
from tennis_data.ledger.lifecycle import MatchState
from tennis_data.modeling.evaluation import metrics as shared_metrics

P1, P2 = 104925, 207989
PAIR = frozenset((P1, P2))


def _st(result=None, status="scheduled", history=()):
    st = MatchState("m", status=status)
    if result:
        st.result, st.result_event, st.result_history = result, "rslt-1", list(history) or ["rslt-1"]
        st.status = "decided"
    return st


def _res(status, winner, score="6-4 6-4", provider="feed_a"):
    return {"status": status, "winner_id": winner, "score": score, "source": {"provider": provider},
            "supersedes_event_id": None, "correction_reason": None}  # fmt: skip


@pytest.mark.parametrize("status,winner,score,expected", [
    ("completed", P1, "6-4 6-4", "COMPLETED"), ("retired", P1, "6-4 2-1 RET", "RETIRED"),
    ("walkover", P1, "W/O", "WALKOVER"), ("default", P2, "6-4 2-1 DEF", "DEFAULT"),
    ("abandoned", None, "6-4 2-2 ABD", "ABANDONED"), ("unknown", None, None, "UNKNOWN"),
])  # fmt: skip
def test_outcome_statuses_are_kept_apart(status, winner, score, expected):
    o = reconcile_match(_st(_res(status, winner, score)), [], [], PAIR)
    assert o["outcome"] == expected and o["winner_id"] == winner


def test_pending_cancelled_and_identity():
    assert reconcile_match(_st(), [], [], PAIR)["outcome"] == "PENDING"
    assert reconcile_match(MatchState("m", status="cancelled"), [], [], PAIR)["outcome"] == "CANCELLED"
    rec = outcome_record("m", "feed_b", "completed", P1, (P1, P2))
    assert reconcile_match(MatchState("m", status="cancelled"), [], [rec], PAIR)["outcome"] == "DISPUTED"
    wrong_pair = outcome_record("m", "feed_b", "completed", P1, (P1, 5))
    assert reconcile_match(_st(), [], [wrong_pair], PAIR)["outcome"] == "IDENTITY_MISMATCH"
    assert reconcile_match(_st(_res("completed", 77)), [], [], PAIR)["outcome"] == "IDENTITY_MISMATCH"


def test_conflicting_sources_are_quarantined_not_resolved_by_preference():
    led = _st(_res("completed", P1))
    b = outcome_record("m", "feed_b", "completed", P2, (P1, P2))
    o = reconcile_match(led, [], [b], PAIR)
    assert o["outcome"] == "DISPUTED" and "sources_disagree" in o["flags"] and o["winner_id"] is None
    # a newer version from the disagreeing provider resolves it
    b2 = {**b, "source_version": 2, "winner_id": P1}
    assert reconcile_match(led, [], [b, b2], PAIR)["outcome"] == "COMPLETED"
    # two different contents under one version: reissued -> disputed
    b3 = {**b, "score": "7-5 6-4"}
    o = reconcile_match(led, [], [{**b, "winner_id": P1}, {**b3, "winner_id": P1}], PAIR)
    assert o["outcome"] == "DISPUTED" and any(f.startswith("version_reissued") for f in o["flags"])
    # a score whose own marker contradicts the stated status
    rr = outcome_record("m", "feed_b", "completed", P1, (P1, P2), score="6-4 2-1 RET")
    assert reconcile_match(led, [], [rr], PAIR)["outcome"] == "DISPUTED"


def test_result_correction_preserves_history(tmp_path):
    wd = World(tmp_path)
    s = T0 + timedelta(days=1)
    wd.predict("m", s, s - timedelta(hours=5))
    wd.start(s + timedelta(minutes=10), "m", s + timedelta(minutes=5))
    first = wd.result(s + timedelta(hours=3), "m", P2)
    wd.result(s + timedelta(hours=5), "m", P1, supersedes=first.event_id, reason="reversed")
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    oc = load_outcomes(write_outcomes(tmp_path / "o.jsonl", [outcome_record("m", "feed_a", "completed", P1, (P1, P2))]))
    res = evaluate(snap, oc, wd.cfg.protocol)
    o = res["outcomes"][0]
    assert o["outcome"] == "COMPLETED" and o["winner_id"] == P1 and o["corrected"] and len(o["result_history"]) == 2
    r = res["classification"][0]
    assert r["category"] == "SCORABLE" and r["outcome_corrected"] and r["y"] == (1 if r["player_a_id"] == P1 else 0)


def test_outcome_snapshot_validation(tmp_path):
    bad = outcome_record("m", "f", "won", P1, (P1, P2))
    with pytest.raises(UntrustedInputError, match="status"):
        load_outcomes(write_outcomes(tmp_path / "a.jsonl", [bad]))
    extra = {**outcome_record("m", "f", "completed", P1, (P1, P2)), "exec": "rm -rf /"}
    with pytest.raises(UntrustedInputError, match="fields"):
        load_outcomes(write_outcomes(tmp_path / "b.jsonl", [extra]))
    (tmp_path / "c.jsonl").write_text('{"format":"x"}\n')
    with pytest.raises(UntrustedInputError):
        load_outcomes(tmp_path / "c.jsonl")


# ------------------------------------------------------------------------------------------------------ metrics
def _protocol(**over):
    cfg = load_evaluation_config()
    return {**cfg.protocol, **over}


def _rows(ps, es, ys, start="2030-03-05T10:00:00.000000Z", tid="t"):
    return [{"match_id": f"m{i:04d}", "seq": i + 1, "probability_a": p, "baseline_elo_probability_a": e, "y": y,
             "actual_start_at": start, "tournament_id": tid, "surface": "Hard", "tournament_level": "A", "best_of": 3,
             "prior_service_points": 2000, "elo_matches": 50, "freshness_status": "ok", "scheduled_start_at": start}
            for i, (p, e, y) in enumerate(zip(ps, es, ys, strict=True))]  # fmt: skip


def test_point_metrics_match_hand_calculation_and_shared_scorer():
    ps, es, ys = [0.8, 0.3, 0.5, 0.9], [0.6, 0.4, 0.5, 0.7], [1, 0, 1, 0]
    s = score(_rows(ps, es, ys), _protocol())
    ll = -sum(math.log(p) if y else math.log(1 - p) for p, y in zip(ps, ys, strict=True)) / 4
    br = sum((p - y) ** 2 for p, y in zip(ps, ys, strict=True)) / 4
    m = s["models"]["M4A"]
    assert m["log_loss"] == pytest.approx(ll, abs=1e-15) and m["brier"] == pytest.approx(br, abs=1e-15)
    assert m["accuracy"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)  # p = 0.5 gets half credit (shared convention)
    df = pl.DataFrame({"p": ps, "target_a_wins": [bool(y) for y in ys]})
    assert m["log_loss"] == shared_metrics(df, "p")["log_loss"][0]
    d = [x["d_ll"] for x in s["matches"]]
    assert s["paired"]["log_loss_diff"] == pytest.approx(
        s["models"]["M4A"]["log_loss"] - s["models"]["Elo"]["log_loss"]
    )
    assert math.fsum(d) / 4 == pytest.approx(s["paired"]["log_loss_diff"], abs=1e-15)
    with pytest.raises(ValueError, match="clip"):
        score(_rows(ps, es, ys), _protocol(probability_clip=1e-6))


def test_same_clipping_for_both_models_and_extremes():
    s = score(_rows([1.0, 0.0], [1.0, 0.0], [0, 1]), _protocol())
    assert math.isfinite(s["models"]["M4A"]["log_loss"])
    assert s["models"]["M4A"]["log_loss"] == s["models"]["Elo"]["log_loss"] and s["paired"]["log_loss_diff"] == 0


def test_calibration_bins():
    rows = _rows([0.05, 0.15, 0.15, 0.95], [0.5] * 4, [0, 0, 1, 1])
    t = calibration_tables(rows, 10)["M4A"]
    by = {r["bin"]: r for r in t}
    assert by[1]["n"] == 2 and by[1]["observed"] == 0.5 and by[1]["mean_predicted"] == pytest.approx(0.15)
    assert by[9]["observed_minus_predicted"] == pytest.approx(0.05)
    assert sum(r["n"] for r in t) == 4


def test_bootstrap_interval_is_deterministic_and_gated_by_sample_size():
    rng = np.random.default_rng(1)
    n = 400
    p_true = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p_true).astype(int).tolist()
    starts = [f"2030-{3 + i // 120:02d}-{1 + (i % 28):02d}T10:00:00.000000Z" for i in range(n)]
    rows = _rows(p_true.tolist(), np.clip(p_true + rng.normal(0, 0.1, n), 0.01, 0.99).tolist(), y)
    for r, st in zip(rows, starts, strict=True):
        r["actual_start_at"] = st
        r["tournament_id"] = f"t{st[5:10]}"
    prot = _protocol(bootstrap_samples=500)
    u1, u2 = paired_uncertainty(rows, prot), paired_uncertainty(rows, prot)
    assert u1 == u2 and u1["status"] == "inferential"
    w = u1["intervals"]["week"]
    assert w["status"] == "reported" and w["ll_ci95"][0] < w["log_loss_diff"] < w["ll_ci95"][1] and w["blocks"] >= 10
    small = paired_uncertainty(rows[:40], prot)
    assert small["status"].startswith("insufficient") and "ll_ci95" not in small["intervals"]["week"]
    assert "insufficient evidence" in small["intervals"]["week"]["status"]
    assert all(s["status"] == "descriptive only" for s in small["seasons"])
    assert paired_uncertainty([], prot)["status"] == "no scorable matches"


def test_small_subgroups_are_counts_only():
    rows = _rows([0.6] * 60 + [0.7] * 5, [0.55] * 65, [1] * 40 + [0] * 25)
    for r in rows[60:]:
        r["surface"] = "Clay"
    t = {x["group"]: x for x in subgroup_tables(rows, _protocol())["surface"]}
    assert t["Hard"]["status"] == "reported" and "ll_m4a" in t["Hard"]
    assert t["Clay"]["status"].startswith("inconclusive") and "ll_m4a" not in t["Clay"] and t["Clay"]["n"] == 5


def test_protocol_guards():
    from tennis_data.config import ConfigError

    cfg = load_evaluation_config()
    with pytest.raises(ConfigError):
        protocol_from({"scored_outcomes": ["COMPLETED", "WALKOVER"]}, {}, cfg.ledger)
    with pytest.raises(ConfigError):
        protocol_from({}, {"min_matches_inferential": 5}, cfg.ledger)
    with pytest.raises(ConfigError):
        protocol_from({"cohort_mode": "LIVE"}, {}, cfg.ledger)


def test_retirements_are_secondary_only(tmp_path):
    wd = World(tmp_path)
    scored_match(wd, "done", T0 + timedelta(days=1))
    s = T0 + timedelta(days=2)
    wd.predict("ret", s, s - timedelta(hours=5))
    wd.start(s + timedelta(minutes=10), "ret", s + timedelta(minutes=5))
    wd.result(s + timedelta(hours=2), "ret", P1, status="retired", score="6-4 2-1 RET")
    wd.close()
    snap = snapshot_from_ledger(wd.cfg.ledger.ledger_path, wd.cfg.snapshot_dir)
    from tennis_data.evaluation_prospective.outcomes import empty_outcomes

    res = evaluate(snap, empty_outcomes(), wd.cfg.protocol)
    assert res["scores"]["operational"]["n"] == 1 and res["scores"]["retirement_secondary"]["n"] == 1
    res = evaluate(snap, empty_outcomes(), {**wd.cfg.protocol, "retirement_policy": "exclude"})
    assert res["scores"]["retirement_secondary"]["n"] == 0

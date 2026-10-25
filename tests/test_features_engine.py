"""Aggregation, windows, decay and priors of the snapshot engine (known answers)."""

from __future__ import annotations

import pytest

from feature_helpers import cfg, match, row, run


def test_service_and_return_points_from_both_sides():
    t = run([match("m1", 0, 100, 200, a_sp=(60, 40), b_sp=(50, 30)), match("m2", 1, 100, 300)])
    r = row(t, "m2")
    assert (r["a_service_points_observed"], r["a_service_points_won"]) == (60, 40)
    # return points = opponent's service points; won = opponent svpt - opponent service points won
    assert (r["a_return_points_observed"], r["a_return_points_won"]) == (50, 20)
    assert r["a_spw_raw"] == pytest.approx(40 / 60) and r["a_rpw_raw"] == pytest.approx(20 / 50)


def test_rates_are_point_weighted_not_match_averaged():
    rows = [match("m1", 0, 100, 200, a_sp=(10, 9)), match("m2", 1, 100, 300, a_sp=(100, 50)), match("m3", 2, 100, 400)]
    r = row(run(rows), "m3")
    assert r["a_spw_raw"] == pytest.approx(59 / 110)
    assert r["a_spw_raw"] != pytest.approx((0.9 + 0.5) / 2)


def test_missing_statistics_are_preserved_as_null_not_zero():
    rows = [match("m1", 0, 100, 200, eligible=False), match("m2", 1, 100, 300)]
    r = row(run(rows, cfg(min_prior_points=10**9)), "m2")
    assert r["a_has_stats_history"] is False and r["a_stat_matches"] == 0
    assert r["a_service_points_observed"] == 0
    assert r["a_spw_raw"] is None and r["a_rpw_raw"] is None  # zero denominator -> null, not 0%
    assert r["a_spw_last5"] is None and r["a_spw_decay52w"] is None
    assert r["a_spw_shrunk"] is None  # no prior released yet -> null, not a fabricated value


def test_prior_only_estimate_for_player_without_history():
    rows = [match("m1", 0, 100, 200, a_sp=(60, 36), b_sp=(40, 24)), match("m2", 1, 300, 400)]
    r = row(run(rows), "m2")
    assert r["a_prior_spw"] == pytest.approx(60 / 100)  # population: (36 + 24) / (60 + 40)
    assert r["a_spw_shrunk"] == pytest.approx(0.6) and r["a_spw_prior_weight"] == 1.0
    assert r["a_service_prior_dominated"] is True
    assert r["a_rpw_shrunk"] == pytest.approx(0.4)


def test_population_prior_needs_minimum_released_points():
    rows = [match("m1", 0, 100, 200), match("m2", 1, 100, 300), match("m3", 2, 100, 400)]
    t = run(rows, cfg(min_prior_points=150))
    assert row(t, "m2")["a_prior_spw"] is None  # 110 released points < 150
    assert row(t, "m3")["a_prior_spw"] == pytest.approx(140 / 220)


def test_shrinkage_formula_and_convergence():
    rows = [match(f"m{i}", i, 100, 200 + i, a_sp=(100, 70), b_sp=(100, 60)) for i in range(30)] + [
        match("z", 30, 100, 999)
    ]
    r = row(run(rows, cfg(kappa_service=200.0)), "z")
    w, n, mu = 70 * 30, 100 * 30, (70 + 60) / 200
    assert r["a_spw_shrunk"] == pytest.approx((w + 200 * mu) / (n + 200))
    assert abs(r["a_spw_shrunk"] - 0.7) < abs(r["a_spw_shrunk"] - mu)  # data dominate at n >> kappa


def test_surface_counts_are_tracked_separately():
    rows = [
        match("h", 0, 100, 200, a_sp=(60, 45), surface="Hard"),
        match("c", 1, 100, 300, a_sp=(80, 40), surface="Clay"),
        match("q", 2, 100, 400, surface="Clay"),
        match("g", 2, 500, 100, surface="Grass"),
    ]
    t = run(rows)
    q = row(t, "q")
    assert q["a_surface_service_points"] == 80 and q["a_surface_spw_raw"] == pytest.approx(0.5)
    assert q["a_service_points_observed"] == 140
    g = row(t, "g")
    assert g["b_surface_service_points"] == 0 and g["b_surface_spw_raw"] is None
    assert g["b_surface_prior_weight"] == 1.0
    assert g["b_surface_spw"] is None  # no grass points released anywhere -> no grass prior


def test_missing_surface_gives_null_surface_features():
    t = run([match("m1", 0, 100, 200), match("m2", 1, 100, 300, surface=None)])
    r = row(t, "m2")
    assert r["a_surface_service_points"] is None and r["a_surface_spw"] is None
    assert r["a_service_points_observed"] == 60


def test_last_n_match_window():
    rows = [match(f"m{i}", i, 100, 200 + i, a_sp=(10 * (i + 1), i + 1)) for i in range(7)] + [match("z", 7, 100, 999)]
    r = row(run(rows), "z")
    last5 = range(2, 7)
    assert r["a_service_points_last5"] == sum(10 * (i + 1) for i in last5)
    assert r["a_spw_last5"] == pytest.approx(sum(i + 1 for i in last5) / sum(10 * (i + 1) for i in last5))
    assert r["a_service_points_last20"] == sum(10 * (i + 1) for i in range(7))


def test_week_window_excludes_old_observations():
    rows = [
        match("old", 0, 100, 200, a_sp=(50, 50)),
        match("new", 20, 100, 300, a_sp=(50, 25)),
        match("z", 21, 100, 400),
    ]
    r = row(run(rows), "z")
    assert r["a_service_points_last13w"] == 50 and r["a_spw_last13w"] == pytest.approx(0.5)
    assert r["a_service_points_last52w"] == 100
    t = run([match("old", 0, 100, 200), match("z", 30, 100, 400)])
    assert row(t, "z")["a_spw_last13w"] is None and row(t, "z")["a_service_points_last13w"] == 0


def test_recency_weights_are_exact():
    rows = [
        match("a", 0, 100, 200, a_sp=(100, 80)),
        match("b", 52, 100, 300, a_sp=(100, 40)),
        match("z", 104, 100, 400),
    ]
    r = row(run(rows), "z")
    w_a, w_b = 2 ** (-104 / 52), 2 ** (-52 / 52)
    assert r["a_service_points_decay52w"] == pytest.approx(100 * w_a + 100 * w_b)
    assert r["a_spw_decay52w"] == pytest.approx((80 * w_a + 40 * w_b) / (100 * w_a + 100 * w_b))
    assert r["a_spw_raw"] == pytest.approx(0.6)  # unweighted kept separately


def test_weeks_since_last_stats():
    r = row(run([match("a", 0, 100, 200), match("z", 9, 100, 300)]), "z")
    assert r["a_weeks_since_last_stats"] == 9


def test_diff_features():
    rows = [match("a", 0, 100, 200, a_sp=(100, 70), b_sp=(100, 60)), match("z", 1, 100, 200)]
    r = row(run(rows), "z")
    assert r["spw_diff"] == pytest.approx(r["a_spw_shrunk"] - r["b_spw_shrunk"])
    assert r["rpw_diff"] == pytest.approx(r["a_rpw_shrunk"] - r["b_rpw_shrunk"])

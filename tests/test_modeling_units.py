"""Phase 3B unit tests: logistic fit, calibration, preprocessing, chronology, metrics."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import optimize

from tennis_data.modeling import folds
from tennis_data.modeling.calibration import fit_isotonic, fit_temperature, logit, pava
from tennis_data.modeling.config import load_modeling_config
from tennis_data.modeling.evaluation import paired_bootstrap, row_losses, verdict
from tennis_data.modeling.experiments import choose_with_tolerance
from tennis_data.modeling.logistic import fit_logistic, objective, predict, sigmoid
from tennis_data.modeling.preprocessing import fit_scaler


def _data(n=4000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k)) * np.array([1.0, 0.1, 3.0, 0.5])[:k]
    beta = np.array([0.8, -4.0, 0.2, 0.0])[:k]
    y = (rng.random(n) < sigmoid(x @ beta)).astype(float)
    return x, y


@pytest.mark.parametrize("c", [0.01, 0.1, 1.0, 10.0])
def test_logistic_matches_generic_optimizer(c):
    x, y = _data()
    fit = fit_logistic(x, y, c, tol=1e-12)
    ref = optimize.minimize(objective, np.zeros(x.shape[1]), args=(x, y, c), method="L-BFGS-B",
                            options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000})  # fmt: skip
    assert fit.converged
    assert fit.objective <= ref.fun + 1e-8
    np.testing.assert_allclose(fit.coef, ref.x, atol=1e-5)


def test_logistic_penalty_shrinks_and_has_no_intercept():
    x, y = _data()
    strong, weak = fit_logistic(x, y, 1e-4), fit_logistic(x, y, 100.0)
    assert np.linalg.norm(strong.coef) < np.linalg.norm(weak.coef)
    assert predict(weak.coef, np.zeros((1, x.shape[1])))[0] == 0.5  # no intercept: equal inputs -> coin flip


def test_player_order_symmetry_is_exact():
    x, y = _data()
    fit = fit_logistic(x, y, 1.0)
    p, q = predict(fit.coef, x), predict(fit.coef, -x)
    assert np.max(np.abs(p + q - 1)) < 1e-15
    # refitting on the swapped dataset gives the same coefficients
    swapped = fit_logistic(-x, 1 - y, 1.0)
    np.testing.assert_allclose(swapped.coef, fit.coef, atol=1e-10)


def test_logistic_is_deterministic():
    x, y = _data()
    assert np.array_equal(fit_logistic(x, y, 0.1).coef, fit_logistic(x, y, 0.1).coef)


def test_scaler_uses_rms_without_centring():
    x = np.array([[1.0, 0.0], [-3.0, 0.0], [2.0, 0.0]])
    sc = fit_scaler(x, ["a", "b"])
    assert sc.scale[0] == pytest.approx(np.sqrt(14 / 3))
    assert sc.scale[1] == 1.0  # constant zero column: scale 1, not a division by zero
    np.testing.assert_array_equal(sc.transform(-x), -sc.transform(x))  # antisymmetry preserved
    with pytest.raises(ValueError):
        fit_scaler(np.array([[np.nan, 1.0]]), ["a", "b"])


def test_temperature_matches_one_dimensional_optimum():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 3000)
    y = (rng.random(3000) < sigmoid(0.7 * logit(p, 1e-6))).astype(float)
    cal = fit_temperature(p, y, 1e-6)
    z = logit(p, 1e-6)

    def nll(b):
        return float(np.sum(np.logaddexp(0, b * z) - y * b * z))

    ref = optimize.minimize_scalar(nll, bounds=(0.01, 5), method="bounded", options={"xatol": 1e-10})
    assert cal.slope == pytest.approx(ref.x, abs=1e-6)
    q = np.linspace(0.01, 0.99, 99)
    np.testing.assert_allclose(cal.apply(q) + cal.apply(1 - q), 1, atol=1e-14)


def test_pava_against_brute_force_reference():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 15, 200).astype(float)
    y = rng.random(200)
    ux, fy = pava(x, y)
    assert np.all(np.diff(fy) >= -1e-15)
    # reference: the isotonic fit minimises squared error; compare with a fine projected solver
    w = np.array([np.sum(x == u) for u in ux])
    m = np.array([y[x == u].mean() for u in ux])
    ref = optimize.minimize(lambda v: np.sum(w * (v - m) ** 2), m, method="SLSQP",
                            constraints=[{"type": "ineq", "fun": lambda v: np.diff(v)}], options={"ftol": 1e-14})  # fmt: skip
    np.testing.assert_allclose(fy, ref.x, atol=1e-5)


def test_isotonic_is_symmetric_and_monotone():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < p**1.5).astype(float)
    cal = fit_isotonic(p, y, 1e-6)
    q = np.linspace(0, 1, 1001)
    g = cal.apply(q)
    assert np.all(np.diff(g) >= -1e-12)
    np.testing.assert_allclose(g + cal.apply(1 - q), 1, atol=1e-12)
    assert cal.apply(np.array([0.5]))[0] == pytest.approx(0.5)


def test_bootstrap_point_estimate_ci_and_determinism():
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.random(n) < 0.5).astype(float)
    base = np.full(n, 0.5)
    right = rng.random(n) < 0.7  # informative but noisy: the per-row loss difference varies
    better = np.where((y == 1) == right, 0.6, 0.4)
    cl = np.repeat(np.arange(100), 30)
    b = paired_bootstrap(better, base, y, cl, 500, 7)
    ll_new, _ = row_losses(better, y)
    ll_b, _ = row_losses(base, y)
    assert b["log_loss_diff"] == pytest.approx(ll_new.mean() - ll_b.mean(), abs=1e-14)
    assert b["clusters"] == 100 and b["rows"] == n
    lo, hi = b["ll_ci95"]
    assert lo <= b["log_loss_diff"] <= hi < 0
    assert verdict(lo, hi) == "candidate better"
    assert paired_bootstrap(better, base, y, cl, 500, 7) == b
    same = paired_bootstrap(base, base, y, cl, 200, 1)
    assert same["log_loss_diff"] == 0 and same["ll_ci95"] == (0.0, 0.0)
    assert verdict(-0.1, 0.1).startswith("inconclusive")


def test_simplicity_rule():
    scores = {"big": 0.5900, "small": 0.5904, "tiny": 0.6}
    assert choose_with_tolerance(scores, ["tiny", "small", "big"], 0.0005)[0] == "small"
    assert choose_with_tolerance(scores, ["tiny", "small", "big"], 0.0)[0] == "big"


def _chron_frame():
    s0 = date(2001, 12, 22)  # a Saturday
    rows = []
    for w in range(6):
        b = s0 + timedelta(weeks=w)
        for lag in (0, 1):  # lag 1 = two-week event: label released one batch later
            rows.append({"match_id": f"{w}-{lag}", "prediction_batch": b, "release_batch": b + timedelta(weeks=lag)})
    return pl.DataFrame(rows)


def test_training_mask_uses_release_batch_not_tournament_start():
    cfg = load_modeling_config()
    df = _chron_frame()
    cut = folds.cutoffs(df)
    assert cut[2002] == date(2002, 1, 5)
    tm = df.filter(folds.train_mask(df, cfg, cut[2002]))
    # the two-week event predicted on 2001-12-29 is released on 2002-01-05 = cutoff: not yet known
    assert set(tm["match_id"]) == {"0-0", "0-1", "1-0"}
    assert (tm["release_batch"] < cut[2002]).all()


def test_calibration_rows_exclude_unreleased_labels():
    cfg = load_modeling_config()
    df = _chron_frame()
    m = folds.calibration_mask(df, cfg, 2002, date(2002, 1, 5))
    assert set(df.filter(m)["match_id"]) == {"0-0", "0-1", "1-0"}


def test_config_validation(tmp_path):
    from tennis_data.config import ConfigError

    text = (load_modeling_config().config_path).read_text()
    bad = tmp_path / "m.toml"
    bad.write_text(text.replace("folds = [[2002, 2004], [2005, 2007], [2008, 2009]]", "folds = [[2002, 2004]]"))
    with pytest.raises(ConfigError):
        load_modeling_config(bad)
    bad.write_text(text.replace('id = "M0R"', 'id = "M0"'))
    with pytest.raises(ConfigError):
        load_modeling_config(bad)

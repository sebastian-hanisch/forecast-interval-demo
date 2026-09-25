"""Vehikel, Punktmodelle mit Anpassungsende, Auswertung und Aufschlüsselungen; die Auswertung darf nichts aus der Zukunft des Ursprungs verwenden."""

import dataclasses

import numpy as np
import pytest

import fi_baselines as B
import fi_constants as C
import fi_evaluation as E
import fi_intervals as I
import fi_scenario as S


def test_scenario_is_deterministic_and_seeded():
    a, b, c = S.generate(5, seed=1), S.generate(5, seed=1), S.generate(5, seed=2)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (5, C.N_DAYS) and a.sigma.shape == a.y.shape


def test_shift_changes_only_the_noise_after_the_shift_day():
    base, shifted = S.generate(4, seed=5, shift=1.0), S.generate(4, seed=5, shift=2.0)
    assert np.allclose(shifted.sigma[:, :C.SHIFT_DAY], base.sigma[:, :C.SHIFT_DAY]) and np.allclose(shifted.sigma[:, C.SHIFT_DAY:], 2.0 * base.sigma[:, C.SHIFT_DAY:])
    assert np.array_equal(base.mu, shifted.mu) and np.array_equal(base.y[:, :C.SHIFT_DAY], shifted.y[:, :C.SHIFT_DAY])


def test_log_noise_matches_the_true_sigma():
    p = S.generate(6, noise_mean=0.2, seed=2, events=0.5)
    r = np.log((p.y + 1) / (p.mu + 1))
    for i in range(p.n):
        assert r[i].std() == pytest.approx(p.sigma[i, 0], abs=0.03 + 0.5 / p.mu[i].mean() ** 0.5)


def test_outliers_fatten_the_tails():
    plain, out = S.generate(6, seed=3, outliers=0.0), S.generate(6, seed=3, outliers=0.05)
    r0, r1 = np.log((plain.y + 1) / (plain.mu + 1)), np.log((out.y + 1) / (out.mu + 1))
    assert np.mean(np.abs(r1) > 0.6) > np.mean(np.abs(r0) > 0.6) + 0.01 and r1.std() > r0.std()


def test_point_models_estimate_parameters_only_before_fit_end():
    port = S.generate(2, seed=4)
    y = port.y[0]
    X = B.regression_design(port.dow, port.holiday, port.after, port.promo[0])
    org = np.array([700, 800])
    got = I.point_forecasts(y, X, "regression", org, 5)
    Xt, z = X[:C.FIT_END], np.log(np.maximum(y[:C.FIT_END], 1.0))
    A = Xt.T @ Xt
    beta = np.linalg.solve(A + 1e-3 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), Xt.T @ z)
    assert np.allclose(got[0], np.exp(X[700:705] @ beta)) and np.allclose(got[1], np.exp(X[800:805] @ beta))
    y2 = y.copy()
    y2[C.FIT_END:] = 1.0                                                   # nach dem Anpassungsende ändern: Regression-Parameter bleiben
    assert np.allclose(I.point_forecasts(y2, X, "regression", org, 5), got)


def _settings(**kw):
    return E.Settings(n_depots=10, horizon=7, **kw)


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(_settings())


def test_analysis_shapes(analysis):
    a = analysis
    n_org = C.N_DAYS - 7 - C.FIRST_TEST + 1
    assert a.forecast.shape == (10, n_org, 7) == a.actual.shape and set(a.quantiles) == set(C.METHODS) | {"oracle"}
    assert all(q.shape == (10, n_org, 7, 6) for q in a.quantiles.values()) and a.alpha_trace.shape == (3, n_org, 10, 7)
    t = a.test_org[5]
    assert np.array_equal(a.actual[3, 5], a.port.y[3, t:t + 7])


def test_quantiles_are_monotone_in_the_level(analysis):
    for m, q in analysis.quantiles.items():
        assert (np.diff(q, axis=-1) >= -1e-4).all(), m
        assert (q >= 0).all()


def test_oracle_is_calibrated_and_best(analysis):
    s = analysis.summary
    for p in C.NOMINALS:
        assert s["oracle"]["coverage"][p] == pytest.approx(p, abs=0.03)
    assert min(s, key=lambda m: s[m]["pinball"]) == "oracle" and min(s, key=lambda m: s[m]["iscore"][0.8]) == "oracle"


def test_summary_by_hand():
    Y = np.array([[[1.0, 5.0], [10.0, 20.0]]])                            # (1, 2, 2)
    Q = np.zeros(Y.shape + (6,))
    Q[..., I.LEVEL_INDEX[0.1]], Q[..., I.LEVEL_INDEX[0.9]] = 2.0, 12.0    # Intervall [2, 12]
    Q[..., I.LEVEL_INDEX[0.25]], Q[..., I.LEVEL_INDEX[0.75]] = 4.0, 8.0
    Q[..., I.LEVEL_INDEX[0.025]], Q[..., I.LEVEL_INDEX[0.975]] = 0.0, 30.0
    out = E._summary({"m": Q}, Y, np.array([2.0]))["m"]
    # 80 %: getroffen sind 5 und 10 -> 2/4; Breite 10; Überschreitungen: 1 (unten 1), 8 (oben 20 - 12); alpha = 0,2 -> 2/alpha = 10
    assert out["coverage"][0.8] == 0.5 and out["coverage"][0.95] == 1.0 and out["coverage"][0.5] == 0.25
    assert out["width"][0.8] == pytest.approx(10 / 2) and out["iscore"][0.8] == pytest.approx((10 + (10 * 1 + 10 * 8) / 4) / 2)


def test_breakdowns_average_back_to_the_overall_coverage(analysis):
    a = analysis
    for m in C.METHODS:
        cov = a.summary[m]["coverage"][0.8]
        assert E.coverage_by_horizon(a, 0.8)[m].mean() == pytest.approx(cov)
    days = a.test_org[:, None] + np.arange(7)[None, :]
    dow = a.port.dow[days]
    weights = np.array([np.sum(dow == d) for d in range(7)])
    wd = E.coverage_by_weekday(a, 0.8)["gauss_raw"]
    assert (wd * weights).sum() / weights.sum() == pytest.approx(a.summary["gauss_raw"]["coverage"][0.8])
    tl = E.coverage_by_target_day(a, 0.8)["conformal"]
    assert tl.shape == (C.N_DAYS - C.FIRST_TEST,) and 0.6 < tl.mean() < 0.95


def test_quantiles_at_an_origin_ignore_the_future():
    s = _settings()
    port = E._portfolio(s.portfolio_key)
    F_all, org_all = E._point(s.portfolio_key, s.point, s.horizon)
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - s.horizon + 1)
    Q1, _ = E._quantile_dict(port, F_all, org_all, s, test_org)
    t = 800
    y2 = port.y.copy()
    y2[:, t:] = np.random.default_rng(0).integers(0, 999, size=y2[:, t:].shape)
    port2 = dataclasses.replace(port, y=y2)
    F2 = np.stack([I.point_forecasts(y2[i], None, s.point, org_all, s.horizon) for i in range(port.n)])
    Q2, _ = E._quantile_dict(port2, F2, org_all, s, test_org)
    i = t - C.FIRST_TEST
    for m in C.METHODS:
        assert np.array_equal(Q1[m][:, :i + 1], Q2[m][:, :i + 1]), m
    assert not np.array_equal(Q1["conformal"], Q2["conformal"])


def test_aci_with_zero_step_equals_conformal():
    a = E.analyse(_settings(gamma=0.0))
    assert np.allclose(a.quantiles["aci"], a.quantiles["conformal"], atol=1e-3)


def test_horizon_one_and_all_point_models():
    for pt in C.POINT_MODELS:
        a = E.analyse(E.Settings(n_depots=10, horizon=1, point=pt))
        assert a.forecast.shape[2] == 1 and np.isfinite(a.point_mase) and a.summary["conformal"]["coverage"][0.8] == pytest.approx(0.8, abs=0.06)


def test_oracle_matches_a_direct_simulation():
    port = E._portfolio(_settings().portfolio_key)
    a = E.analyse(_settings())
    i, j = 40, 3
    t = a.test_org[i] + j
    sim = port.mu[2, t] * np.exp(port.sigma[2, t] * np.random.default_rng(0).normal(size=200000) - 0.5 * port.sigma[2, t] ** 2)
    assert a.quantiles["oracle"][2, i, j, I.LEVEL_INDEX[0.9]] == pytest.approx(np.quantile(sim, 0.9), rel=0.01)

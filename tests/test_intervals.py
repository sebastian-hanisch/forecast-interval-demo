"""Die Intervall-Verfahren von Hand nachgerechnet und gegen unabhängige Schleifen / scipy (nur als Gegenprobe in requirements-dev); keine Verwendung von Werten, die zum Zeitpunkt des Ursprungs unbekannt sind."""

import numpy as np
import pytest

import fi_constants as C
import fi_intervals as I


def test_nominal_levels_and_normal_quantiles():
    assert I.nominal_levels(0.8) == (0.1, 0.9) and I.nominal_levels(0.95) == (0.025, 0.975) and I.nominal_levels(0.5) == (0.25, 0.75)
    stats = pytest.importorskip("scipy.stats")
    for tau in I.LEVELS:
        assert I.z(tau) == pytest.approx(stats.norm.ppf(tau), abs=1e-9)


def test_to_orders_by_hand():
    assert I.to_orders(np.array([99.0]), np.array([np.log(2.0)]))[0] == pytest.approx(199.0)         # (99+1)*2-1
    assert I.to_orders(np.array([0.0]), np.array([-5.0]))[0] == 0.0                                    # nie negativ


def test_scores_by_hand():
    y = np.arange(10, dtype=float) + 1                     # 1..10
    F = np.array([[1.0, 1.0], [3.0, 3.0]])                # Ursprünge 3 und 6, Horizonte 1 und 2
    org = np.array([3, 6])
    got = I.scores(y, F, org)
    assert got[0, 0] == pytest.approx(np.log((y[3] + 1) / 2)) and got[0, 1] == pytest.approx(np.log((y[4] + 1) / 2))
    assert got[1, 1] == pytest.approx(np.log((y[7] + 1) / 4))


def test_window_quantile_by_hand():
    A = np.arange(1.0, 11.0)                              # m = 10
    assert I.window_quantile(A, 0.9) == 10                # ceil(11 * 0.9) = 10
    assert I.window_quantile(A, 0.1) == 1                 # floor(11 * 0.1) = 1
    assert I.window_quantile(A, 0.75) == 9                # ceil(8.25) = 9
    assert I.window_quantile(A, 0.25) == 2                # floor(2.75) = 2
    B = np.array([1.0, 2.0, 3.0, np.nan, np.nan])         # m = 3: 0,975 verlangt den 4. von 3 Werten -> Rand; 0,025 den 0. -> 1.
    assert I.window_quantile(B, 0.975) == 3 and I.window_quantile(B, 0.025) == 1


def test_conformal_interval_has_the_finite_sample_coverage():
    # m = 19 austauschbare Werte, Stufen 0,1 / 0,9: k_l = floor(20 * 0,1) = 2, k_u = ceil(20 * 0,9) = 18 -> Abdeckung (18 - 2) / 20 = 0,8 (der neue Wert liegt mit Rang 3..18 von 20 zwischen den beiden)
    rng = np.random.default_rng(0)
    N = 100000
    cal = np.sort(rng.normal(size=(N, 19)), axis=1)
    new = rng.normal(size=N)
    lo, hi = I.window_quantile(cal, 0.1), I.window_quantile(cal, 0.9)
    assert np.mean((new >= lo) & (new <= hi)) == pytest.approx(0.80, abs=0.004)


def _naive_conformal(S_ext, W, t, k, tau):
    e = t - k - C.FIT_END
    vals = np.sort([S_ext[e - j] for j in range(W) if e - j >= 0])
    m = len(vals)
    kk = int(np.ceil((m + 1) * tau)) if tau > 0.5 else int(np.floor((m + 1) * tau))
    return vals[min(max(kk, 1), m) - 1]


def test_conformal_quantiles_match_a_naive_loop():
    rng = np.random.default_rng(1)
    h = 3
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - h + 1)
    S = rng.normal(size=(2, C.N_DAYS - h + 1 - C.FIT_END, h))
    got = I.conformal_quantiles(S, 45, test_org, h)
    for dep in range(2):
        for i in (0, 1, 100, len(test_org) - 1):
            for k in range(1, h + 1):
                for a, tau in enumerate(I.LEVELS):
                    assert got[dep, i, k - 1, a] == pytest.approx(_naive_conformal(S[dep, :, k - 1], 45, test_org[i], k, tau))


def test_conformal_uses_only_realised_scores():
    rng = np.random.default_rng(2)
    h = 4
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - h + 1)
    S = rng.normal(size=(1, C.N_DAYS - h + 1 - C.FIT_END, h))
    base = I.conformal_quantiles(S, 60, test_org, h)
    t = 800
    S2 = S.copy()
    for k in range(1, h + 1):
        S2[:, t - k - C.FIT_END + 1:, k - 1] = 99.0                    # alle Scores mit Zieltag >= t verändern
    changed = I.conformal_quantiles(S2, 60, test_org, h)
    assert np.array_equal(base[:, :t - C.FIRST_TEST + 1], changed[:, :t - C.FIRST_TEST + 1])
    assert not np.array_equal(base, changed)


def test_aci_without_adaptation_equals_conformal_and_updates_by_hand():
    rng = np.random.default_rng(3)
    h = 2
    n = 2
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - h + 1)
    S = rng.normal(0, 0.2, size=(n, C.N_DAYS - h + 1 - C.FIT_END, h))
    F = rng.uniform(50, 150, size=(n, len(test_org), h))
    y = rng.uniform(20, 200, size=(n, C.N_DAYS))
    q0, tr0 = I.aci_quantiles(S, y, F, 60, test_org, h, 0.0)
    conf = I.conformal_quantiles(S, 60, test_org, h)
    for a, p in enumerate(C.NOMINALS):
        lo, hi = I.nominal_levels(p)
        assert np.allclose(q0[..., 2 * a], conf[..., I.LEVEL_INDEX[lo]]) and np.allclose(q0[..., 2 * a + 1], conf[..., I.LEVEL_INDEX[hi]])
    assert np.allclose(tr0[1], 1 - 0.8)
    gamma = 0.05
    q, tr = I.aci_quantiles(S, y, F, 60, test_org, h, gamma)
    for a, p in enumerate(C.NOMINALS):
        assert np.allclose(tr[a, 0], 1 - p)                              # Start: α = 1 - p
        lo_o = I.to_orders(F[:, 0, 0], q[:, 0, 0, 2 * a])
        hi_o = I.to_orders(F[:, 0, 0], q[:, 0, 0, 2 * a + 1])
        err = ((y[:, test_org[0]] < lo_o) | (y[:, test_org[0]] > hi_o)).astype(float)                   # Ursprung 730, Horizont 1: Zieltag 730, Rückmeldung bei Ursprung 731
        assert np.allclose(tr[a, 1, :, 0], (1 - p) + gamma * ((1 - p) - err))


def test_oracle_quantiles_are_lognormal():
    stats = pytest.importorskip("scipy.stats")
    mu = np.full((1, C.N_DAYS), 100.0)
    sigma = np.full((1, C.N_DAYS), 0.2)
    Q = I.oracle_quantiles(mu, sigma, np.array([800]), 1)
    for a, tau in enumerate(I.LEVELS):
        assert Q[0, 0, 0, a] == pytest.approx(stats.lognorm.ppf(tau, s=0.2, scale=100.0 * np.exp(-0.5 * 0.04)))
    assert Q[0, 0, 0, 0] < Q[0, 0, 0, 2] < Q[0, 0, 0, 5]


def test_in_sample_one_step_uses_only_training_origins():
    y = np.arange(C.N_DAYS, dtype=float) + 10
    org = np.arange(I.FIRST_ORIGIN, 800)
    F1 = np.full(len(org), 50.0)
    raw, log = I.in_sample_one_step(y, F1, org)
    assert len(raw) == C.FIT_END - I.FIRST_ORIGIN and raw[0] == y[I.FIRST_ORIGIN] - 50 and log[0] == pytest.approx(np.log((y[I.FIRST_ORIGIN] + 1) / 51))

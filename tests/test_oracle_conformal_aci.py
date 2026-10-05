"""Orakel: die konformen Quantile und das adaptive Verfahren (ACI) noch einmal als zeitlich ablaufende Schleife aus der Definition (Rang ceil((m+1)τ) bzw. floor((m+1)τ) in exakten Brüchen; Fehlerrate nach Gibbs/Candès
mit der Rückmeldung 'Prognose von Ursprung t-k, Horizont k, hat den Zieltag t-1'), dazu Intervall-Score und Pinball-Verlust einer kleinen Analyse per Schleife."""

import math
from fractions import Fraction

import numpy as np
import pytest

import fi_constants as C
import fi_evaluation as EV
import fi_intervals as I


def _rank(m, tau):
    ft = Fraction(repr(tau))
    k = math.ceil((m + 1) * ft) if tau > 0.5 else math.floor((m + 1) * ft)
    return min(max(k, 1), m)


def _window(S, d, k, t, W):
    last = t - k                                                    # Ursprung o hat den Zieltag o + k - 1 <= t - 1
    return sorted(S[d, o - C.FIT_END, k - 1] for o in range(max(C.FIT_END, last - W + 1), last + 1))


def test_conformal_quantiles_equal_the_rank_definition_in_exact_fractions():
    rng = np.random.default_rng(1717)
    for inst in range(12):
        h, W, n_test = int(rng.integers(1, 6)), int(rng.integers(3, 121)), 6
        test_org = np.arange(730, 730 + n_test)
        S = np.round(rng.normal(size=(2, 730 + n_test + h - C.FIT_END, h)), 1 if inst % 2 else 6)       # jede zweite Instanz mit Gleichständen
        Q = I.conformal_quantiles(S, W, test_org, h)
        for d in range(2):
            for ti, t in enumerate(test_org):
                for k in range(1, h + 1):
                    vals = _window(S, d, k, t, W)
                    for a, tau in enumerate(I.LEVELS):
                        assert Q[d, ti, k - 1, a] == vals[_rank(len(vals), tau) - 1]


def test_aci_equals_a_sequential_loop_over_the_definition():
    rng = np.random.default_rng(18)
    for inst in range(8):
        h, W, n_test, gamma = int(rng.integers(1, 5)), int(rng.integers(5, 100)), 12, float(rng.choice([0.0, 0.02, 0.1]))
        test_org = np.arange(730, 730 + n_test)
        S = rng.normal(size=(1, 730 + n_test + h - C.FIT_END, h)) * 0.3
        F = rng.uniform(5, 100, size=(1, n_test, h))
        y = rng.uniform(0, 150, size=(1, 1095))
        q, trace = I.aci_quantiles(S, y, F, W, test_org, h, gamma)
        for pi, p in enumerate(C.NOMINALS):
            alpha_nom, alpha, store = 1 - p, [1 - p] * h, {}
            for i, t in enumerate(test_org):
                for k in range(1, h + 1):
                    if i - k >= 0:
                        lo, hi = store[(i - k, k)]
                        alpha[k - 1] += gamma * (alpha_nom - (1.0 if (y[0, t - 1] < lo or y[0, t - 1] > hi) else 0.0))
                assert trace[pi, i, 0, :] == pytest.approx(alpha, abs=1e-12)
                for k in range(1, h + 1):
                    vals = _window(S, 0, k, t, W)
                    m, al = len(vals), min(max(alpha[k - 1], 0.0), 1.0)
                    qlo = vals[min(max(math.floor((m + 1) * al / 2), 1), m) - 1]
                    qhi = vals[min(max(math.ceil((m + 1) * (1 - al / 2)), 1), m) - 1]
                    assert (q[0, i, k - 1, 2 * pi], q[0, i, k - 1, 2 * pi + 1]) == (qlo, qhi)
                    store[(i, k)] = tuple(max((F[0, i, k - 1] + 1) * math.exp(v) - 1, 0.0) for v in (qlo, qhi))


def test_summary_metrics_equal_loops_on_a_small_analysis():
    a = EV.analyse(EV.Settings(n_depots=10, horizon=3, window=30, seed=5, point="snaive_k"))
    n, T, h = a.actual.shape
    scale = np.array([np.mean([abs(a.port.y[i, d] - a.port.y[i, d - 7]) for d in range(7, C.FIRST_TEST)]) for i in range(n)])
    for m in ("conformal", "aci", "oracle"):
        q = a.quantiles[m].astype(float)
        for p in C.NOMINALS:
            lo, hi = I.nominal_levels(p)
            il, ih, al = I.LEVELS.index(lo), I.LEVELS.index(hi), 1 - p
            hits = wid = isc = 0.0
            for i in range(n):
                for ti in range(T):
                    for k in range(h):
                        y, l, u = a.actual[i, ti, k], q[i, ti, k, il], q[i, ti, k, ih]
                        hits += (l <= y <= u)
                        wid += (u - l) / scale[i]
                        isc += ((u - l) + 2 / al * max(l - y, 0) + 2 / al * max(y - u, 0)) / scale[i]
            N = n * T * h
            assert (a.summary[m]["coverage"][p], a.summary[m]["width"][p], a.summary[m]["iscore"][p]) == pytest.approx((hits / N, wid / N, isc / N), rel=1e-6)
        pin = 0.0
        for i in range(n):
            for ti in range(T):
                for k in range(h):
                    for ai, tau in enumerate(I.LEVELS):
                        e = a.actual[i, ti, k] - q[i, ti, k, ai]
                        pin += (tau * e if e >= 0 else (tau - 1) * e) / scale[i] / len(I.LEVELS)
        assert a.summary[m]["pinball"] == pytest.approx(pin / (n * T * h), rel=1e-6)

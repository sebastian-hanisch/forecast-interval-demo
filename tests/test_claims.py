"""Jede Zahl aus README und PRESET_HELP als Test. Die Verfahren sind deterministisch bei festem Seed; Bänder um die gerundeten Angaben, dazu Rangfolgen mit Abstand
(numpy-Versionen und Plattformen können den Zufallsstrom des HW-Fits um Rundung verschieben, feedback_ci_platform_robust_tests)."""

from functools import lru_cache

import numpy as np
import pytest

import fi_constants as C
import fi_evaluation as E
import fi_presets as P

STD = "Standardfall: Holt-Winters, 30 Depots"


@lru_cache(maxsize=None)
def _preset(name):
    v = P.PRESETS[name]
    return E.analyse(E.Settings(v["n_depots"], v["noise"], v["events"], v["trend"], v["horizon"], v["shift"], v["outliers"], v["point"], v["window"], v["gamma"], v["seed"]))


def _cov(a, p):
    return {m: a.summary[m]["coverage"][p] for m in a.summary}


def test_standard_preset():
    a = _preset(STD)
    c = _cov(a, 0.8)
    assert c["gauss_raw"] == pytest.approx(0.757, abs=0.02) and c["gauss_log"] == pytest.approx(0.780, abs=0.02) and c["empirical"] == pytest.approx(0.765, abs=0.02)
    assert c["conformal"] == pytest.approx(0.794, abs=0.015) and c["aci"] == pytest.approx(0.799, abs=0.01) and c["oracle"] == pytest.approx(0.797, abs=0.01)
    assert c["gauss_raw"] < c["gauss_log"] < c["conformal"] and c["empirical"] < c["conformal"]
    c95 = _cov(a, 0.95)
    assert c95["gauss_raw"] == pytest.approx(0.89, abs=0.02) and c95["gauss_log"] == pytest.approx(0.931, abs=0.02) and c95["conformal"] == pytest.approx(0.952, abs=0.01)
    assert a.point_mase == pytest.approx(0.869, abs=0.03)
    wd = E.coverage_by_weekday(a, 0.8)["gauss_raw"]
    assert wd.min() == pytest.approx(0.629, abs=0.04) and wd.max() == pytest.approx(0.978, abs=0.03) and wd.argmax() == 6 and wd.argmin() in (0, 1, 2, 3, 4)


def test_standard_scores_and_the_price_of_calibration():
    s = _preset(STD).summary
    assert s["gauss_log"]["iscore"][0.8] == pytest.approx(3.99, abs=0.2) and s["conformal"]["iscore"][0.8] == pytest.approx(4.08, abs=0.2) and s["oracle"]["iscore"][0.8] == pytest.approx(3.15, abs=0.1)
    assert s["gauss_raw"]["iscore"][0.8] > s["conformal"]["iscore"][0.8] > s["oracle"]["iscore"][0.8] and s["gauss_raw"]["pinball"] > s["gauss_log"]["pinball"] > s["oracle"]["pinball"]
    assert s["conformal"]["width"][0.8] > s["gauss_log"]["width"][0.8] > s["gauss_raw"]["width"][0.8] and s["oracle"]["width"][0.8] == pytest.approx(2.28, abs=0.1)
    assert s["conformal"]["pinball"] == pytest.approx(0.211, abs=0.01) and s["oracle"]["pinball"] == pytest.approx(0.165, abs=0.01)
    assert s["gauss_raw"]["iscore"][0.8] == pytest.approx(4.48, abs=0.25) and s["aci"]["iscore"][0.8] == pytest.approx(4.16, abs=0.2) and s["gauss_raw"]["pinball"] == pytest.approx(0.228, abs=0.01) and s["gauss_log"]["pinball"] == pytest.approx(0.207, abs=0.01) and s["aci"]["pinball"] == pytest.approx(0.214, abs=0.01)
    assert s["gauss_raw"]["width"][0.8] == pytest.approx(2.48, abs=0.15) and s["gauss_log"]["width"][0.8] == pytest.approx(2.66, abs=0.15) and s["conformal"]["width"][0.8] == pytest.approx(2.81, abs=0.15) and s["aci"]["width"][0.8] == pytest.approx(2.92, abs=0.15)


def test_regression_preset():
    a = _preset("Regression als Punktmodell")
    c = _cov(a, 0.8)
    assert a.point_mase == pytest.approx(0.784, abs=0.03) and c["conformal"] == pytest.approx(0.80, abs=0.015) and c["gauss_log"] == pytest.approx(0.752, abs=0.02)
    hz = E.coverage_by_horizon(a, 0.8)["conformal"]
    assert abs(hz[0] - hz[-1]) < 0.02
    assert a.summary["conformal"]["iscore"][0.8] == pytest.approx(3.25, abs=0.15) and a.summary["oracle"]["iscore"][0.8] == pytest.approx(3.15, abs=0.1)
    assert a.summary["conformal"]["iscore"][0.8] < a.summary["gauss_log"]["iscore"][0.8]


def test_shift_preset():
    a = _preset("Verteilungswechsel (Rauschen ×2)")
    after = E.coverage_after_shift(a, 0.8)
    assert after["gauss_log"] == pytest.approx(0.509, abs=0.03) and after["empirical"] == pytest.approx(0.496, abs=0.03) and after["conformal"] == pytest.approx(0.733, abs=0.03) and after["aci"] == pytest.approx(0.80, abs=0.015)
    assert after["gauss_log"] + 0.15 < after["conformal"] < after["aci"]


def test_outlier_preset():
    a = _preset("Ausreißer (5 %)")
    c = _cov(a, 0.5)
    assert c["gauss_raw"] == pytest.approx(0.670, abs=0.03) and c["gauss_log"] == pytest.approx(0.601, abs=0.03) and c["empirical"] == pytest.approx(0.465, abs=0.03) and c["conformal"] == pytest.approx(0.490, abs=0.02) and c["aci"] == pytest.approx(0.499, abs=0.015)
    assert c["gauss_raw"] > c["gauss_log"] > 0.55 and abs(c["conformal"] - 0.5) < 0.03
    s = a.summary                                                                                     # bei 95 %: ACI wird nach Ausreißern breit
    assert s["aci"]["width"][0.95] == pytest.approx(8.11, abs=0.6) and s["conformal"]["width"][0.95] == pytest.approx(6.0, abs=0.4) and s["gauss_log"]["width"][0.95] == pytest.approx(4.75, abs=0.4) and s["empirical"]["width"][0.95] == pytest.approx(4.05, abs=0.4) and s["oracle"]["width"][0.95] == pytest.approx(2.73, abs=0.3)
    assert s["aci"]["width"][0.95] > s["conformal"]["width"][0.95] + 1.0 and s["aci"]["iscore"][0.95] == pytest.approx(11.59, abs=1.0) and s["conformal"]["iscore"][0.95] == pytest.approx(9.82, abs=0.8) and s["gauss_log"]["iscore"][0.95] == pytest.approx(8.69, abs=0.8)
    assert s["aci"]["iscore"][0.95] > s["conformal"]["iscore"][0.95] > s["empirical"]["iscore"][0.95]


def test_small_window_preset():
    a = _preset("Kleines Kalibrierfenster (30)")
    c = _cov(a, 0.95)
    assert c["conformal"] == pytest.approx(0.917, abs=0.02) and c["gauss_log"] == pytest.approx(0.931, abs=0.02) and c["empirical"] == pytest.approx(0.933, abs=0.02)
    assert c["conformal"] < 0.95 - 0.015 and c["aci"] == pytest.approx(c["conformal"], abs=0.005)


def test_long_horizon_preset():
    a = _preset("Langer Horizont (28 Tage)")
    hz = E.coverage_by_horizon(a, 0.8)
    assert hz["gauss_log"][0] == pytest.approx(0.798, abs=0.02) and hz["gauss_log"][-1] == pytest.approx(0.728, abs=0.04) and hz["conformal"][0] == pytest.approx(0.804, abs=0.02) and hz["conformal"][-1] == pytest.approx(0.741, abs=0.04)
    assert hz["aci"][-1] == pytest.approx(0.791, abs=0.02) and hz["aci"][-1] > hz["conformal"][-1] + 0.02 and hz["gauss_log"][0] - hz["gauss_log"][-1] > 0.03


def test_finite_sample_coverage_of_the_default_window():
    # 90 Scores, Stufen 0,1 / 0,9: k_l = floor(91 * 0,1) = 9, k_u = ceil(91 * 0,9) = 82 -> (82 - 9) / 91 = 0,802; für 95 %: floor(2,275) = 2, ceil(88,725) = 89 -> 87 / 91 = 0,956
    m = 90
    assert (int(np.ceil((m + 1) * 0.9)) - int(np.floor((m + 1) * 0.1))) / (m + 1) == pytest.approx(0.802, abs=0.001)
    assert (int(np.ceil((m + 1) * 0.975)) - int(np.floor((m + 1) * 0.025))) / (m + 1) == pytest.approx(0.956, abs=0.001)
    # 30 Scores, Stufe 0,975: ceil(31 * 0,975) = 31 > 30 -> Rand; die Nennabdeckung ist mit 30 Werten nicht mehr erreichbar (höchstens (30 - 1) / 31 = 0,935 mit Randquantilen)
    m = 30
    assert int(np.ceil((m + 1) * 0.975)) == 31 and (m - 1) / (m + 1) == pytest.approx(0.935, abs=0.001)


@lru_cache(maxsize=None)
def _shift():
    return E.shift_experiment(C.SHIFT_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _outliers():
    return E.outlier_experiment(C.OUTLIER_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _window():
    return E.window_experiment(C.WINDOW_LEVELS, C.EXP_SEEDS)


def test_shift_experiment():
    r = {x["shift"]: {m: v[0] for m, v in x["coverage"].items()} for x in _shift()}
    assert r[1.0]["gauss_log"] == pytest.approx(0.782, abs=0.02) and r[1.0]["empirical"] == pytest.approx(0.772, abs=0.02) and r[1.0]["conformal"] == pytest.approx(0.796, abs=0.015) and r[1.0]["aci"] == pytest.approx(0.801, abs=0.01)
    assert r[1.5]["gauss_log"] == pytest.approx(0.623, abs=0.03) and r[1.5]["conformal"] == pytest.approx(0.759, abs=0.03) and r[1.5]["aci"] == pytest.approx(0.801, abs=0.01)
    assert r[2.0]["gauss_log"] == pytest.approx(0.504, abs=0.03) and r[2.0]["empirical"] == pytest.approx(0.494, abs=0.03) and r[2.0]["conformal"] == pytest.approx(0.733, abs=0.03) and r[2.0]["aci"] == pytest.approx(0.802, abs=0.01)
    assert r[2.0]["gauss_raw"] == pytest.approx(0.539, abs=0.04)
    assert r[1.0]["gauss_log"] > r[1.5]["gauss_log"] > r[2.0]["gauss_log"] and r[2.0]["gauss_log"] + 0.15 < r[2.0]["conformal"] < r[2.0]["aci"]


def test_outlier_experiment():
    r = {x["outliers"]: x for x in _outliers()}
    cov = {o: {m: v[0] for m, v in x["coverage"].items()} for o, x in r.items()}
    wid = {o: {m: v[0] for m, v in x["width"].items()} for o, x in r.items()}
    assert cov[0.0]["gauss_raw"] == pytest.approx(0.525, abs=0.03) and cov[0.0]["conformal"] == pytest.approx(0.512, abs=0.02)
    assert cov[0.05]["gauss_raw"] == pytest.approx(0.662, abs=0.03) and cov[0.05]["gauss_log"] == pytest.approx(0.591, abs=0.03) and cov[0.05]["empirical"] == pytest.approx(0.474, abs=0.03) and cov[0.05]["conformal"] == pytest.approx(0.498, abs=0.02)
    assert wid[0.05]["gauss_raw"] == pytest.approx(1.87, abs=0.15) and wid[0.05]["empirical"] == pytest.approx(1.22, abs=0.1) and wid[0.05]["conformal"] == pytest.approx(1.32, abs=0.1)
    assert cov[0.0]["gauss_raw"] < cov[0.02]["gauss_raw"] < cov[0.05]["gauss_raw"] and wid[0.05]["gauss_raw"] > wid[0.05]["conformal"] + 0.3 and wid[0.05]["gauss_raw"] > wid[0.0]["gauss_raw"] + 0.4


def test_window_experiment():
    r = {x["window"]: x for x in _window()}
    assert r[30]["conformal"]["coverage"][0] == pytest.approx(0.918, abs=0.02) and r[30]["aci"]["coverage"][0] == pytest.approx(r[30]["conformal"]["coverage"][0], abs=0.005)
    assert r[120]["conformal"]["coverage"][0] == pytest.approx(0.945, abs=0.015) and r[120]["aci"]["coverage"][0] == pytest.approx(0.949, abs=0.01)
    assert r[120]["conformal"]["sd_depot"][0] == pytest.approx(0.011, abs=0.006) and r[120]["aci"]["sd_depot"][0] == pytest.approx(0.005, abs=0.004)
    assert r[30]["conformal"]["coverage"][0] < 0.95 - 0.015 and r[120]["aci"]["sd_depot"][0] < r[120]["conformal"]["sd_depot"][0]
    assert r[30]["conformal"]["width"][0] == pytest.approx(4.51, abs=0.4) and r[120]["conformal"]["width"][0] == pytest.approx(4.62, abs=0.4)

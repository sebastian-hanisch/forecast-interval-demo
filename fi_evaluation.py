"""Auswertung der Prognoseintervalle: ein Punktmodell je Depot, fünf Intervall-Verfahren und das Orakel im Rolling-Origin-Vergleich über das Testjahr, dazu drei Experimente (Verteilungswechsel, Ausreißer, Kalibrierfenster).

Kennzahlen (je Nennabdeckung p = 50 %, 80 %, 95 %): **Abdeckung** (Anteil der Ist-Werte im Intervall), **Breite** und **Intervall-Score** (Winkler: Breite + 2/α mal Überschreitung, α = 1 - p) - beide durch den saisonal
naiven Trainingsfehler des Depots geteilt (wie die MASE), dann über die Depots gemittelt - und der **Pinball-Verlust** über alle sechs Quantilstufen."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import fi_baselines as B
import fi_constants as C
import fi_intervals as I
import fi_scenario as S


@dataclass(frozen=True)
class Settings:
    n_depots: int = C.DEFAULT_DEPOTS
    noise: float = C.DEFAULT_NOISE
    events: float = C.DEFAULT_EVENTS
    trend: int = C.DEFAULT_TREND
    horizon: int = C.DEFAULT_HORIZON
    shift: float = C.DEFAULT_SHIFT
    outliers: float = C.DEFAULT_OUTLIER
    point: str = "hw_mult"
    window: int = C.DEFAULT_WINDOW
    gamma: float = C.DEFAULT_GAMMA
    seed: int = 3

    @property
    def portfolio_key(self):
        return (self.n_depots, self.noise, self.events, self.trend, self.shift, self.outliers, self.seed)


@dataclass
class Analysis:
    settings: Settings
    port: S.Portfolio
    test_org: np.ndarray
    forecast: np.ndarray            # (n, T, h) Punktprognose
    actual: np.ndarray              # (n, T, h)
    quantiles: dict                 # Verfahren -> (n, T, h, 6) Quantile in Aufträgen (Verfahren + "oracle")
    alpha_trace: np.ndarray         # (3, T, n, h) Fehlerrate des adaptiven Verfahrens
    scale: np.ndarray               # (n,) MASE-Nenner
    summary: dict                   # Verfahren -> Kennzahlen
    point_mase: float


def _scales(port):
    return np.array([B.mase_scale(port.y[i]) for i in range(port.n)])


@lru_cache(maxsize=16)
def _portfolio(key):
    n, noise, events, trend, shift, outliers, seed = key
    return S.generate(n, noise, events, float(trend), seed, shift, outliers)


@lru_cache(maxsize=16)
def _point(key, model, horizon):
    """Punktprognosen für alle Ursprünge ab FIRST_ORIGIN (auch die im Training und in der Kalibrierung): (n, O_all, h) und die Ursprünge."""
    port = _portfolio(key)
    org = np.arange(I.FIRST_ORIGIN, C.N_DAYS - horizon + 1)
    out = []
    for i in range(port.n):
        X = B.regression_design(port.dow, port.holiday, port.after, port.promo[i]) if model == "regression" else None
        out.append(I.point_forecasts(port.y[i], X, model, org, horizon))
    return np.stack(out), org


def _quantile_dict(port, F_all, org_all, s, test_org):
    h, n = s.horizon, port.n
    idx_test = test_org - I.FIRST_ORIGIN
    F_test = F_all[:, idx_test]
    y = port.y
    S_all = np.stack([I.scores(y[i], F_all[i], org_all) for i in range(n)])                     # (n, O_all, h)
    S_ext = S_all[:, C.FIT_END - I.FIRST_ORIGIN:, :]
    ins_raw, ins_log = [], []
    for i in range(n):
        r, l = I.in_sample_one_step(y[i], F_all[i][:, 0], org_all)
        ins_raw.append(r), ins_log.append(l)
    ins_raw, ins_log = np.stack(ins_raw), np.stack(ins_log)
    zs = np.array([I.z(t) for t in I.LEVELS])
    Q = {}
    sig_raw = ins_raw.std(axis=1)
    Q["gauss_raw"] = np.maximum(F_test[..., None] + sig_raw[:, None, None, None] * zs, 0.0)
    sig_log = ins_log.std(axis=1)
    Q["gauss_log"] = I.to_orders(F_test[..., None], sig_log[:, None, None, None] * zs)
    q_emp = np.quantile(ins_log, I.LEVELS, axis=1).T                                              # (n, 6)
    Q["empirical"] = I.to_orders(F_test[..., None], q_emp[:, None, None, :])
    Q["conformal"] = I.to_orders(F_test[..., None], I.conformal_quantiles(S_ext, s.window, test_org, h))
    q_aci, trace = I.aci_quantiles(S_ext, y, F_test, s.window, test_org, h, s.gamma)
    aci = np.empty_like(Q["conformal"])
    for a, p in enumerate(C.NOMINALS):
        lo, hi = I.nominal_levels(p)
        aci[..., I.LEVEL_INDEX[lo]] = I.to_orders(F_test, q_aci[..., 2 * a])
        aci[..., I.LEVEL_INDEX[hi]] = I.to_orders(F_test, q_aci[..., 2 * a + 1])
    Q["aci"] = aci
    Q["oracle"] = I.oracle_quantiles(port.mu, port.sigma, test_org, h)
    return {m: v.astype(np.float32) for m, v in Q.items()}, trace


def _hits(Q, Y, p):
    lo, hi = I.nominal_levels(p)
    return (Y >= Q[..., I.LEVEL_INDEX[lo]]) & (Y <= Q[..., I.LEVEL_INDEX[hi]])


def _summary(Q, Y, scale):
    out = {}
    sc = scale[:, None, None]
    for m, q in Q.items():
        d = {"coverage": {}, "width": {}, "iscore": {}, "cov_depot": {}}
        for p in C.NOMINALS:
            lo, hi = I.nominal_levels(p)
            ql, qh = q[..., I.LEVEL_INDEX[lo]], q[..., I.LEVEL_INDEX[hi]]
            hit = (Y >= ql) & (Y <= qh)
            al = 1.0 - p
            d["coverage"][p] = float(hit.mean())
            d["cov_depot"][p] = hit.mean(axis=(1, 2))
            d["width"][p] = float(((qh - ql) / sc).mean())
            d["iscore"][p] = float((((qh - ql) + 2.0 / al * np.maximum(ql - Y, 0) + 2.0 / al * np.maximum(Y - qh, 0)) / sc).mean())
        pin = [np.maximum(tau * (Y - q[..., a]), (tau - 1.0) * (Y - q[..., a])) for a, tau in enumerate(I.LEVELS)]
        d["pinball"] = float((np.mean(pin, axis=0) / sc).mean())
        out[m] = d
    return out


@lru_cache(maxsize=6)
def analyse(s):
    port = _portfolio(s.portfolio_key)
    F_all, org_all = _point(s.portfolio_key, s.point, s.horizon)
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - s.horizon + 1)
    Q, trace = _quantile_dict(port, F_all, org_all, s, test_org)
    days = test_org[:, None] + np.arange(s.horizon)[None, :]
    Y = port.y[:, days]
    F_test = F_all[:, test_org - I.FIRST_ORIGIN]
    scale = _scales(port)
    summ = _summary(Q, Y, scale)
    point_mase = float((np.abs(F_test - Y).mean(axis=(1, 2)) / scale).mean())
    return Analysis(s, port, test_org, F_test, Y, Q, trace, scale, summ, point_mase)


# --- Aufschlüsselungen (für die Diagramme) ---------------------------------------------------------------------------------------------------------

def coverage_by_weekday(a, p):
    """Abdeckung je Wochentag des Zieltags: {Verfahren: (7,)}."""
    days = a.test_org[:, None] + np.arange(a.settings.horizon)[None, :]
    dow = a.port.dow[days]
    return {m: np.array([_hits(a.quantiles[m], a.actual, p)[:, dow == d].mean() for d in range(7)]) for m in C.METHODS}


def coverage_by_horizon(a, p):
    return {m: _hits(a.quantiles[m], a.actual, p).mean(axis=(0, 1)) for m in C.METHODS}


def coverage_by_target_day(a, p, mask_from=None):
    """Abdeckung je Zieltag (alle Depots und Horizonte, deren Zieltag das ist): {Verfahren: (Tage,)}, Tage = FIRST_TEST .. N_DAYS - 1."""
    h, T = a.settings.horizon, len(a.test_org)
    days = a.test_org[:, None] + np.arange(h)[None, :]
    out = {}
    for m in C.METHODS:
        hit = _hits(a.quantiles[m], a.actual, p).mean(axis=0)                                     # (T, h)
        tot = np.zeros(C.N_DAYS)
        cnt = np.zeros(C.N_DAYS)
        np.add.at(tot, days.ravel(), hit.ravel())
        np.add.at(cnt, days.ravel(), 1.0)
        out[m] = (tot / np.maximum(cnt, 1.0))[C.FIRST_TEST:]
    return out


def coverage_after_shift(a, p):
    """Abdeckung nur für Zieltage ab dem Verteilungswechsel: {Verfahren: Abdeckung}."""
    h = a.settings.horizon
    days = a.test_org[:, None] + np.arange(h)[None, :]
    m_ = days >= C.SHIFT_DAY
    return {m: float(_hits(a.quantiles[m], a.actual, p)[:, m_].mean()) for m in C.METHODS}


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def shift_experiment(levels=None, seeds=None, p=C.DEFAULT_NOMINAL, base=None):
    """Abdeckung (Nennabdeckung p) der Zieltage ab dem Verteilungswechsel für wachsenden Rauschfaktor."""
    levels = C.SHIFT_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for f in levels:
        cov = {m: [] for m in C.METHODS}
        for sd in seeds:
            a = analyse(_replace(base, shift=f, seed=sd))
            for m, v in coverage_after_shift(a, p).items():
                cov[m].append(v)
        rows.append({"shift": f, "n_seeds": len(seeds), "coverage": {m: _mean_se(v) for m, v in cov.items()}})
    return rows


def outlier_experiment(levels=None, seeds=None, p=0.5, base=None):
    """Abdeckung und Breite (durch die Breite des Orakels geteilt) für wachsenden Anteil an Ausreißertagen."""
    levels = C.OUTLIER_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for o in levels:
        cov = {m: [] for m in C.METHODS}
        wid = {m: [] for m in C.METHODS}
        for sd in seeds:
            a = analyse(_replace(base, outliers=o, seed=sd))
            for m in C.METHODS:
                cov[m].append(a.summary[m]["coverage"][p])
                wid[m].append(a.summary[m]["width"][p] / a.summary["oracle"]["width"][p])
        rows.append({"outliers": o, "n_seeds": len(seeds), "coverage": {m: _mean_se(v) for m, v in cov.items()}, "width": {m: _mean_se(v) for m, v in wid.items()}})
    return rows


def window_experiment(levels=None, seeds=None, p=0.95, base=None):
    """Konform und ACI für verschiedene Kalibrierfenster: Abdeckung, Streuung der Abdeckung zwischen den Depots und Breite."""
    levels = C.WINDOW_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for w in levels:
        r = {"window": w, "n_seeds": len(seeds)}
        for m in ("conformal", "aci"):
            cov, sd_dep, wid = [], [], []
            for sd in seeds:
                a = analyse(_replace(base, window=w, seed=sd))
                cov.append(a.summary[m]["coverage"][p])
                sd_dep.append(float(a.summary[m]["cov_depot"][p].std()))
                wid.append(a.summary[m]["width"][p])
            r[m] = {"coverage": _mean_se(cov), "sd_depot": _mean_se(sd_dep), "width": _mean_se(wid)}
        rows.append(r)
    return rows


def _replace(base, **kw):
    d = {f: getattr(base, f) for f in base.__dataclass_fields__}
    d.update(kw)
    return Settings(**d)

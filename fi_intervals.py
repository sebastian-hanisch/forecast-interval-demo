"""Prognoseintervalle für ein Punktmodell: vier Verfahren aus Fehlern und ein Orakel, alle als Quantile der Aufträge am Zieltag.

Bezeichnungen: Ursprung t = bekannt sind die Tage 0..t-1, prognostiziert wird der Tag t+k-1 (Horizont k). Ein Fehler-Wert ("Score") ist das Log-Verhältnis s = log((y + 1) / (F + 1)) von Ist zu Punktprognose;
ein Quantil q des Scores gibt das Prognosequantil (F + 1) exp(q) - 1 (mindestens 0).

  gauss_raw    F + z σ mit der Standardabweichung σ der Ein-Schritt-Fehler y - F auf den Trainingstagen (eine Zahl je Depot, für alle Tage und Horizonte)
  gauss_log    (F + 1) exp(z σ_log) - 1 mit σ_log der Ein-Schritt-Scores
  empirical    Quantile der Ein-Schritt-Scores auf den Trainingstagen (keine Normalverteilung, aber weiter ein Horizont)
  conformal    Quantile der zuletzt REALISIERTEN Scores DESSELBEN Horizonts (Kalibrierfenster der letzten W Ursprünge, nur Prognosen, deren Zieltag schon bekannt ist), mit der Endlichkeitskorrektur
               k = ceil((m + 1) τ) für obere und k = floor((m + 1) τ) für untere Quantile (m Scores im Fenster)
  aci          Adaptiv konform (Gibbs/Candès 2021): wie conformal, aber die Fehlerrate α_k je Horizont wird nach jeder realisierten Prognose um γ (α - Fehlschlag) nachgeführt
  oracle       die wahren Quantile (log-normales Rauschen mit der wahren Streuung; Ausreißer und Rundung nicht berücksichtigt)"""

from statistics import NormalDist

import numpy as np

import fi_baselines as B
import fi_constants as C

FIRST_ORIGIN = 91
LEVELS = (0.025, 0.1, 0.25, 0.75, 0.9, 0.975)
LEVEL_INDEX = {tau: i for i, tau in enumerate(LEVELS)}
_ND = NormalDist()


def nominal_levels(p):
    """Untere und obere Quantilstufe eines zentralen Intervalls mit Nennabdeckung p."""
    lo = round((1.0 - p) / 2.0, 6)
    return lo, round(1.0 - lo, 6)


def z(tau):
    return _ND.inv_cdf(tau)


def to_orders(F, q):
    """Prognose (F + 1) exp(q) - 1, mindestens 0. F und q müssen broadcasten."""
    return np.maximum((F + 1.0) * np.exp(q) - 1.0, 0.0)


def point_forecasts(y, X, model, org, h):
    """Punktprognosen (n_org, h) eines der drei Modelle; Parameter aus den Tagen vor FIT_END."""
    if model == "hw_mult":
        return B.hw_forecast(y, org, h, first=C.FIT_END)
    if model == "regression":
        return B.regression_forecast(y, X, org, h, first=C.FIT_END)
    if model == "snaive_k":
        return B.snaive_k(y, org, h)
    raise ValueError(model)


def scores(y, F, org):
    """Log-Verhältnisse Ist/Prognose für alle Ursprünge und Horizonte: (n_org, h)."""
    h = F.shape[1]
    days = org[:, None] + np.arange(h)[None, :]
    return np.log((y[days] + 1.0) / (F + 1.0))


def in_sample_one_step(y, F1, org):
    """Ein-Schritt-Fehler roh und im Log auf den Trainingstagen (Ursprünge vor FIT_END): (raw, log)."""
    m = org < C.FIT_END
    act = y[org[m]]
    return act - F1[m], np.log((act + 1.0) / (F1[m] + 1.0))


def _sorted_windows(S_k, W, ends):
    """Sortierte Fenster (NaN am Ende) der Scores eines Horizonts. S_k: (n, O_ext) Scores der Ursprünge ab FIT_END; ends: Index (in S_k) des letzten Ursprungs im Fenster je Testursprung.
    Ergebnis (n, len(ends), W); Fenster vor dem ersten Ursprung sind mit NaN aufgefüllt."""
    n = S_k.shape[0]
    P = np.concatenate([np.full((n, W), np.nan), S_k], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(P, W, axis=1)          # win[:, j] = P[:, j:j+W]; Fenster mit letztem Element S_k[:, e] ist j = e + 1
    return np.sort(win[:, np.asarray(ends) + 1, :], axis=-1)


def window_quantile(A, tau):
    """Konformes Quantil aus sortierten Fenstern A (..., W) (NaN am Ende): obere Stufen ceil((m+1)τ), untere floor((m+1)τ), jeweils auf 1..m begrenzt."""
    m = np.sum(~np.isnan(A), axis=-1)
    kk = np.ceil((m + 1) * tau) if tau > 0.5 else np.floor((m + 1) * tau)
    kk = np.clip(kk, 1, np.maximum(m, 1)).astype(int)
    return np.take_along_axis(A, (kk - 1)[..., None], axis=-1)[..., 0]


def conformal_quantiles(S_ext, W, test_org, h, levels=LEVELS):
    """Konforme Score-Quantile für alle Testursprünge und Horizonte: (n, n_test, h, len(levels)). S_ext: (n, O_ext, h) Scores der Ursprünge ab FIT_END."""
    n = S_ext.shape[0]
    out = np.empty((n, len(test_org), h, len(levels)))
    for k in range(1, h + 1):
        ends = test_org - k - C.FIT_END
        A = _sorted_windows(S_ext[:, :, k - 1], W, ends)
        for a, tau in enumerate(levels):
            out[:, :, k - 1, a] = window_quantile(A, tau)
    return out


def aci_quantiles(S_ext, y, F_test, W, test_org, h, gamma, nominals=C.NOMINALS):
    """Adaptiv konforme Score-Quantile (Gibbs/Candès): je Nennabdeckung ein eigener Lauf mit einer Fehlerrate α je Depot und Horizont. Rückgabe: (n, n_test, h, 2 * len(nominals)) mit den Stufen
    (untere, obere) je Nennabdeckung in der Reihenfolge von `nominals`, dazu die Verläufe der Fehlerrate (len(nominals), n_test, n, h)."""
    n, _, _ = S_ext.shape
    T = len(test_org)
    P = np.concatenate([np.full((n, h, W), np.nan), np.moveaxis(S_ext, 2, 1)], axis=2)          # (n, h, W + O_ext)
    ks = np.arange(1, h + 1)
    alpha_nom = np.array([1.0 - p for p in nominals])
    alpha = np.broadcast_to(alpha_nom[:, None, None], (len(nominals), n, h)).copy()             # (P, n, h)
    q = np.empty((n, T, h, 2 * len(nominals)))
    lo_v = np.empty((len(nominals), n, T, h))
    hi_v = np.empty((len(nominals), n, T, h))
    trace = np.empty((len(nominals), T, n, h))
    for i, t in enumerate(test_org):
        # Rückmeldung: die Prognosen von Ursprung t-k mit Horizont k haben den Zieltag t-1 und sind jetzt realisiert
        o_idx = i - ks
        ok = o_idx >= 0
        if ok.any():
            kk = np.nonzero(ok)[0]
            yt = y[:, t - 1][:, None]
            lo_prev = lo_v[:, :, o_idx[kk], kk]
            hi_prev = hi_v[:, :, o_idx[kk], kk]
            err = ((yt[None] < lo_prev) | (yt[None] > hi_prev)).astype(float)
            alpha[:, :, kk] += gamma * (alpha_nom[:, None, None] - err)
        trace[:, i] = alpha
        e = t - ks - C.FIT_END
        A = np.sort(P[:, np.arange(h)[:, None], (e + 1)[:, None] + np.arange(W)[None, :]], axis=-1)       # (n, h, W)
        m = np.sum(~np.isnan(A), axis=-1)
        for a in range(len(nominals)):
            al = np.clip(alpha[a], 0.0, 1.0)
            k_hi = np.clip(np.ceil((m + 1) * (1.0 - al / 2.0)), 1, np.maximum(m, 1)).astype(int)
            k_lo = np.clip(np.floor((m + 1) * (al / 2.0)), 1, np.maximum(m, 1)).astype(int)
            q_hi = np.take_along_axis(A, (k_hi - 1)[..., None], axis=-1)[..., 0]
            q_lo = np.take_along_axis(A, (k_lo - 1)[..., None], axis=-1)[..., 0]
            q[:, i, :, 2 * a], q[:, i, :, 2 * a + 1] = q_lo, q_hi
            lo_v[a, :, i, :] = to_orders(F_test[:, i, :], q_lo)
            hi_v[a, :, i, :] = to_orders(F_test[:, i, :], q_hi)
    return q, trace


def oracle_quantiles(mu, sigma, test_org, h, levels=LEVELS):
    """Wahre Quantile des log-normalen Rauschens: mu exp(σ z - σ²/2). mu, sigma: (n, T)."""
    days = test_org[:, None] + np.arange(h)[None, :]
    m, s = mu[:, days], sigma[:, days]
    return np.stack([m * np.exp(s * z(tau) - 0.5 * s ** 2) for tau in levels], axis=-1)

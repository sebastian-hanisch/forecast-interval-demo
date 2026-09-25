"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. bf_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import fi_constants as C


def _point(value):
    v = str(value).strip().lower()
    if v not in C.POINT_MODELS:
        raise ValueError(value)
    return v


def _method(value):
    v = str(value).strip().lower()
    if v not in C.METHODS:
        raise ValueError(value)
    return v


def _nominal(value):
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(value)
    return min(C.NOMINALS, key=lambda p: abs(p - v))


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "depots_slider": SettingSpec("depots", int, C.DEFAULT_DEPOTS, C.DEPOTS_MIN, C.DEPOTS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "point_select": SettingSpec("point", _point, "hw_mult"),
    "shift_slider": SettingSpec("shift", float, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "outlier_slider": SettingSpec("outliers", float, C.DEFAULT_OUTLIER, C.OUTLIER_MIN, C.OUTLIER_MAX),
    "window_slider": SettingSpec("window", int, C.DEFAULT_WINDOW, C.WINDOW_MIN, C.WINDOW_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "nominal_select": SettingSpec("level", _nominal, C.DEFAULT_NOMINAL),
    "show_select": SettingSpec("show", _method, "conformal"),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_depots": "depots_slider", "noise": "noise_slider", "trend": "trend_slider", "events": "events_slider", "horizon": "horizon_slider", "point": "point_select", "shift": "shift_slider",
               "outliers": "outlier_slider", "window": "window_slider", "gamma": "gamma_slider", "nominal": "nominal_select", "show": "show_select", "seed": "seed_input"}
STEPS = {"depots_slider": C.DEPOTS_STEP, "noise_slider": C.NOISE_STEP, "trend_slider": C.TREND_STEP, "events_slider": C.EVENTS_STEP, "shift_slider": C.SHIFT_STEP, "outlier_slider": C.OUTLIER_STEP,
         "window_slider": C.WINDOW_STEP, "gamma_slider": C.GAMMA_STEP}


def _p(**kw):
    base = {"n_depots": C.DEFAULT_DEPOTS, "noise": C.DEFAULT_NOISE, "trend": C.DEFAULT_TREND, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON, "point": "hw_mult", "shift": C.DEFAULT_SHIFT,
            "outliers": C.DEFAULT_OUTLIER, "window": C.DEFAULT_WINDOW, "gamma": C.DEFAULT_GAMMA, "nominal": C.DEFAULT_NOMINAL, "show": "conformal", "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: Holt-Winters, 30 Depots": _p(),
    "Regression als Punktmodell": _p(point="regression"),
    "Verteilungswechsel (Rauschen ×2)": _p(shift=2.0, show="aci"),
    "Ausreißer (5 %)": _p(outliers=0.05, nominal=0.5, show="gauss_log"),
    "Kleines Kalibrierfenster (30)": _p(window=30, nominal=0.95),
    "Langer Horizont (28 Tage)": _p(horizon=28),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: Holt-Winters, 30 Depots": "Bei Nennabdeckung 80 % (Seed 3) deckt Gauß roh 75,7 %, Gauß im Log 78,0 %, empirisch 76,5 %, konform 79,4 %, ACI 79,9 % (Orakel 79,7 %); je Wochentag schwankt Gauß roh zwischen 62,9 und 97,8 %.",
    "Regression als Punktmodell": "Punktprognose aus der Regression (MASE 0,78 statt 0,87): die Abdeckung ist bei allen Horizonten gleich; konform 80,0 %, Gauß im Log 75,2 %; der Intervall-Score der Konformen (3,25) liegt nahe am Orakel (3,15).",
    "Verteilungswechsel (Rauschen ×2)": "Das Rauschen verdoppelt sich ab Tag 900: nach dem Wechsel decken bei Nennabdeckung 80 % Gauß im Log 50,9 %, empirisch 49,6 %, konform 73,3 % und ACI 80,0 %.",
    "Ausreißer (5 %)": "5 % Ausreißertage, Nennabdeckung 50 %: Gauß roh deckt 67,0 %, Gauß im Log 60,1 % - die Standardabweichung ist aufgebläht; empirisch 46,5 %, konform 49,0 %, ACI 49,9 %.",
    "Kleines Kalibrierfenster (30)": "Fenster von 30 Ursprüngen bei Nennabdeckung 95 %: konform und ACI erreichen nur 91,7 % (das Quantil bleibt am Rand des Fensters, ACI kann es nicht ausgleichen); Gauß im Log 93,1 %, empirisch 93,3 %.",
    "Langer Horizont (28 Tage)": "Horizont 28, Nennabdeckung 80 %: von Tag 1 zu Tag 28 fällt die Abdeckung bei Gauß im Log von 79,8 auf 72,8 %, bei konform von 80,4 auf 74,1 %; ACI hält 79,1 %.",
}

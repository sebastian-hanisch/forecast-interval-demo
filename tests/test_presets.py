"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import fi_constants as C
import fi_evaluation as E
import fi_presets as P


def _settings(p):
    return E.Settings(p["n_depots"], p["noise"], p["events"], p["trend"], p["horizon"], p["shift"], p["outliers"], p["point"], p["window"], p["gamma"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_depots", "depots_slider"), ("noise", "noise_slider"), ("trend", "trend_slider"), ("events", "events_slider"), ("shift", "shift_slider"), ("outliers", "outlier_slider"),
                               ("window", "window_slider"), ("gamma", "gamma_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        assert p["point"] in C.POINT_MODELS and p["show"] in C.METHODS and p["nominal"] in C.NOMINALS


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: Holt-Winters, 30 Depots"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("window_slider") == (C.WINDOW_MIN, C.WINDOW_MAX) and P.bounds("shift_slider") == (C.SHIFT_MIN, C.SHIFT_MAX)
    assert set(P.STEPS) == {"depots_slider", "noise_slider", "trend_slider", "events_slider", "shift_slider", "outlier_slider", "window_slider", "gamma_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_casters_reject_bad_values_and_snap_nominals():
    for caster, bad in ((P._point, "arima"), (P._method, "quatsch"), (P._nominal, "nan")):
        try:
            caster(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert P._point(" REGRESSION ") == "regression" and P._nominal("0.9") == 0.95 and P._nominal("0.6") == 0.5 and P._method("ACI") == "aci"

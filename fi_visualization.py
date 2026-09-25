"""Plotly-Abbildungen der Intervall-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import fi_constants as C
import fi_evaluation as E
import fi_intervals as I

COLORS = {"gauss_raw": "#d62728", "gauss_log": "#e6550d", "empirical": "#8c6bb1", "conformal": "#00897b", "aci": "#1f77b4", "oracle": "#54a24b"}
NOMINAL_COLORS = {0.5: "#9ecae1", 0.8: "#4292c6", 0.95: "#08519c"}
ACTUAL = "#14233B"
WARN = "#f58518"
SHORT = {**C.METHOD_SHORT, "oracle": "Orakel"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_series(a, dep, origin, method):
    """42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Punktprognose und die drei Intervalle des gewählten Verfahrens."""
    port, h = a.port, a.settings.horizon
    i = int(origin - a.test_org[0])
    x_hist, x_fut = np.arange(origin - 42, origin), np.arange(origin, origin + h)
    fig = go.Figure()
    q = a.quantiles[method][dep, i]                                                                # (h, 6)
    for p in C.NOMINALS[::-1]:
        lo, hi = I.nominal_levels(p)
        fig.add_trace(go.Scatter(x=np.concatenate([x_fut, x_fut[::-1]]), y=np.concatenate([q[:, I.LEVEL_INDEX[hi]], q[::-1, I.LEVEL_INDEX[lo]]]), fill="toself",
                                 fillcolor=_rgba(COLORS[method], {0.5: 0.35, 0.8: 0.22, 0.95: 0.12}[p]), line=dict(width=0), name=f"{int(round(p * 100))} %-Intervall", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_hist, y=port.y[dep, origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=port.y[dep, origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=1), marker=dict(size=7, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=a.forecast[dep, i], mode="lines", name="Punktprognose", line=dict(color=COLORS[method], width=2.5)))
    fig.add_trace(go.Scatter(x=x_fut, y=a.quantiles["oracle"][dep, i, :, I.LEVEL_INDEX[0.1]], mode="lines", name="Orakel 80 %", line=dict(color=COLORS["oracle"], width=1.4, dash="dot"), legendgroup="o"))
    fig.add_trace(go.Scatter(x=x_fut, y=a.quantiles["oracle"][dep, i, :, I.LEVEL_INDEX[0.9]], mode="lines", showlegend=False, line=dict(color=COLORS["oracle"], width=1.4, dash="dot"), legendgroup="o"))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_methods_origin(a, dep, origin, p):
    """Das Intervall mit Nennabdeckung p aller Verfahren an einem Ursprung (Ober- und Untergrenze) neben dem Ist."""
    port, h = a.port, a.settings.horizon
    i = int(origin - a.test_org[0])
    x = np.arange(origin, origin + h)
    lo, hi = I.nominal_levels(p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=port.y[dep, origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=1), marker=dict(size=7, symbol="circle-open")))
    for m in C.METHODS + ("oracle",):
        col = COLORS[m]
        q = a.quantiles[m][dep, i]
        fig.add_trace(go.Scatter(x=x, y=q[:, I.LEVEL_INDEX[hi]], mode="lines", name=SHORT[m], legendgroup=m, line=dict(color=col, width=1.6, dash="dot" if m == "oracle" else "solid")))
        fig.add_trace(go.Scatter(x=x, y=q[:, I.LEVEL_INDEX[lo]], mode="lines", showlegend=False, legendgroup=m, line=dict(color=col, width=1.6, dash="dot" if m == "oracle" else "solid")))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text=f"Grenzen des {int(round(p * 100))} %-Intervalls", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.35))


def build_deviation(a):
    """Abweichung der beobachteten Abdeckung von der Nennabdeckung (Prozentpunkte), je Verfahren und Nennabdeckung."""
    fig = go.Figure()
    for p in C.NOMINALS:
        vals = [100 * (a.summary[m]["coverage"][p] - p) for m in C.METHODS]
        fig.add_trace(go.Bar(x=[SHORT[m] for m in C.METHODS], y=vals, name=f"{int(round(p * 100))} %", marker=dict(color=NOMINAL_COLORS[p]), text=[f"{v:+.1f}".replace(".", ",") for v in vals], textposition="outside"))
    fig.add_hline(y=0, line=dict(color="#7f7f7f", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="beobachtet minus Nennabdeckung (Prozentpunkte)", zeroline=False)
    return _base(fig, 340)


def build_score_bars(a, p):
    """Intervall-Score (Winkler, skaliert) und Breite der Verfahren bei Nennabdeckung p."""
    ms = sorted(C.METHODS, key=lambda m: a.summary[m]["iscore"][p]) + ["oracle"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[SHORT[m] for m in ms], y=[a.summary[m]["iscore"][p] for m in ms], marker=dict(color=[COLORS[m] for m in ms]), text=[de(a.summary[m]["iscore"][p]) for m in ms], textposition="outside", showlegend=False))
    fig.update_yaxes(title_text=f"Intervall-Score bei {int(round(p * 100))} % (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 320)


def build_weekday(a, p):
    cov = E.coverage_by_weekday(a, p)
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=list(C.WEEKDAYS), y=100 * cov[m], mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5 if m in ("gauss_raw", "conformal") else 1.5), marker=dict(size=6)))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Nennabdeckung", annotation_position="top left")
    fig.update_xaxes(title_text="Wochentag des Zieltags")
    fig.update_yaxes(title_text="beobachtete Abdeckung (%)")
    return _base(fig, 320)


def build_horizon(a, p):
    cov = E.coverage_by_horizon(a, p)
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=100 * cov[m], mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5 if m in ("gauss_log", "conformal") else 1.5), marker=dict(size=5)))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Nennabdeckung", annotation_position="top left")
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="beobachtete Abdeckung (%)")
    return _base(fig, 320)


def _rolling(v, w=21):
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])])
    return np.convolve(pad, k, mode="valid")[:len(v)]


def build_time(a, p):
    """Abdeckung je Zieltag im Testjahr (21-Tage-Mittel); senkrecht der Verteilungswechsel."""
    cov = E.coverage_by_target_day(a, p)
    lo_i, hi_i = a.settings.horizon - 1, C.N_DAYS - C.FIRST_TEST - (a.settings.horizon - 1)                # nur Zieltage, an denen alle Horizonte beitragen
    x = np.arange(C.FIRST_TEST, C.N_DAYS)[lo_i:hi_i]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=x, y=100 * _rolling(cov[m][lo_i:hi_i]), mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=2.5 if m in ("gauss_log", "aci") else 1.5)))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"))
    fig.add_vline(x=C.SHIFT_DAY, line=dict(color=WARN, dash="dot"), annotation_text=f"Rauschfaktor ×{a.settings.shift:g}".replace(".", ",") if a.settings.shift > 1 else "Stelle des Verteilungswechsels", annotation_position="top left")
    fig.update_xaxes(title_text="Zieltag")
    fig.update_yaxes(title_text="beobachtete Abdeckung (%, 21-Tage-Mittel)")
    return _base(fig, 320)


def build_depot_strip(a, p):
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Box(y=100 * a.summary[m]["cov_depot"][p], name=SHORT[m], marker=dict(color=COLORS[m]), boxpoints="all", jitter=0.4, pointpos=0, showlegend=False))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"))
    fig.update_yaxes(title_text="Abdeckung je Depot (%)")
    return _base(fig, 320)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def build_shift(rows, p=C.DEFAULT_NOMINAL):
    xs = [f"×{r['shift']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[100 * r["coverage"][m][0] for r in rows], error_y=dict(type="data", array=[100 * r["coverage"][m][1] for r in rows]), mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5)))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Nennabdeckung", annotation_position="top left")
    fig.update_xaxes(title_text="Faktor auf die Streuung ab dem Verteilungswechsel", type="category")
    fig.update_yaxes(title_text=f"Abdeckung nach dem Wechsel (%, Nenn {int(round(p * 100))} %)")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3))


def build_outliers(rows, p=0.5):
    xs = [f"{100 * r['outliers']:g} %".replace(".", ",") for r in rows]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[100 * r["coverage"][m][0] for r in rows], error_y=dict(type="data", array=[100 * r["coverage"][m][1] for r in rows]), mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.5)))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Nennabdeckung", annotation_position="top left")
    fig.update_xaxes(title_text="Anteil der Ausreißertage", type="category")
    fig.update_yaxes(title_text=f"beobachtete Abdeckung (%, Nenn {int(round(p * 100))} %)")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.3))


def build_window(rows, p=0.95):
    xs = [f"{r['window']} Tage" for r in rows]
    fig = go.Figure()
    for m in ("conformal", "aci"):
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[m]["coverage"][0] for r in rows], error_y=dict(type="data", array=[100 * r[m]["coverage"][1] for r in rows]), mode="lines+markers", name=SHORT[m] + ": Abdeckung",
                                 line=dict(color=COLORS[m], width=2.5)))
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[m]["sd_depot"][0] for r in rows], mode="lines+markers", name=SHORT[m] + ": Streuung zwischen Depots", yaxis="y2", line=dict(color=COLORS[m], width=1.5, dash="dot")))
    fig.add_hline(y=100 * p, line=dict(color="#7f7f7f", dash="dash"), annotation_text="Nennabdeckung", annotation_position="bottom left")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Streuung der Depot-Abdeckung (Prozentpunkte)", rangemode="tozero", fixedrange=True))
    fig.update_xaxes(title_text="Kalibrierfenster", type="category")
    fig.update_yaxes(title_text=f"beobachtete Abdeckung (%, Nenn {int(round(p * 100))} %)")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.35))

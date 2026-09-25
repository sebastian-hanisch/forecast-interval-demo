"""Prognoseintervalle - wie sicher ist die Prognose? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: aus einer Punktprognose ein Intervall machen - mit Normalverteilung, mit den Fehlern der Vergangenheit und mit konformer Kalibrierung
(fest und adaptiv) - und prüfen, ob die Intervalle halten, was ihre Nennabdeckung verspricht.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import fi_constants as C
from fi_evaluation import Settings, analyse, coverage_by_horizon, coverage_by_weekday, outlier_experiment, shift_experiment, window_experiment
from fi_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from fi_visualization import SHORT, build_deviation, build_depot_strip, build_horizon, build_methods_origin, build_outliers, build_score_bars, build_series, build_shift, build_time, build_weekday, build_window

st.set_page_config(page_title="Prognoseintervalle – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=1):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _shift(levels, seeds):
    return shift_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _outliers(levels, seeds):
    return outlier_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _window(levels, seeds):
    return window_experiment(levels, seeds)


st.title("📏 Prognoseintervalle – wie sicher ist die Prognose?")
st.markdown(
    """
Eine Punktprognose sagt "morgen 120 Aufträge" - und schweigt darüber, ob 100 oder 160 noch normal sind. Für Personal- und Bestandsplanung zählt aber genau das: **wie weit darf der Ist-Wert abweichen?** Ein **Prognoseintervall** mit der Nennabdeckung 80 % verspricht, dass vier von fünf Ist-Werten hineinfallen.
Die Demo nimmt die Punktprognosen der Vorgänger (Holt-Winters, Regression, Wochenmittel) auf einem **Portfolio von Depots** und macht daraus Intervalle auf **fünf Arten**: mit der **Normalverteilung** (roh und im Log), mit den **Fehlern der Trainingstage** und mit **konformer Kalibrierung** an den zuletzt realisierten Fehlern (fest und adaptiv).
Geprüft wird, was zählt: **halten die Intervalle ihr Versprechen** - im Mittel, je Wochentag, je Horizont, im Zeitverlauf und je Depot? Und was kostet eine ehrliche Abdeckung an Breite? Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Siebtes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe. **Bezug zu OR:** Sicherheitsbestände, Personalreserven und Kapazitätspuffer folgen aus Quantilen der Prognose, nicht aus dem Punktwert; "
    "das Bestands-Stück der Linie nimmt diese Quantile auf."
)

with st.expander("So entstehen die Intervalle", expanded=True):
    st.markdown(
        """
1. **Fehlermaß.** Der Fehler einer Prognose $F$ zum Ist $y$ ist das **Log-Verhältnis** $s = \\log\\frac{y+1}{F+1}$. Die Aufträge schwanken multiplikativ (sonntags 0,35-fach, freitags 1,2-fach): im Log ist der Fehler an allen Wochentagen gleich groß, roh nicht.
   Ein Quantil $q$ dieser Fehler gibt das Prognosequantil $(F+1)\\,e^{q} - 1$.
2. **Gauß, roh.** $F \\pm z\\,\\sigma$ mit der Standardabweichung der Ein-Schritt-Fehler $y - F$ auf den Trainingstagen: eine Breite für alle Tage.
3. **Gauß im Log.** $(F+1)\\,e^{z\\sigma_{\\log}} - 1$ mit der Streuung der Ein-Schritt-Log-Fehler: die Breite folgt dem Niveau.
4. **Empirisch.** Die Quantile der Ein-Schritt-Log-Fehler auf den Trainingstagen selbst - keine Normalverteilung, aber wieder nur Ein-Schritt-Fehler und die Trainingstage.
5. **Konform.** Die Quantile der zuletzt **realisierten** Fehler **desselben Horizonts** (Kalibrierfenster der letzten $W$ Ursprünge, nur Prognosen, deren Zieltag schon bekannt ist), mit der Endlichkeitskorrektur $\\lceil (m+1)\\tau \\rceil$: bei austauschbaren Fehlern deckt das Intervall mindestens die Nennabdeckung.
6. **Adaptiv konform (ACI).** Wie konform, aber die Fehlerrate $\\alpha_k$ je Horizont wird nach jeder realisierten Prognose um $\\gamma\\,(\\alpha - \\text{Fehlschlag})$ nachgeführt: nach zu vielen Fehlschlägen werden die Intervalle breiter.
7. **Orakel.** Die wahren Quantile des Rauschens - das ist die Messlatte, die keine Methode kennt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Portfolio**")
    n_depots = st.slider("Zahl der Depots", *bounds("depots_slider"), key="depots_slider", step=C.DEPOTS_STEP, help="Wie viele Depots das Portfolio hat.")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Rauschens; die Depots streuen um diesen Wert.")
    trend = st.slider("Trend (% je Jahr, Mittel)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Mittleres Wachstum der Depots in Prozent des Ausgangsniveaus je Jahr.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Feiertags- und Aktionseffekte.")
    shift = st.slider("Verteilungswechsel (Faktor auf das Rauschen ab Tag 900)", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="Ab Tag 900 (im Testjahr) wird das Rauschen aller Depots mit diesem Faktor multipliziert; 1 = kein Wechsel.")
    outliers = st.slider("Ausreißertage (Anteil)", *bounds("outlier_slider"), key="outlier_slider", step=C.OUTLIER_STEP, help="Anteil der Tage (im ganzen Zeitraum) mit einem zusätzlichen log-normalen Sprung (Streuung 0,7).", format="%.2f")
    st.markdown("**Punktmodell und Horizont**")
    point = st.selectbox("Punktprognose", list(C.POINT_MODELS), key="point_select", format_func=lambda k: C.POINT_NAMES[k], help="Aus welcher Punktprognose die Intervalle gebaut werden; die Parameter stammen aus den Tagen vor Tag 610.")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    st.markdown("**Konforme Verfahren**")
    window = st.slider("Kalibrierfenster (Ursprünge)", *bounds("window_slider"), key="window_slider", step=C.WINDOW_STEP, help="Wie viele der zuletzt realisierten Prognosen je Horizont die Fehlerquantile bestimmen (höchstens 120: die Tage 610 bis 729).")
    gamma = st.slider("ACI-Schrittweite γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.3f", help="Wie stark die Fehlerrate nach jeder realisierten Prognose nachgeführt wird; 0 = kein Nachführen (dann wie konform).")
    st.markdown("**Anzeige**")
    nominal = st.selectbox("Nennabdeckung der Diagramme", list(C.NOMINALS), key="nominal_select", format_func=lambda p: f"{int(round(p * 100))} %", help="Für welche Nennabdeckung die Aufschlüsselungen und der Vergleich der Intervalle gezeigt werden.")
    show = st.selectbox("Verfahren im Depot-Diagramm", list(C.METHODS), key="show_select", format_func=lambda m: C.METHOD_NAMES[m], help="Welches Verfahren im Diagramm des einzelnen Depots als Fächer gezeigt wird.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Portfolio fest.")
    st.button("🎲 Neues Portfolio generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"depots_slider": int(n_depots), "noise_slider": round(float(noise), 2), "trend_slider": int(trend), "events_slider": round(float(events), 2), "horizon_slider": int(horizon), "point_select": point,
                   "shift_slider": round(float(shift), 2), "outlier_slider": round(float(outliers), 2), "window_slider": int(window), "gamma_slider": round(float(gamma), 3), "nominal_select": nominal,
                   "show_select": show, "seed_input": int(seed)})

settings = Settings(int(n_depots), round(float(noise), 2), round(float(events), 2), int(trend), int(horizon), round(float(shift), 2), round(float(outliers), 2), point, int(window), round(float(gamma), 3), int(seed))
with st.spinner("Die Punktprognosen und Intervalle werden berechnet ..."):
    a = analyse(settings)
port, sm = a.port, a.summary
pn = int(round(nominal * 100))

# --- Ein Depot -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Depot und die Intervalle an einem Ursprung")
st.session_state["depot_slider"] = min(port.n - 1, max(0, st.session_state.get("depot_slider", 0)))
dep = int(st.slider("Depot", 0, port.n - 1, key="depot_slider", help="Welches Depot des Portfolios gezeigt wird."))
lo_o, hi_o = int(a.test_org[0]), int(a.test_org[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, dep, origin, show), width="stretch", key="series_chart")
st.caption(
    f"Depot {dep}: Niveau {de(port.level[dep], 0)} Aufträge, Rauschen {de(port.noise[dep], 2)}. Der Fächer zeigt die Intervalle des Verfahrens **{SHORT[show]}** mit Nennabdeckung 50, 80 und 95 % für die nächsten {settings.horizon} Tage "
    f"(Punktprognose: {C.POINT_SHORT[point]}); grün gepunktet die wahren 10- und 90-%-Quantile (Orakel)."
)
st.plotly_chart(build_methods_origin(a, dep, origin, nominal), width="stretch", key="methods_chart")
st.caption(f"Die {pn} %-Intervalle aller Verfahren am selben Ursprung. Sie unterscheiden sich in Breite und Lage; ob eines davon 'richtig' ist, zeigt erst die Abdeckung über alle {len(a.test_org)} Ursprünge, {port.n} Depots und {settings.horizon} Horizonte.")

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Halten die Intervalle ihr Versprechen?")
mets = C.METHODS
cols = st.columns(len(mets) + 1)
for col, m in zip(cols, list(mets) + ["oracle"]):
    dev = sm[m]["coverage"][nominal] - nominal
    col.metric(SHORT[m], pct(sm[m]["coverage"][nominal]), delta=f"{de(100 * dev, 1)} Punkte", delta_color="off", help=f"Beobachtete Abdeckung des {pn} %-Intervalls über alle Ursprünge, Depots und Horizonte des Testjahres.")
st.plotly_chart(build_deviation(a), width="stretch", key="deviation_chart")
rows = []
for m in list(mets) + ["oracle"]:
    r = {"Verfahren": "Orakel (wahre Quantile)" if m == "oracle" else C.METHOD_NAMES[m]}
    for p in C.NOMINALS:
        r[f"Abdeckung {int(round(p * 100))} %"] = pct(sm[m]["coverage"][p])
    r[f"Breite {pn} %"] = de(sm[m]["width"][nominal], 2)
    r[f"Intervall-Score {pn} %"] = de(sm[m]["iscore"][nominal], 2)
    r["Pinball"] = de(sm[m]["pinball"], 3)
    rows.append(r)
st.dataframe(rows, hide_index=True)
worst = max(mets, key=lambda m: abs(sm[m]["coverage"][nominal] - nominal))
best_is = min(mets, key=lambda m: sm[m]["iscore"][nominal])
dev_w = sm[worst]["coverage"][nominal] - nominal
dev_c = sm["conformal"]["coverage"][nominal] - nominal
if abs(dev_c) <= 0.02 and abs(dev_w) <= 0.02:
    st.success(f"✅ Bei Nennabdeckung {pn} % liegen alle Verfahren innerhalb von 2 Prozentpunkten (am weitesten weg: {SHORT[worst]} mit {pct(sm[worst]['coverage'][nominal])}).")
elif abs(dev_c) <= 0.02:
    st.info(f"Bei Nennabdeckung {pn} % trifft das konforme Verfahren ({pct(sm['conformal']['coverage'][nominal])}); am weitesten daneben liegt **{SHORT[worst]}** mit {pct(sm[worst]['coverage'][nominal])} ({'zu schmal' if dev_w < 0 else 'zu breit'}). "
            f"Der beste Intervall-Score ({de(sm[best_is]['iscore'][nominal], 2)}) gehört {SHORT[best_is]}.")
else:
    st.warning(f"⚠️ Selbst das konforme Verfahren verfehlt hier die Nennabdeckung {pn} % ({pct(sm['conformal']['coverage'][nominal])}); am weitesten daneben liegt {SHORT[worst]} mit {pct(sm[worst]['coverage'][nominal])}. "
               "Mögliche Gründe: ein zu kleines Kalibrierfenster für die gewünschte Stufe, ein Verteilungswechsel im Testjahr oder ein langer Horizont.")
st.caption(
    f"Punktprognose: {C.POINT_NAMES[point]} (MASE {de(a.point_mase, 2)}). Breite und Intervall-Score sind durch den saisonal naiven Trainingsfehler des Depots geteilt und über die Depots gemittelt; der Intervall-Score (Winkler) addiert zur Breite "
    f"2/α mal die Überschreitung außerhalb des Intervalls (α = 1 − Nennabdeckung); Pinball ist der mittlere Quantilverlust über die Stufen 2,5 / 10 / 25 / 75 / 90 / 97,5 %. {len(a.test_org)} Ursprünge, {port.n} Depots."
)
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown(f"##### Intervall-Score bei {pn} %")
    st.plotly_chart(build_score_bars(a, nominal), width="stretch", key="score_chart")
with sc2:
    st.markdown(f"##### Abdeckung je Depot bei {pn} %")
    st.plotly_chart(build_depot_strip(a, nominal), width="stretch", key="strip_chart")

st.markdown("---")

st.markdown("## 🎯 Wo die Abdeckung bricht")
w1, w2 = st.columns(2)
with w1:
    st.markdown(f"##### Je Wochentag des Zieltags ({pn} %)")
    st.plotly_chart(build_weekday(a, nominal), width="stretch", key="weekday_chart")
with w2:
    st.markdown(f"##### Je Prognosehorizont ({pn} %)")
    st.plotly_chart(build_horizon(a, nominal), width="stretch", key="horizon_chart")
st.markdown(f"##### Im Zeitverlauf des Testjahres ({pn} %)")
st.plotly_chart(build_time(a, nominal), width="stretch", key="time_chart")
wd = coverage_by_weekday(a, nominal)
hz = coverage_by_horizon(a, nominal)
wd_span = {m: (float(wd[m].min()), float(wd[m].max())) for m in mets}
st.caption(
    f"Je Wochentag: Gauß roh deckt zwischen {pct(wd_span['gauss_raw'][0])} und {pct(wd_span['gauss_raw'][1])}, Gauß im Log zwischen {pct(wd_span['gauss_log'][0])} und {pct(wd_span['gauss_log'][1])}, konform zwischen "
    f"{pct(wd_span['conformal'][0])} und {pct(wd_span['conformal'][1])} (Nennabdeckung {pn} %). Je Horizont: von Tag 1 zu Tag {settings.horizon} ändert sich die Abdeckung bei Gauß im Log von {pct(hz['gauss_log'][0])} auf {pct(hz['gauss_log'][-1])}, "
    f"bei konform von {pct(hz['conformal'][0])} auf {pct(hz['conformal'][-1])}. Die gepunktete Linie im Zeitverlauf markiert Tag {C.SHIFT_DAY}, ab dem der Verteilungswechsel gilt (gezeigt sind nur Zieltage, an denen alle Horizonte beitragen)."
)

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Verteilungswechsel: das Rauschen wächst mitten im Testjahr")
st.caption(f"{C.EXP_DEPOTS} Depots, Holt-Winters, Standardeinstellungen; ab Tag {C.SHIFT_DAY} wächst das Rauschen um den Faktor {', '.join(f'{f:g}'.replace('.', ',') for f in C.SHIFT_LEVELS)}. Gezeigt wird die Abdeckung der Zieltage danach bei Nennabdeckung 80 %. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa eine halbe Minute.")
if st.button("Verteilungswechsel durchrechnen", key="shift_start"):
    st.session_state["shift_on"] = True
if st.session_state.get("shift_on"):
    rs = _shift(C.SHIFT_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_shift(rs), width="stretch", key="shift_chart")
    r0, r1 = rs[0]["coverage"], rs[-1]["coverage"]
    f1 = rs[-1]["shift"]
    st.warning(
        f"**Befund:** Ohne Wechsel decken alle nahe der Nennabdeckung 80 % (Gauß im Log {pct(r0['gauss_log'][0])}, empirisch {pct(r0['empirical'][0])}, konform {pct(r0['conformal'][0])}, ACI {pct(r0['aci'][0])}). "
        f"Bei Faktor {de(f1, 1)} fallen die Verfahren, die nur die Trainingstage kennen, auf {pct(r1['gauss_log'][0])} (Gauß im Log) und {pct(r1['empirical'][0])} (empirisch); das konforme Verfahren mit {C.DEFAULT_WINDOW} Tagen Fenster auf {pct(r1['conformal'][0])}, "
        f"weil sein Fenster den Wechsel erst nach und nach aufnimmt. Das **adaptive** Verfahren hält {pct(r1['aci'][0])}: die Rückmeldung der Fehlschläge macht die Intervalle rechtzeitig breiter. **Die Abdeckungsgarantie der Konformen gilt nur für austauschbare Fehler** - ein Wechsel bricht sie."
    )

st.markdown("---")

st.subheader("🔬 Ausreißer: die Standardabweichung wird aufgebläht")
st.caption(f"{C.EXP_DEPOTS} Depots, Holt-Winters; ein Anteil {', '.join(f'{100 * o:g}'.replace('.', ',') for o in C.OUTLIER_LEVELS)} % der Tage (im ganzen Zeitraum, auch im Training) bekommt einen zusätzlichen log-normalen Sprung. Gezeigt wird die Abdeckung des 50-%-Intervalls; "
           f"Breite relativ zum Orakel (dessen Rauschen die Ausreißer nicht kennt). Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Ausreißer durchrechnen", key="outlier_start"):
    st.session_state["outlier_on"] = True
if st.session_state.get("outlier_on"):
    ro = _outliers(C.OUTLIER_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_outliers(ro), width="stretch", key="outlier_chart")
    o1 = ro[-1]
    st.warning(
        f"**Befund:** Bei {de(100 * o1['outliers'], 0)} % Ausreißertagen überdeckt das **Gauß-roh-Intervall** mit {pct(o1['coverage']['gauss_raw'][0])} statt 50 % und ist {de(o1['width']['gauss_raw'][0], 2)}-mal so breit wie das des Orakels; Gauß im Log {pct(o1['coverage']['gauss_log'][0])}. "
        f"Empirisch ({pct(o1['coverage']['empirical'][0])}) und konform ({pct(o1['coverage']['conformal'][0])}) bleiben bei den 50 %, bei Breiten von {de(o1['width']['empirical'][0], 2)} und {de(o1['width']['conformal'][0], 2)}: Quantile sind gegen Ausreißer robust, die Standardabweichung nicht."
    )

st.markdown("---")

st.subheader("🔬 Wie groß muss das Kalibrierfenster sein?")
st.caption(f"{C.EXP_DEPOTS} Depots, Holt-Winters; Fenster von {', '.join(str(w) for w in C.WINDOW_LEVELS)} Tagen, Nennabdeckung 95 %. Gezeigt: die beobachtete Abdeckung und die Streuung der Abdeckung zwischen den Depots. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine halbe Minute.")
if st.button("Kalibrierfenster durchrechnen", key="window_start"):
    st.session_state["window_on"] = True
if st.session_state.get("window_on"):
    rw = _window(C.WINDOW_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_window(rw), width="stretch", key="window_chart")
    w_lo, w_hi = rw[0], rw[-1]
    st.warning(
        f"**Befund:** Mit {w_lo['window']} Tagen reicht das Fenster für 95 % nicht: die Korrektur verlangt den 31. von 30 Werten, das Quantil bleibt am Rand des Fensters (die Garantie reicht mit 30 Werten höchstens bis 93,5 %), und die Abdeckung liegt bei {pct(w_lo['conformal']['coverage'][0])} (ACI kann das nicht ausgleichen: {pct(w_lo['aci']['coverage'][0])}). "
        f"Mit {w_hi['window']} Tagen sind es {pct(w_hi['conformal']['coverage'][0])} (konform) und {pct(w_hi['aci']['coverage'][0])} (ACI); die Abdeckung schwankt zwischen den Depots noch um {de(100 * w_hi['conformal']['sd_depot'][0], 1)} Prozentpunkte (ACI: {de(100 * w_hi['aci']['sd_depot'][0], 1)}). "
        "Ein größeres Fenster macht die Quantile ruhiger, folgt aber Änderungen langsamer."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Fehler sind normalverteilt und gleich breit** | Roh gerechnet sind die Intervalle sonntags viel zu breit und freitags zu schmal (multiplikatives Rauschen); Ausreißer blähen die Standardabweichung auf. | Rechnen im Log, Quantile statt Standardabweichung |
| **Trainingsfehler sind Prognosefehler** | Ein-Schritt-Fehler auf den Trainingstagen unterschätzen die Fehler bei längerem Horizont und außerhalb der Stichprobe. | Kalibrierung auf realisierten Fehlern je Horizont (konform) |
| **Die Fehler sind austauschbar** | Die Garantie der konformen Verfahren beruht darauf. Ein Verteilungswechsel bricht sie; ACI gleicht ihn aus, braucht dafür aber zeitnahe Rückmeldung und wird nach Ausreißern breit. | ACI mit passender Schrittweite, kürzere Fenster |
| **Genug Kalibrierpunkte** | Für 95 % braucht es mindestens 39 Werte im Fenster; darunter bleibt das Quantil am Rand. | Größeres Fenster, Stufen zusammenlegen |
| **Abdeckung im Mittel genügt** | Die Garantie gilt im Mittel über Wochentage, Horizonte und Depots, nicht je Wochentag oder je Depot. | Konditionale Konforme (nach Wochentag oder Depot getrennt), Normierung |
| **Punktprognose und Intervall getrennt** | Intervalle folgen aus Fehlern eines Punktmodells; eine gute Punktprognose macht sie schmaler, ersetzt aber die Kalibrierung nicht. | Quantilverlust im Modell (Boosting mit Pinball-Verlust; nicht gebaut) |
| **Erzeugte Portfolios, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, **Prognoseintervalle**, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Fehler.** $s = \log\frac{y+1}{F+1}$; Prognosequantil zur Stufe $\tau$: $Q_\tau = \max\big((F+1)\,e^{q_\tau} - 1,\ 0\big)$. Ein zentrales Intervall mit Nennabdeckung $p$ nimmt die Stufen $\tau_\ell = (1-p)/2$ und $\tau_u = 1 - \tau_\ell$.

**Gauß, roh:** $Q_\tau = \max(F + z_\tau \sigma, 0)$, $\sigma$ = Standardabweichung der Ein-Schritt-Fehler $y - F$ auf den Tagen 91 bis 609. **Gauß im Log:** $q_\tau = z_\tau \sigma_{\log}$. **Empirisch:** $q_\tau$ = das $\tau$-Quantil der Ein-Schritt-Log-Fehler.

**Konform** (Split-Konforme, Vovk et al. 2005; Angelopoulos/Bates 2023). Zu Ursprung $t$ und Horizont $k$ sind die Scores $s_{o,k}$ der Ursprünge $o \in (t-k-W,\ t-k]$ bekannt (Zieltag $o + k - 1 \le t - 1$). Mit $m$ Scores, sortiert $s_{(1)} \le \dots \le s_{(m)}$:
obere Stufe $q = s_{(\min(\lceil (m+1)\tau \rceil,\ m))}$, untere Stufe $q = s_{(\max(\lfloor (m+1)\tau \rfloor,\ 1))}$. Bei austauschbaren Scores gilt $\Pr(y \in [s_{(a)}, s_{(b)}]) = (b - a)/(m+1)$, mit den Stufen oben also mindestens die Nennabdeckung.

**ACI** (Gibbs/Candès 2021). Je Horizont $k$ eine Fehlerrate $\alpha_k$, Start $\alpha = 1 - p$; nach jeder realisierten Prognose (Ursprung $t-k$, Zieltag $t-1$): $\alpha_k \leftarrow \alpha_k + \gamma\,(\alpha - \mathbb 1[y \notin I])$; das Intervall des nächsten Ursprungs nimmt die Stufen $\alpha_k/2$ und $1 - \alpha_k/2$ (auf $[0,1]$ begrenzt) aus dem Fenster.

**Kennzahlen.** Abdeckung $= \tfrac1N\sum \mathbb 1[l \le y \le u]$. **Intervall-Score** (Winkler/Gneiting-Raftery): $(u - l) + \tfrac2\alpha (l - y)^+ + \tfrac2\alpha (y - u)^+$, $\alpha = 1 - p$. **Pinball** zur Stufe $\tau$: $\max(\tau (y - q),\ (\tau - 1)(y - q))$. Alle Größen in Aufträgen, je Depot durch den saisonal naiven Trainingsfehler (Tage vor 730) geteilt, über die Depots gemittelt.
**Orakel:** $Q_\tau = \mu\,\exp(\sigma z_\tau - \sigma^2/2)$ mit dem wahren Erwartungswert $\mu$ und der wahren Streuung $\sigma$ des Tages (Ausreißer und Rundung nicht berücksichtigt).

Implementiert in `fi_intervals.py` (Verfahren), `fi_baselines.py`, `fi_ets.py` (Punktmodelle), `fi_scenario.py` (das Portfolio), `fi_evaluation.py` (Analyse, Aufschlüsselungen, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)

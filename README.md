# 📏 Prognoseintervalle – wie sicher ist die Prognose?

Siebtes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger des [Boostings mit Lag-Merkmalen](https://github.com/sebastian-hanisch/boosting-forecast-demo): dort ging es um die Punktprognose, hier um die Frage, **wie weit der Ist-Wert davon abweichen darf**.
Geplant sind vier weitere Stücke (Hierarchische Abstimmung, Kombination, Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Eine Punktprognose sagt "morgen 120 Aufträge" und schweigt darüber, ob 100 oder 160 noch normal sind. Ein **Prognoseintervall** mit der Nennabdeckung 80 % verspricht, dass vier von fünf Ist-Werten hineinfallen. Die Demo nimmt die Punktprognosen der Vorgänger (Holt-Winters, Regression, Wochenmittel) auf einem **Portfolio von Depots** und macht daraus Intervalle auf **fünf Arten**:
mit der **Normalverteilung** (roh und im Log), mit den **Fehlern der Trainingstage** und mit **konformer Kalibrierung** an den zuletzt realisierten Fehlern (fest und adaptiv, ACI). Geprüft wird, **ob die Intervalle halten, was ihre Nennabdeckung verspricht** – im Mittel, je Wochentag, je Horizont, im Zeitverlauf und je Depot –, und was das an Breite kostet.
Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (`scipy` nur als Gegenprobe im Test).

**Bezug zu OR:** Sicherheitsbestände, Personalreserven und Kapazitätspuffer folgen aus Quantilen der Prognose, nicht aus dem Punktwert; das Bestands-Stück der Linie nimmt diese Quantile auf.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Konforme Intervalle brauchen stationäre Fehler." Das bestätigt sich – und es kommen drei Befunde dazu, die der Plan nicht vorhersah:

1. **Roh gerechnet sind Intervalle wochentagsabhängig falsch.** Bei Nennabdeckung 80 % deckt das Gauß-roh-Intervall im Mittel 75,7 %, aber je Wochentag zwischen **62,9 % und 97,8 %**: sonntags (0,35-faches Niveau) viel zu breit, an den starken Tagen zu schmal. Im Log gerechnet liegt derselbe Ansatz je Wochentag zwischen 75 und 80 %.
2. **Im stationären Fall kauft die konforme Kalibrierung Abdeckung, nicht Genauigkeit.** Standardfall (Holt-Winters, 30 Depots, Seed 3), Nennabdeckung 80 %: Gauß roh 75,7 %, Gauß im Log 78,0 %, empirisch 76,5 %, **konform 79,4 %, ACI 79,9 %** (Orakel 79,7 %); bei 95 % 89,0 / 93,1 / 93,3 / **95,2 / 95,0 %**. Der Intervall-Score bei 80 % aber ist für Gauß im Log 3,99, für konform 4,08 (Orakel 3,15): die Intervalle der Konformen sind breiter (2,81 gegen 2,66, skaliert), und das Fenster von 90 Werten macht die Quantile vermutlich unruhiger (nicht isoliert).
   Mit der Regression als Punktmodell (bessere, horizontunabhängige Prognose) kehrt sich das um: konform 3,25 gegen Gauß im Log 3,51 (Orakel 3,15).
3. **Ein Verteilungswechsel bricht alles, was nur die Vergangenheit kennt.** Verdoppelt sich das Rauschen ab Tag 900, decken die Intervalle danach bei Nennabdeckung 80 % nur noch **50,9 %** (Gauß im Log) und 49,6 % (empirisch); das konforme mit 90 Tagen Fenster 73,3 %, weil sein Fenster den Wechsel erst nach und nach aufnimmt; das **adaptive** Verfahren hält **80,0 %**. Die Garantie der Konformen gilt für austauschbare Fehler, nicht für einen Wechsel.
4. **Die Standardabweichung ist nicht robust.** Bei 5 % Ausreißertagen überdeckt das Gauß-roh-Intervall mit 66,2 % statt 50 % und ist 1,87-mal so breit wie das des Orakels; empirisch (47,4 %) und konform (49,8 %) bleiben bei den 50 %. Dafür wird ACI bei 95 % dann am breitesten: Breite 8,1 gegen 6,0 bei konform (Intervall-Score 11,6 gegen 9,8; empirisch 8,7, Gauß im Log 8,7).
5. **Die Nennabdeckung braucht genug Kalibrierpunkte.** Mit 30 Werten im Fenster verlangt 95 % den 31. von 30 Werten: das Quantil bleibt am Rand, die Garantie reicht nur bis (30−1)/31 = 93,5 %, gemessen 91,8 % – und ACI kann das nicht ausgleichen (91,8 %).
6. **Auch konform verliert bei langem Horizont.** Horizont 28, Holt-Winters: von Tag 1 zu Tag 28 fällt die Abdeckung bei Gauß im Log von 79,8 auf 72,8 %, bei konform von 80,4 auf 74,1 %; ACI hält 79,1 %. Warum die Konformen hier nachgeben, ist nicht isoliert (vermutet: die Fehler des Modells wachsen mit dem Alter der Parameter und das Fenster hinkt hinterher).

## Modell

- **Das Portfolio** (`fi_scenario.py`): 1 095 Tage je Depot wie in [Boosting](https://github.com/sebastian-hanisch/boosting-forecast-demo) (multiplikativ: Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, log-normales Rauschen, eigene Parameter je Depot), dazu zwei neue Regler: ein **Verteilungswechsel** (Faktor auf das Rauschen ab Tag 900) und **Ausreißertage** (Anteil der Tage mit einem zusätzlichen log-normalen Sprung, Streuung 0,7). Die wahren Quantile ($\mu\,e^{\sigma z - \sigma^2/2}$) sind das **Orakel**.
- **Punktmodelle** (`fi_baselines.py`, `fi_ets.py`): Holt-Winters multiplikativ (Stück 2, Standard), Regression auf Kalender und Aktionsplan im Log (Stück 4), Wochenmittel (Stück 1). Die Parameter stammen aus den Tagen **vor Tag 610**; die Tage 610–729 sind Kalibrierung, das Testjahr beginnt bei Tag 730.
- **Fehlermaß:** $s = \log\frac{y+1}{F+1}$; Prognosequantil $(F+1)\,e^{q} - 1$ (mindestens 0).
- **Verfahren** (`fi_intervals.py`): *Gauß roh* $F \pm z\sigma$ mit der Standardabweichung der Ein-Schritt-Fehler auf den Trainingstagen; *Gauß im Log* mit der Streuung der Ein-Schritt-Log-Fehler; *empirisch* mit deren Quantilen; *konform*: Quantile der zuletzt realisierten Fehler desselben Horizonts aus den letzten $W$ Ursprüngen, mit $\lceil (m+1)\tau\rceil$ (oben) und $\lfloor (m+1)\tau\rfloor$ (unten); *ACI* (Gibbs/Candès 2021): eine Fehlerrate je Horizont, nach jeder realisierten Prognose um $\gamma(\alpha - \text{Fehlschlag})$ nachgeführt.
- **Kennzahlen** (`fi_evaluation.py`): Abdeckung, Breite und Intervall-Score (Winkler; beide durch den saisonal naiven Trainingsfehler geteilt, über die Depots gemittelt) bei Nennabdeckung 50, 80 und 95 %; Pinball-Verlust über sechs Quantilstufen; Aufschlüsselungen nach Wochentag, Horizont, Zieltag und Depot.

## Methodik

- **Handrechnungen:** Log-Fehler, Prognosequantil, die Indizes $\lceil (m+1)\tau\rceil$ / $\lfloor (m+1)\tau\rfloor$ auf einem sortierten Fenster (auch mit `NaN`-Auffüllung und dem Randfall $m = 3$), die Tabelle der Kennzahlen (Abdeckung, Breite, Intervall-Score) auf vier Zellen, ein ACI-Schritt.
- **Gegenprobe:** die konformen Quantile des ganzen Portfolios gegen eine **unabhängige Schleife**; Normalquantile und log-normale Orakel-Quantile gegen `scipy`; das Orakel gegen eine Simulation.
- **Ein Fund in der eigenen Formel:** die Endlichkeitsgarantie hatte ich zuerst als $(b - a + 1)/(m+1)$ notiert; der Monte-Carlo-Test zeigte 0,80 statt 0,85 – richtig ist $(b - a)/(m+1)$ (Rang des neuen Werts unter $m+1$ Werten). Text und Test sind korrigiert.
- **Eigenschaften:** die konformen Quantile eines Ursprungs ändern sich nicht, wenn Scores mit Zieltag nach dem Ursprung überschrieben werden; ebenso alle fünf Verfahren, wenn die Reihe ab dem Ursprung überschrieben wird; ACI ohne Schrittweite ist genau konform, mit Schrittweite folgt die Fehlerrate der Rückmeldung von Hand; die Quantile sind monoton in der Stufe; das Orakel deckt bei 50/80/95 % innerhalb von 3 Prozentpunkten und hat den besten Score.
- **Statistik:** die Experimente mitteln über **drei feste Seeds** (Fehlerbalken = Standardfehler), die Preset-Zeilen sind **Einzelportfolios** (Seed 3).
- **Literatur** (nicht nachgebaut): Vovk/Gammerman/Shafer 2005 (konforme Vorhersage); Angelopoulos/Bates 2023 (Einführung); Gibbs/Candès 2021 (ACI); Gneiting/Raftery 2007 (Intervall-Score, Scoring-Regeln); Hyndman/Athanasopoulos, FPP3 (Prognoseintervalle).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Preset, Holt-Winters, 30 Depots, Seed 3; Nenn 80 %) | Abdeckung: Gauß roh 75,7 %, Gauß im Log 78,0 %, empirisch 76,5 %, konform **79,4 %**, ACI **79,9 %**, Orakel 79,7 %. Bei 95 %: 89,0 / 93,1 / 93,3 / 95,2 / 95,0 %. Gauß roh je Wochentag 62,9–97,8 %. MASE der Punktprognose 0,87. | `test_standard_preset` |
| **Scores im Standardfall** (Nenn 80 %) | Intervall-Score: Gauß roh 4,48, Gauß im Log 3,99, konform 4,08, ACI 4,16, Orakel 3,15; Pinball: 0,228 / 0,207 / 0,211 / 0,214, Orakel 0,165; Breite: 2,48 / 2,66 / 2,81 / 2,92, Orakel 2,28. | `test_standard_scores_and_the_price_of_calibration` |
| **Regression als Punktmodell** (Preset) | MASE 0,78; konform 80,0 %, Gauß im Log 75,2 %; Abdeckung bei Horizont 1 und 14 gleich; Intervall-Score konform 3,25 (Gauß im Log 3,51, Orakel 3,15). | `test_regression_preset` |
| **Verteilungswechsel** (Experiment, 20 Depots, 3 Seeds, Nenn 80 %, Zieltage ab Tag 900) | Faktor 1 / 1,5 / 2: Gauß im Log 78,2 / 62,3 / **50,4 %**, empirisch 77,2 / 61,3 / 49,4 %, konform 79,6 / 75,9 / **73,3 %**, ACI 80,1 / 80,1 / **80,2 %**; Gauß roh bei 2: 53,9 %. | `test_shift_experiment` |
| Verteilungswechsel (Preset, 30 Depots, Seed 3) | Nach dem Wechsel: Gauß im Log 50,9 %, empirisch 49,6 %, konform 73,3 %, ACI 80,0 %. | `test_shift_preset` |
| **Ausreißer** (Experiment, Nenn 50 %) | Abdeckung bei 0 / 2 / 5 %: Gauß roh 52,5 / 59,5 / **66,2 %**, Gauß im Log 49,1 / 54,3 / 59,1 %, empirisch 47,7 / 47,3 / 47,4 %, konform 51,2 / 50,5 / **49,8 %**. Breite relativ zum Orakel bei 5 %: Gauß roh 1,87, empirisch 1,22, konform 1,32. | `test_outlier_experiment` |
| Ausreißer (Preset, 5 %) | Nenn 50 %: Gauß roh 67,0 %, Gauß im Log 60,1 %, empirisch 46,5 %, konform 49,0 %, ACI 49,9 %. Nenn 95 %: Breite Gauß im Log 4,75, empirisch 4,05, konform 6,00, **ACI 8,11**, Orakel 2,73; Intervall-Score 8,69 / 8,72 / 9,82 / **11,59** / 8,04. | `test_outlier_preset` |
| **Kalibrierfenster** (Experiment, Nenn 95 %) | Konform 91,8 % (30 Tage) / 95,9 % (60) / 94,5 % (120); ACI 91,8 / 94,6 / 94,9 %. Streuung der Abdeckung zwischen den Depots bei 120 Tagen: konform 1,1, ACI 0,5 Prozentpunkte. | `test_window_experiment` |
| Kleines Fenster (Preset, 30, Nenn 95 %) | konform und ACI 91,7 %, Gauß im Log 93,1 %, empirisch 93,3 %. | `test_small_window_preset` |
| **Langer Horizont** (Preset, 28 Tage, Nenn 80 %) | Tag 1 → Tag 28: Gauß im Log 79,8 → 72,8 %, konform 80,4 → 74,1 %, ACI 80,2 → 79,1 %. | `test_long_horizon_preset` |
| Endlichkeitsgarantie | 90 Scores, Stufen 0,1 / 0,9: (82−9)/91 = 0,802; 95 %: 87/91 = 0,956; 30 Scores: höchstens 29/31 = 0,935. | `test_finite_sample_coverage_of_the_default_window` |

Die Preset-Zeilen sind **Einzelportfolios** (Seed 3); belastbar sind die Zeilen über drei Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Fehler sind normalverteilt und gleich breit** | Roh gerechnet sind die Intervalle je Wochentag falsch; Ausreißer blähen die Standardabweichung auf. | Rechnen im Log, Quantile statt Standardabweichung |
| **Trainingsfehler sind Prognosefehler** | Ein-Schritt-Fehler auf den Trainingstagen unterschätzen die Fehler bei längerem Horizont und außerhalb der Stichprobe. | Kalibrierung auf realisierten Fehlern je Horizont |
| **Die Fehler sind austauschbar** | Die Garantie der Konformen beruht darauf; ein Wechsel bricht sie. ACI gleicht ihn aus, braucht dafür zeitnahe Rückmeldung (die Rückmeldung für Horizont $k$ kommt $k$ Tage später) und wird nach Ausreißern breit (5 % Ausreißer, Nenn 95 %: Breite 8,1 gegen 6,0 bei konform). | ACI mit passender Schrittweite, kürzere Fenster |
| **Genug Kalibrierpunkte** | Für 95 % braucht es mindestens 39 Werte im Fenster; darunter bleibt das Quantil am Rand. | Größeres Fenster |
| **Abdeckung im Mittel genügt** | Die Garantie gilt im Mittel über Wochentage, Horizonte und Depots, nicht je Wochentag oder je Depot. | Konditionale Konforme (nach Wochentag oder Depot getrennt), Normierung |
| **Punktprognose und Intervall getrennt** | Intervalle folgen aus Fehlern eines Punktmodells; ein Quantilverlust im Modell (Boosting mit Pinball-Verlust) ist nicht gebaut. | Quantil-Boosting |
| **Erzeugte Portfolios, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal); echte Portfolios sind unordentlicher. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, 59 Tests, rund eine Minute): die Verfahren von Hand und gegen eine unabhängige Schleife und `scipy` (Indizes der Endlichkeitskorrektur, Monte-Carlo-Abdeckung, ACI-Schritt, log-normale Quantile), Eigenschaften (kein Blick in die Zukunft, Monotonie, Orakel kalibriert und am besten), das Portfolio (Wechsel wirkt nur nach Tag 900, Ausreißer verdicken die Enden), Kennzahlen von Hand, Aufschlüsselungen, Preset- und Permalink-Klemmen,
AppTest-Rauchtests (Standard, jedes Preset, Depot- und Ursprungs-Regler, Extremwerte, drei Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen mit Bändern und Rangfolgen).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `fi_constants.py` | Regler-Grenzen, Verfahren, Experiment-Seeds |
| `fi_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `fi_scenario.py` | Das Portfolio (Depots, Verteilungswechsel, Ausreißer, Orakel-Streuung) |
| `fi_baselines.py`, `fi_ets.py` | Punktmodelle: Wochenmittel, Holt-Winters, Regression |
| `fi_intervals.py` | Die fünf Verfahren und das Orakel |
| `fi_evaluation.py` | Analyse, Kennzahlen, Aufschlüsselungen, drei Experimente |
| `fi_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Quantil-Boosting (Pinball-Verlust im Modell) und Quantilregression; die Intervalle hier folgen aus Fehlern eines Punktmodells.
- Konditionale Konforme (nach Wochentag oder Depot getrennt kalibriert) und die Normierung der Scores durch eine Streuungsschätzung.
- Modellbasierte Intervalle aus der Zustandsraumform von Holt-Winters (geschlossene Varianzformel); *Gauß* nimmt die Ein-Schritt-Streuung für alle Horizonte.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: scipy).

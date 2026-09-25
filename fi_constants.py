"""Konstanten der Intervall-Demo: Vehikel "Tagesaufträge mehrerer Depots" (Stück 7 der Zeitreihen-Prognose-Linie), Intervall-Verfahren, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
FIT_END = 610                 # die Punktmodelle schätzen ihre Parameter auf den Tagen vor FIT_END; die Tage danach sind Kalibrierung für die konformen Verfahren
CAL_MAX = FIRST_TEST - FIT_END
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
SEASON_PERIOD = 7
SHIFT_DAY = 900               # ab diesem Tag gilt der Rauschfaktor des Verteilungswechsels
OUTLIER_SD = 0.7              # Streuung (im Log) eines Ausreißertages

# --- Glättung aus Stück 2 (Punktprognose) --------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------
DEFAULT_TREND = 10
DEFAULT_NOISE = 0.14
DEFAULT_EVENTS = 0.5
DEPOTS_MIN, DEPOTS_MAX, DEPOTS_STEP, DEFAULT_DEPOTS = 10, 100, 10, 30
NOISE_MIN, NOISE_MAX, NOISE_STEP = 0.04, 0.4, 0.02
TREND_MIN, TREND_MAX, TREND_STEP = -20, 40, 5
EVENTS_MIN, EVENTS_MAX, EVENTS_STEP = 0.0, 1.0, 0.25
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = 1.0, 2.0, 0.25, 1.0
OUTLIER_MIN, OUTLIER_MAX, OUTLIER_STEP, DEFAULT_OUTLIER = 0.0, 0.06, 0.01, 0.0
WINDOW_MIN, WINDOW_MAX, WINDOW_STEP, DEFAULT_WINDOW = 30, CAL_MAX, 15, 90
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.0, 0.1, 0.005, 0.02
NOMINALS = (0.5, 0.8, 0.95)
DEFAULT_NOMINAL = 0.8
POINT_MODELS = ("hw_mult", "regression", "snaive_k")
POINT_NAMES = {"hw_mult": "Holt-Winters multiplikativ (Stück 2)", "regression": "Regression auf Kalender und Aktionsplan (Stück 4)", "snaive_k": "Wochenmittel (Stück 1)"}
POINT_SHORT = {"hw_mult": "Holt-Winters", "regression": "Regression", "snaive_k": "Wochenmittel"}

# Intervall-Verfahren (Reihenfolge = Anzeige)
METHODS = ("gauss_raw", "gauss_log", "empirical", "conformal", "aci")
METHOD_NAMES = {"gauss_raw": "Gauß, roh (Punkt ± z·σ)", "gauss_log": "Gauß im Log (log-normal)", "empirical": "Empirisch (Trainingsfehler)", "conformal": "Konform (Kalibrierfenster)", "aci": "Adaptiv konform (ACI)"}
METHOD_SHORT = {"gauss_raw": "Gauß roh", "gauss_log": "Gauß log", "empirical": "Empirisch", "conformal": "Konform", "aci": "ACI"}

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_DEPOTS = 20
SHIFT_LEVELS = (1.0, 1.5, 2.0)
OUTLIER_LEVELS = (0.0, 0.02, 0.05)
WINDOW_LEVELS = (30, 60, 120)

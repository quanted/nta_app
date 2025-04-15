# Define pos/neg/neutral adduct lists
# Proton added - we observe Mass-(H+) and Mass+(Adduct)
NEG_ADDUCT_LI = [
    ("Cl", 35.976678),
    ("Br", 79.926161),
    ("HCO2", 46.005477),
    ("CH3CO2", 60.021127),
    ("CF3CO2", 113.992862),
]

# Proton subtracted - we observe Mass+(H+) and Mass+(Adduct)
POS_ADDUCT_LI = [
    ("Na", 21.981942),
    ("K", 37.955882),
    ("NH4", 17.026547),
]

NEUTRAL_LOSSES_LI = [
    ("H2O", -18.010565),
    ("2H2O", -36.02113),
    ("3H2O", -54.031695),
    ("4H2O", -72.04226),
    ("5H2O", -90.052825),
    ("NH3", -17.0265),
    ("O", -15.99490),
    ("CO", -29.00220),
    ("CO2", -43.989829),
    ("C2H4", -28.03130),
    ("CH2O2", 46.00550),  # note here and below - not losses? but still neutral?
    ("CH3COOH", 60.02110),
    ("CH3OH", 32.02620),
    ("CH3CN", 41.02650),
    ("(CH3)2CHOH", 60.05810),
]

# Set to tested memory capacity of WebApp for number of features in 'adduct_matrix'
MAX_NUM_ADDUCT_FEATURES = 12000

# Column names accessed throughout app
FEATURE_ID_COL = "Feature ID"
DASHBOARD_SEARCH_COL = "For_Dashboard_Search"
FORMULA_COL = "Formula"
MASS_COL = "Mass"
RETENTION_COL = "Retention_Time"
IONIZATION_COL = "Ionization_Mode"
MOLECULAR_FORMULA_COL = "MOLECULAR_FORMULA"

# Format lists to test values agains
ALLOWED_BLANK_FORMATS_LIST = ["Blank", "blank", "BLANK", "MB", "Mb", "mb", "mB"]
ACTIVE_COLUMNS_LIST = [
    "Retention_Time",
    "Mass",
    "Ionization_Mode",
    "Compound",
]

# Establish ordering of all possible front matter (tracer/no tracer, flags/no flags, etc.)
FRONT_MATTER_ORDERING = [
    "Ionization_Mode",
    "Mass",
    "Retention_Time",
    "Compound",
    "Tracer Chemical Match?",
    "Duplicate Feature?",
    "Is Adduct or Loss?",
    "Has Adduct or Loss?",
    "Adduct or Loss Info",
    "Final Occurrence Count",
    "Final Occurrence Percentage",
    "Final Occurrence Count (with flags)",
    "Final Occurrence Percentage (with flags)",
]
# config.py — THS Prediction Project
# Jedyne miejsce z ścieżkami i parametrami. Importuj w każdym notebooku.

from pathlib import Path

# ── Ścieżki bazowe ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MIMIC_DIR     = BASE_DIR / 'data' / 'MIMIC_III_Clinical'
PROCESSED_DIR = BASE_DIR / 'processed'
RESULTS_DIR   = BASE_DIR / 'results'
FIGURES_DIR   = RESULTS_DIR / 'figures'
MODELS_DIR    = RESULTS_DIR / 'models'
IMPUTED_DIR   = PROCESSED_DIR / 'time_windows_imputed'
SPLIT_DIR     = PROCESSED_DIR / 'splits'
WINDOWS_DIR   = PROCESSED_DIR / 'time_windows'

# Utwórz katalogi przy imporcie
for _d in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
           IMPUTED_DIR, SPLIT_DIR, WINDOWS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Parametry próbkowania ─────────────────────────────────────────────────────
SAMPLE_SIZE  = 46520   # 76.9% wszystkich pacjentów MIMIC-III (46 520)
RANDOM_SEED  = 42

# ── Okna czasowe i zestawy cech ───────────────────────────────────────────────
TIME_WINDOWS = [0.5, 1, 2, 3]      # godziny przed referencją
FEATURE_SETS = ['VS', 'VS_RB', 'VS_RB_BG']
N_FOLDS      = 10
N_BOOTSTRAP  = 1000

# ── Kolumny metadanych (nie-features) ─────────────────────────────────────────
META_COLS = ['HADM_ID', 'SUBJECT_ID', 'TIME_WINDOW',
             'REFERENCE_TIME', 'ONSET_TIME', 'HOURS_TO_SHOCK']

# ── MIMIC-III Item IDs ────────────────────────────────────────────────────────
ITEMIDS = {
    # Vital signs
    'HR':   [211, 220045],
    'SBP':  [51, 442, 455, 6701, 220179, 220050],
    'DBP':  [8368, 8440, 8441, 8555, 220180, 220051],
    'MBP':  [52, 6702, 443, 220052, 220181, 225312],
    'RESP': [618, 615, 220210, 224690],
    'TEMP': [223761, 678, 223762, 676],
    'SpO2': [646, 220277],
    # Labs — Routine Blood
    'Hemoglobin':  [50811, 51222],
    'Hematocrit':  [50810, 51221],
    'WBC':         [51300, 51301],
    'RBC':         [51279],
    'Platelets':   [51265],
    # Labs — Blood Gas
    'pH':          [50820],
    'PaCO2':       [50818],
    'PaO2':        [50821],
    'Lactate':     [50813],
    'Base_Excess': [50802],
    'TCO2':        [50804],
    # Transfusions
    'Packed_RBC':           [30001, 30179, 220996], # +225168
    'Whole_Blood':         [30002, 30004, 221013],
    'FFP':                  [30005, 30180, 220970, 30103, 226367],  # +30103, +226367
    'Platelet_Transfusion': [30006, 225170, 30105, 226369],         # +30105, +226369
}

# ── Zakresy fizjologiczne (outlier removal) ────────────────────────────────────
VITAL_RANGES = {
    'HR':   (20,  200),
    'SBP':  (30,  250), # 50,
    'DBP':  (20,  150),
    'MBP':  (30,  200),
    'RESP': (5,   60), # , 50
    'TEMP': (25,  42),    # °C 30, 
    'SpO2': (50,  100),
}

LAB_RANGES = {
    'Hemoglobin':  (3,   20), !!!
    'Hematocrit':  (10,  65),
    'WBC':         (0,   200), # , 50
    'RBC':         (1,   8),
    'Platelets':   (0,   1500), # ,1000
    'pH':          (6.8, 7.8),
    'PaCO2':       (10,  100),
    'PaO2':        (30,  700), # , 600
    'Lactate':     (0,   30), # , 20
    'Base_Excess': (-30, 30),
    'TCO2':        (5,   50),
}

# ── ICD-9 — identyfikacja traumy i wykluczenia ────────────────────────────────
TRAUMA_PATTERN = r'^(8[0-9][0-9]|9[0-5][0-9]|E8[0-9]{2}|E9[0-2][0-9])'

CHRONIC_DISEASE_PATTERN = r'^(1[4-9][0-9]|20[0-8]|28[0-9]|58[56]|57[12])'

EXCLUSION_ICD9 = {
    'septic_shock':      ['78552', '99592'],
    'cardiogenic_shock': ['78551'],
    'anaphylactic_shock':['9950', '99560', '99561', '99562', '99569'],
}

# ── XGBoost params (Supplemental Table 3 z publikacji) ───────────────────────
PUB_XGB_PARAMS = {
    'VS': dict(
        learning_rate=0.1, n_estimators=100, max_depth=4,
        subsample=0.7, colsample_bytree=0.6,
        tree_method='hist', device='cuda',   # usuń device='cuda' jeśli brak GPU
        eval_metric='logloss', random_state=RANDOM_SEED,
    ),
    'VS_RB': dict(
        learning_rate=0.1, n_estimators=100, max_depth=4,
        subsample=0.7, colsample_bytree=0.7,
        tree_method='hist', device='cuda',
        eval_metric='logloss', random_state=RANDOM_SEED,
    ),
    'VS_RB_BG': dict(
        learning_rate=0.1, n_estimators=180, max_depth=3,
        subsample=0.7, colsample_bytree=0.7,
        tree_method='hist', device='cuda',
        eval_metric='logloss', random_state=RANDOM_SEED,
    ),
}

# ── Feature categories ─────────────
FEATURE_CATEGORIES = {
    'Vital Signs':   ['HR', 'SBP', 'DBP', 'RESP', 'TEMP'],   # 5 × 3 timesteps = 15
    'Routine Blood': ['Hemoglobin', 'Hematocrit', 'WBC', 'RBC', 'Platelets'],
    'Blood Gas':     ['pH', 'PaCO2', 'PaO2', 'Lactate', 'Base_Excess', 'TCO2'],
}

# renormalizacja (0.292 + 0.249 + 0.225 = 0.766, nie 1.0)
_pub_sum = 0.292 + 0.249 + 0.225
PUB_IMPORTANCE = {
    'Vital Signs':   round(0.292 / _pub_sum, 4),   # 0.3812
    'Routine Blood': round(0.249 / _pub_sum, 4),   # 0.3251
    'Blood Gas':     round(0.225 / _pub_sum, 4),   # 0.2938
}

# ── Wartości referencyjne z publikacji (Zhao et al. 2022) ────────────────────
PUBLICATION = {
    'AUROC': {0.5: 0.958, 1.0: 0.968, 2.0: 0.953, 3.0: 0.945},
    'F1_5':  {1.0: 0.900},
    'note':  'PLAGH-ERD test set (ED trauma) — nieporównywalny 1:1 z MIMIC ICU',
}
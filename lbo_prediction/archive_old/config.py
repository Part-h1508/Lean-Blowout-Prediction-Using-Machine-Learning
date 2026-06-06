# config.py — all tunable parameters in one place

import os

DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models/")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots/")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results/")

# Column name in your CSV that contains the timestamp (change if needed)
TIME_COL = "time"

# WARNING: You must set this manually after inspecting your data.
# LBO_ROW_INDEX is the row number (integer) where LBO occurs in the file.
# Look at your CSV, find the row where LBO happens, and set it here.
# Example: if LBO occurs at row 39800 in a 40000-row file, set LBO_ROW_INDEX = 39800
# Based on data analysis: signal anomaly peaks at row 39571 in file 90
# File 90 starts at row 360000 (9 files × 40000), so combined index = 360000 + 39571
LBO_ROW_INDEX = 399571

# Columns to drop (non-sensor columns like IDs or notes)
COLS_TO_DROP = []

# Early warning window (rows before LBO to consider as precursor region)
# Adjust this to tune when the model starts warning before actual LBO
# Based on paper: NRMS precursor ~10% lead = ~4000 rows in 40K row file
WARNING_WINDOW = 4000

# Training parameters
TEST_SIZE = 0.2  # Percentage of data for testing (chronological split)
RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
}

# Threshold for LBO prediction
PREDICTION_THRESHOLD = 0.5

# Plotting parameters
PLOT_DPI = 100
PLOT_FIGSIZE = (14, 5)

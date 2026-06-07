import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

TIME_COL = 0

LBO_ROW_INDEX = 39000

COLS_TO_DROP = []

WARNING_WINDOW = 500

TEST_SIZE = 0.2

RANDOM_STATE = 42

SMOTE_K_NEIGHBORS = 5

PREDICTION_THRESHOLD = 0.5

PLOT_DPI = 300

PLOT_FIGSIZE = (12, 6)

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror"
}
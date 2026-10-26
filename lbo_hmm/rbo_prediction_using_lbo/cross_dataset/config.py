from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
LBO_DATA_DIR = ROOT_DIR / "lbo_prediction" / "data"
RBO_DATA_DIR = ROOT_DIR / "rbo_prediction_using_lbo" / "data" / "RBO"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

LBO_TIME_COLUMN = 0
LBO_SIGNAL_COLUMN = 1
RBO_TIME_COLUMN = 0
RBO_SIGNAL_COLUMN = 1

RBO_LIMIT = 6.3
RBO_FILENAME_VALUES_ARE_EQUIVALENCE_RATIOS = True
TARGET_COMPARABILITY_CONFIRMED = True
TIME_UNITS_CONFIRMED = True
SAMPLING_INTERVAL_TOLERANCE = 0.05

ROLLING_WINDOWS = [50, 100, 500]
LAGS = [1, 2, 5]
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror"
}
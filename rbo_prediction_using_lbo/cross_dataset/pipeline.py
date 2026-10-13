"""Train on LBO signals and test on RBO signals."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config

sys.path.insert(0, str(config.LBO_DATA_DIR))
from private_config import PHI_RATIO


@dataclass
class Run:
    time: np.ndarray
    signal: np.ndarray
    target: float


def load_lbo_runs():
    files, targets = find_lbo_files(config.LBO_DATA_DIR, PHI_RATIO)
    if config.LBO_SIGNAL_COLUMN is None:
        raise ValueError("Set LBO_SIGNAL_COLUMN to 1 or 2 in config.py after confirming the sensor mapping.")
    if config.LBO_SIGNAL_COLUMN == config.LBO_TIME_COLUMN:
        raise ValueError("LBO_SIGNAL_COLUMN cannot be the time column.")

    runs = []
    for path in files:
        data = pd.read_excel(
            path,
            header=None,
            usecols=[config.LBO_TIME_COLUMN, config.LBO_SIGNAL_COLUMN]
        )
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            raise ValueError("An LBO file has no usable time and signal rows.")
        runs.append(Run(
            time=data.iloc[:, 0].to_numpy(dtype=float),
            signal=data.iloc[:, 1].to_numpy(dtype=float),
            target=targets[path.stem] - 1.0
        ))
    return runs


def find_lbo_files(data_dir, phi_ratio):
    try:
        targets = {str(key).strip(): float(value) for key, value in phi_ratio.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("PHI_RATIO must contain numeric targets.") from error
    if not targets:
        raise ValueError("PHI_RATIO has no LBO target entries.")

    workbooks = {
        path.stem: path
        for path in data_dir.glob("*.xlsx")
        if _is_numeric(path.stem)
    }
    missing_files = set(targets) - set(workbooks)
    missing_targets = set(workbooks) - set(targets)
    if missing_files or missing_targets:
        problems = []
        if missing_files:
            problems.append(f"{len(missing_files)} expected workbook(s) missing")
        if missing_targets:
            problems.append(f"{len(missing_targets)} workbook(s) missing PHI_RATIO mappings")
        raise ValueError("LBO input mismatch: " + "; ".join(problems) + ".")

    files = sorted(workbooks.values(), key=lambda path: float(path.stem))
    return files, targets


def load_rbo_runs(data_dir=None):
    data_dir = config.RBO_DATA_DIR if data_dir is None else Path(data_dir)
    if not config.RBO_FILENAME_VALUES_ARE_EQUIVALENCE_RATIOS:
        raise ValueError("Confirm RBO filenames contain equivalence ratios before loading.")
    if config.RBO_LIMIT <= 0:
        raise ValueError("RBO_LIMIT must be positive.")

    runs = []
    files = sorted(
        (path for path in data_dir.glob("*.txt") if _is_numeric(path.stem)),
        key=lambda path: float(path.stem)
    )
    if not files:
        raise FileNotFoundError("No numeric-named RBO text files were found.")

    for path in files:
        try:
            data = np.loadtxt(path, usecols=(config.RBO_TIME_COLUMN, config.RBO_SIGNAL_COLUMN), ndmin=2)
        except (OSError, ValueError) as error:
            raise ValueError("RBO text files must contain whitespace-delimited numeric time and signal columns.") from error
        data = data[np.isfinite(data).all(axis=1)]
        if len(data) == 0:
            raise ValueError("An RBO file has no usable time and signal rows.")
        runs.append(Run(
            time=data[:, 0],
            signal=data[:, 1],
            target=1.0 - float(path.stem) / config.RBO_LIMIT
        ))
    return runs


def _is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def check_sampling_compatibility(runs):
    if not config.TIME_UNITS_CONFIRMED:
        raise ValueError("Confirm matching LBO and RBO time units in config.py before running.")

    intervals = []
    for run in runs:
        differences = np.diff(run.time)
        if len(differences) == 0 or not np.isfinite(differences).all() or (differences <= 0).any():
            raise ValueError("Sampling check failed: time values must increase in every file.")
        median_interval = float(np.median(differences))
        variation = np.max(np.abs(differences - median_interval)) / median_interval
        if variation > config.SAMPLING_INTERVAL_TOLERANCE:
            raise ValueError("Sampling check failed: a file has irregular time intervals.")
        intervals.append(median_interval)

    reference = float(np.median(intervals))
    if any(abs(interval - reference) / reference > config.SAMPLING_INTERVAL_TOLERANCE for interval in intervals):
        raise ValueError("Sampling check failed: LBO and RBO intervals are incompatible.")


def create_features(signal):
    signal = pd.Series(signal, dtype=float)
    scale = signal.rolling(window=500, min_periods=1).mean().abs()
    relative_signal = signal / (scale + 1e-8)
    data = pd.DataFrame({"signal": relative_signal})
    features = data.copy()

    for window in config.ROLLING_WINDOWS:
        rolling = data["signal"].rolling(window=window, min_periods=1)
        features[f"signal_roll_mean_{window}"] = rolling.mean()
        features[f"signal_roll_std_{window}"] = rolling.std()

    rolling_mean = data["signal"].rolling(window=500, min_periods=1).mean()
    rolling_std = data["signal"].rolling(window=500, min_periods=1).std()
    features["signal_nrms"] = rolling_std / (rolling_mean.abs() + 1e-8)

    for lag in config.LAGS:
        features[f"signal_lag_{lag}"] = data["signal"].shift(lag)
    features["signal_diff"] = data["signal"].diff()

    threshold = 0.72 * rolling_mean
    features["signal_theta"] = (data["signal"] < threshold).rolling(window=500, min_periods=1).mean()
    features["signal_autocorr_1"] = data["signal"].rolling(window=500, min_periods=50).corr(data["signal"].shift(1))
    features["signal_psd_proxy"] = rolling_std ** 2
    return features.ffill().fillna(0)


def make_dataset(runs):
    feature_frames = [create_features(run.signal) for run in runs]
    names = feature_frames[0].columns.tolist()
    if any(frame.columns.tolist() != names for frame in feature_frames):
        raise ValueError("Feature columns do not match between files.")
    features = pd.concat(feature_frames, ignore_index=True)
    targets = np.concatenate([
        np.full(len(run.signal), run.target, dtype=float)
        for run in runs
    ])
    return features, targets


def evaluate(train_runs, test_runs):
    X_train, y_train = make_dataset(train_runs)
    X_test, y_test = make_dataset(test_runs)
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("LBO and RBO feature columns do not match.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return (
        mean_absolute_error(y_test, predictions),
        np.sqrt(mean_squared_error(y_test, predictions)),
        r2_score(y_test, predictions)
    )


def validate_config():
    issues = []
    if config.LBO_SIGNAL_COLUMN not in (1, 2):
        issues.append("set LBO_SIGNAL_COLUMN to 1 or 2")
    if not config.RBO_FILENAME_VALUES_ARE_EQUIVALENCE_RATIOS:
        issues.append("confirm RBO filenames contain equivalence ratios")
    if config.RBO_LIMIT <= 0:
        issues.append("set a positive RBO_LIMIT")
    if not config.TARGET_COMPARABILITY_CONFIRMED:
        issues.append("confirm target equivalence with PHI_RATIO")
    if not config.TIME_UNITS_CONFIRMED:
        issues.append("confirm matching time units")
    if issues:
        raise ValueError("Configuration needs confirmation: " + "; ".join(issues) + ".")


def main():
    validate_config()
    lbo_runs = load_lbo_runs()
    rbo_runs = load_rbo_runs()
    check_sampling_compatibility(lbo_runs + rbo_runs)

    lbo_to_rbo = evaluate(lbo_runs, rbo_runs)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "LBO-to-RBO regression results",
        "Target: distance from blowout limit (0 at limit; positive on stable side)",
        "Target equivalence: confirmed in config.py",
        "Signal features: normalized by a causal 500-sample running mean",
        "Sampling interval check: passed; time values withheld",
        "Limitations: row-pooled metrics; feature windows are sample counts; no resampling.",
        "",
        "Train LBO, test RBO:",
        f"MAE: {lbo_to_rbo[0]:.6f}",
        f"RMSE: {lbo_to_rbo[1]:.6f}",
        f"R2: {lbo_to_rbo[2]:.6f}",
    ]
    report_path = config.OUTPUT_DIR / "results.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("LBO-to-RBO experiment completed.")
    print(f"Results saved to {report_path}.")


if __name__ == "__main__":
    main()
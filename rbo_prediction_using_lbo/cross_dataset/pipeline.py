"""Train on LBO signals and test on RBO signals."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import config

sys.path.insert(0, str(config.LBO_DATA_DIR))
from private_config import PHI_RATIO


@dataclass
class Run:
    time: np.ndarray
    signal: np.ndarray
    target: float
    name: str = ""


def lbo_target_from_ratio(phi_ratio):
    return float(phi_ratio) - 1.0


def rbo_target_from_ratio(phi_ratio, limit):
    return 1.0 - float(phi_ratio) / limit


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
            target=lbo_target_from_ratio(targets[path.stem]),
            name=path.name
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
            target=rbo_target_from_ratio(path.stem, config.RBO_LIMIT),
            name=path.name
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
    summaries = []
    for frame in feature_frames:
        summaries.append(summarize_features(frame))
    features = pd.DataFrame(summaries).fillna(0)
    targets = np.array([run.target for run in runs], dtype=float)
    return features, targets


def summarize_features(features):
    return pd.concat([
        features.mean().add_suffix("_mean"),
        features.std().fillna(0).add_suffix("_std")
    ])


def regression_metrics(y_true, predictions):
    score = r2_score(y_true, predictions) if np.ptp(y_true) > 0 else np.nan
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
        "r2": score
    }


def evaluate(train_runs, test_runs):
    X_train, y_train = make_dataset(train_runs)
    X_test, y_test = make_dataset(test_runs)
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("LBO and RBO feature columns do not match.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Ridge(alpha=config.RIDGE_ALPHA)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline_target = float(np.mean(y_train))
    baseline_predictions = np.full(len(y_test), baseline_target)

    per_file = []
    for index, run in enumerate(test_runs):
        actual = y_test[index:index + 1]
        per_file.append({
            "file": run.name or f"test_{index + 1}",
            "rows": len(run.signal),
            "target": run.target,
            "model": regression_metrics(actual, predictions[index:index + 1]),
            "baseline": regression_metrics(actual, baseline_predictions[index:index + 1])
        })

    return {
        "model": regression_metrics(y_test, predictions),
        "baseline": regression_metrics(y_test, baseline_predictions),
        "baseline_target": baseline_target,
        "per_file": per_file
    }


def validate_lbo_files(runs):
    features, targets = make_dataset(runs)
    predictions = np.empty(len(runs), dtype=float)
    baseline_predictions = np.empty(len(runs), dtype=float)

    for heldout in range(len(runs)):
        train_mask = np.arange(len(runs)) != heldout
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features.loc[train_mask])
        test_features = scaler.transform(features.loc[[heldout]])
        model = Ridge(alpha=config.RIDGE_ALPHA)
        model.fit(train_features, targets[train_mask])
        predictions[heldout] = model.predict(test_features)[0]
        baseline_predictions[heldout] = targets[train_mask].mean()

    return {
        "model": regression_metrics(targets, predictions),
        "baseline": regression_metrics(targets, baseline_predictions)
    }


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

    validation = validate_lbo_files(lbo_runs)
    results = evaluate(lbo_runs, rbo_runs)

    def format_metrics(name, metrics):
        return [
            f"{name} MAE: {metrics['mae']:.6f}",
            f"{name} RMSE: {metrics['rmse']:.6f}",
            f"{name} R2: {metrics['r2']:.6f}"
        ]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "LBO-to-RBO regression results",
        "Targets: LBO = PHI_RATIO - 1; RBO = 1 - (filename value / RBO_LIMIT)",
        "Target: distance from blowout limit (0 at limit; positive on stable side)",
        f"Model: Ridge regression (alpha={config.RIDGE_ALPHA})",
        "One mean/std feature summary per file; files have equal training weight",
        "Signal features: normalized by a causal 500-sample running mean",
        "Feature calculations restart independently for each file",
        "Sampling interval check: passed; time values withheld",
        "RBO metrics are exploratory; this holdout was previously examined.",
        "Per-file R2 is n/a because each file contributes one test observation.",
        "Feature windows are sample counts; no resampling is applied.",
        "",
        "LBO leave-one-file-out validation:",
        "LBO leave-one-file-out score used for alpha selection; treat as optimistic.",
        *format_metrics("Ridge", validation["model"]),
        *format_metrics("LBO mean baseline", validation["baseline"]),
        "",
        "RBO equal-file metrics:",
        *format_metrics("Model", results["model"]),
        f"LBO mean-target baseline (target={results['baseline_target']:.6f}) MAE: {results['baseline']['mae']:.6f}",
        f"LBO mean-target baseline RMSE: {results['baseline']['rmse']:.6f}",
        f"LBO mean-target baseline R2: {results['baseline']['r2']:.6f}",
        "",
        "Per-RBO-file metrics (R2 is n/a):"
    ]
    for row in results["per_file"]:
        report.append(
            f"{row['file']} ({row['rows']} rows, target={row['target']:.6f}): "
            f"model MAE={row['model']['mae']:.6f}, RMSE={row['model']['rmse']:.6f}; "
            f"baseline MAE={row['baseline']['mae']:.6f}, RMSE={row['baseline']['rmse']:.6f}"
        )
    report_path = config.OUTPUT_DIR / "results.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("LBO-to-RBO experiment completed.")
    print(f"Results saved to {report_path}.")


if __name__ == "__main__":
    main()
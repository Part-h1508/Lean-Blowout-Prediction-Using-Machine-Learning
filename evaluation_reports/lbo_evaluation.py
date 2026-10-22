"""Reproduce the original LBO leave-one-file-out evaluation."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
LBO_DIR = ROOT_DIR / "lbo_prediction"
DATA_DIR = LBO_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent


XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror"
}
LABELS = ["Safe", "Warning", "Critical"]


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_target_mapping():
    target_path = DATA_DIR / "private_config.py"
    spec = importlib.util.spec_from_file_location("_lbo_private_config", target_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError("Could not load the existing LBO target configuration.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = getattr(module, "PHI_RATIO", None)
    if not isinstance(targets, dict):
        raise ValueError("The existing LBO private_config.py has no PHI_RATIO mapping.")
    return targets


def load_data(data_dir=None, target_mapping=None):
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    target_mapping = load_target_mapping() if target_mapping is None else target_mapping
    targets = {str(key).strip(): float(value) for key, value in target_mapping.items()}
    files = {
        path.stem: path
        for path in data_dir.glob("*.xlsx")
        if is_numeric(path.stem)
    }
    missing_files = set(targets) - set(files)
    missing_targets = set(files) - set(targets)
    if missing_files or missing_targets:
        problems = []
        if missing_files:
            problems.append(f"{len(missing_files)} expected workbook(s) missing")
        if missing_targets:
            problems.append(f"{len(missing_targets)} workbook(s) lack PHI_RATIO mappings")
        raise ValueError("LBO input mismatch: " + "; ".join(problems) + ".")
    if not files:
        raise FileNotFoundError("No numeric-named LBO Excel files found.")

    frames = []
    expected_columns = None
    for name, path in sorted(files.items(), key=lambda item: float(item[0])):
        try:
            frame = pd.read_excel(path, header=None)
        except (OSError, ValueError, ImportError) as error:
            raise ValueError(f"Could not read LBO workbook {path.name}.") from error
        if expected_columns is None:
            expected_columns = frame.shape[1]
        elif frame.shape[1] != expected_columns:
            raise ValueError(f"{path.name} has a different number of columns than other LBO files.")
        frame["source_file"] = name
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    source_files = raw["source_file"].astype(str).str.strip()
    data = raw.drop(columns=["source_file"]).apply(pd.to_numeric, errors="coerce").dropna()
    source_files = source_files.loc[data.index]
    targets = pd.Series(targets)
    y = source_files.map(targets)
    if y.isna().any():
        raise ValueError("Some loaded LBO rows do not have PHI_RATIO targets.")
    if data.empty:
        raise ValueError("No numeric LBO rows remain after preprocessing.")
    return data, source_files, y, list(targets.index)


def create_features(data):
    """Match the original LOO feature extraction, including its fill policy."""
    features = data.copy()
    for window in [50, 100, 500]:
        for column in data.columns:
            rolling = data[column].rolling(window=window, min_periods=1)
            features[f"{column}_roll_mean_{window}"] = rolling.mean()
            features[f"{column}_roll_std_{window}"] = rolling.std()

    for column in data.columns:
        rolling_mean = data[column].rolling(window=500, min_periods=1).mean()
        rolling_std = data[column].rolling(window=500, min_periods=1).std()
        features[f"{column}_nrms"] = rolling_std / (rolling_mean.abs() + 1e-8)

    for lag in [1, 2, 5]:
        for column in data.columns:
            features[f"{column}_lag_{lag}"] = data[column].shift(lag)
    for column in data.columns:
        features[f"{column}_diff"] = data[column].diff()

    for column in data.columns:
        rolling_mean = data[column].rolling(window=500, min_periods=1).mean()
        rolling_std = data[column].rolling(window=500, min_periods=1).std()
        threshold = 0.72 * rolling_mean
        features[f"{column}_theta"] = (data[column] < threshold).rolling(
            window=500, min_periods=1
        ).mean()
        features[f"{column}_autocorr_1"] = data[column].rolling(
            window=500, min_periods=50
        ).corr(data[column].shift(1))
        features[f"{column}_psd_proxy"] = rolling_std ** 2

    return features.bfill().ffill().fillna(0).astype(float)


def classify(phi):
    if phi > 1.05:
        return "Safe"
    if phi > 1.025:
        return "Warning"
    return "Critical"


def evaluate(data=None, source_files=None, targets=None, target_files=None, model_class=None):
    if data is None:
        data, source_files, targets, target_files = load_data()
    model_class = XGBRegressor if model_class is None else model_class
    features = create_features(data)
    features.columns = features.columns.astype(str)
    results = []

    for test_file in target_files:
        train_mask = source_files != test_file
        test_mask = source_files == test_file
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"LOO split for {test_file} has an empty train or test set.")

        X_train = features.loc[train_mask]
        X_test = features.loc[test_mask]
        y_train = targets.loc[train_mask]
        y_test = targets.loc[test_mask]
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("LBO train/test feature schemas do not match.")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = model_class(**XGBOOST_PARAMS)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        actual = float(y_test.iloc[0])
        predicted = float(np.mean(predictions))
        percent_error = abs(predicted - actual) / actual * 100
        results.append({
            "file": test_file,
            "actual_target": actual,
            "predicted_target": predicted,
            "actual_state": classify(actual),
            "predicted_state": classify(predicted),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "percent_error": float(percent_error)
        })

    if not results:
        raise ValueError("No LBO files were evaluated.")
    matrix = confusion_matrix(
        [row["actual_state"] for row in results],
        [row["predicted_state"] for row in results],
        labels=LABELS
    )
    return {
        "files": results,
        "average_mae": float(np.mean([row["mae"] for row in results])),
        "average_percent_error": float(np.mean([row["percent_error"] for row in results])),
        "confusion_matrix": matrix
    }


def format_report(results):
    lines = [
        "Original LBO leave-one-file-out evaluation",
        "Target: PHI_RATIO; states use the original 1.05 and 1.025 cutoffs.",
        "Method: features computed over concatenated files, then one file held out per fold.",
        "The original bfill policy is retained for reproducibility.",
        "",
        "File\tActual target\tPredicted target\tActual state\tPredicted state\tMAE\tPercent error"
    ]
    for row in results["files"]:
        lines.append(
            f"{row['file']}\t{row['actual_target']:.6f}\t{row['predicted_target']:.6f}\t"
            f"{row['actual_state']}\t{row['predicted_state']}\t{row['mae']:.6f}\t"
            f"{row['percent_error']:.4f}%"
        )
    lines.extend([
        "",
        f"Average MAE: {results['average_mae']:.6f}",
        f"Average percent error: {results['average_percent_error']:.4f}%",
        "Confusion matrix (rows=actual, columns=predicted):",
        "Actual\\Predicted\t" + "\t".join(LABELS)
    ])
    lines.extend(
        f"{label}\t" + "\t".join(str(value) for value in row)
        for label, row in zip(LABELS, results["confusion_matrix"])
    )
    return "\n".join(lines) + "\n"


def main():
    results = evaluate()
    report_path = OUTPUT_DIR / "lbo_evaluation_results.txt"
    report_path.write_text(format_report(results), encoding="utf-8")
    print(f"LBO evaluation report saved to {report_path}.")


class SyntheticTests(unittest.TestCase):
    def test_file_coverage_fails_for_missing_mapping(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([[0, 1], [1, 2]]).to_excel(Path(folder) / "65.xlsx", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "lack PHI_RATIO mappings"):
                load_data(folder, {"66": 1.0})

    def test_inconsistent_workbook_schema_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([[0, 1], [1, 2]]).to_excel(Path(folder) / "65.xlsx", header=False, index=False)
            pd.DataFrame([[0, 1, 2], [1, 2, 3]]).to_excel(Path(folder) / "66.xlsx", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "different number of columns"):
                load_data(folder, {"65": 1.0, "66": 1.1})

    def test_metrics_and_confusion_matrix_on_synthetic_splits(self):
        source = pd.Series(["65"] * 60 + ["70"] * 60)
        data = pd.DataFrame({0: np.tile(np.arange(60), 2), 1: np.sin(np.arange(120))})
        targets = source.map({"65": 1.04, "70": 1.02})

        class FakeModel:
            def __init__(self, **kwargs):
                pass

            def fit(self, features, target):
                self.value = float(np.mean(target))

            def predict(self, features):
                return np.full(len(features), self.value)

        results = evaluate(data, source, targets, ["65", "70"], FakeModel)
        self.assertEqual(len(results["files"]), 2)
        self.assertEqual(results["confusion_matrix"].sum(), 2)


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        main()
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RESULTS_DIR = Path(__file__).resolve().parent
REPO_DIR = RESULTS_DIR.parents[2]
LBO_DIR = REPO_DIR / "lbo_prediction"
HMM_DIR = REPO_DIR / "lbo_hmm"
XGB_DATA_DIR = LBO_DIR / "data"
HMM_OUTPUT_DIR = HMM_DIR / "outputs" / "forecasting"

sys.path.insert(0, str(HMM_DIR))
sys.path.insert(0, str(XGB_DATA_DIR))

config_spec = importlib.util.spec_from_file_location(
    "lbo_prediction_config", LBO_DIR / "config.py"
)
if config_spec is None or config_spec.loader is None:
    raise ImportError(f"Unable to load XGBoost configuration from {LBO_DIR / 'config.py'}.")
xgb_config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(xgb_config)
import forecasting
from private_config import PHI_RATIO


def features_for_workbook(path):
    data = pd.read_excel(path, header=None)
    data = data.drop(columns=[column for column in xgb_config.COLS_TO_DROP if column in data])
    if xgb_config.TIME_COL in data.columns:
        data = data.drop(columns=[xgb_config.TIME_COL])
    data = data.dropna().apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError(f"{path.name} has no usable numeric sensor rows.")

    features = data.copy()
    for window in (50, 100, 500):
        for column in data.columns:
            features[f"{column}_roll_mean_{window}"] = data[column].rolling(
                window=window, min_periods=1
            ).mean()
            features[f"{column}_roll_std_{window}"] = data[column].rolling(
                window=window, min_periods=1
            ).std()

    for column in data.columns:
        rolling_mean = data[column].rolling(window=500, min_periods=1).mean()
        rolling_std = data[column].rolling(window=500, min_periods=1).std()
        features[f"{column}_nrms"] = rolling_std / (rolling_mean.abs() + 1e-8)

    for lag in (1, 2, 5):
        for column in data.columns:
            features[f"{column}_lag_{lag}"] = data[column].shift(lag)

    for column in data.columns:
        features[f"{column}_diff"] = data[column].diff()
        rolling_mean = data[column].rolling(window=500, min_periods=1).mean()
        features[f"{column}_theta"] = (data[column] < 0.72 * rolling_mean).rolling(
            window=500, min_periods=1
        ).mean()
        features[f"{column}_autocorr_1"] = data[column].rolling(
            window=500, min_periods=50
        ).corr(data[column].shift(1))
        features[f"{column}_psd_proxy"] = (
            data[column].rolling(window=500, min_periods=1).std() ** 2
        )
    features = features.bfill().ffill().fillna(0)
    features.columns = features.columns.astype(str)
    return features.reset_index(drop=True)


def build_xgb_features():
    frames = []
    labels = []
    for path in sorted(
        XGB_DATA_DIR.glob("*.xlsx"),
        key=lambda item: float(item.stem) if item.stem.replace(".", "", 1).isdigit() else float("inf"),
    ):
        try:
            float(path.stem)
        except ValueError:
            continue
        if path.stem not in PHI_RATIO:
            raise KeyError(f"No Phi/Phi_LBO target is configured for {path.name}.")
        frame = features_for_workbook(path)
        frames.append(frame)
        labels.extend([path.stem] * len(frame))
    if not frames:
        raise FileNotFoundError(f"No numeric-named workbooks found in {XGB_DATA_DIR}.")
    return pd.concat(frames, ignore_index=True), np.asarray(labels)


def evaluate_xgb(output_dir):
    features, source_files = build_xgb_features()
    test_file = "88"
    test_mask = source_files == test_file
    if not test_mask.any():
        raise ValueError(f"Holdout file {test_file}.xlsx is not present in the loaded data.")
    train_mask = ~test_mask
    if set(source_files[train_mask]) & set(source_files[test_mask]):
        raise ValueError("Training and holdout files overlap.")

    actual = float(PHI_RATIO[test_file])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.loc[train_mask])
    x_test = scaler.transform(features.loc[test_mask])
    y_train = np.asarray([PHI_RATIO[file_name] for file_name in source_files[train_mask]])
    model = XGBRegressor(**xgb_config.XGBOOST_PARAMS)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "model": "XGBoost regressor using the project's configured parameters",
        "holdout_file": f"{test_file}.xlsx",
        "training_files": sorted(set(source_files[train_mask]), key=float),
        "actual_phi_ratio": actual,
        "mean_predicted_phi_ratio": float(np.mean(predictions)),
        "median_predicted_phi_ratio": float(np.median(predictions)),
        "mae": float(mean_absolute_error(np.full(len(predictions), actual), predictions)),
        "rmse": float(np.sqrt(mean_squared_error(np.full(len(predictions), actual), predictions))),
        "mean_prediction_percent_error": float(
            abs(np.mean(predictions) - actual) / abs(actual) * 100
        ),
        "n_rows": int(len(predictions)),
        "note": (
            "Fresh fit using existing project parameters; features are generated separately "
            "within each workbook, and the scaler is fitted only on training files. R2 is "
            "omitted because this holdout has one constant target value."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_index": np.arange(len(predictions)),
        "actual_phi_ratio": actual,
        "predicted_phi_ratio": predictions,
    }).to_csv(output_dir / "xgboost_holdout_predictions.csv", index=False)
    (output_dir / "xgboost_holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(12, 4))
    plt.axhline(actual, color="black", linestyle="--", label="Actual Phi/Phi_LBO")
    plt.plot(predictions, linewidth=0.8, label="XGBoost prediction")
    plt.xlabel("Sample index in held-out file 88.xlsx")
    plt.ylabel("Phi/Phi_LBO")
    plt.title("XGBoost: file 88 holdout predictions")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_holdout_predictions.png", dpi=160)
    plt.close()

    residuals = actual - predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(predictions, residuals, s=5, alpha=0.35)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted Phi/Phi_LBO")
    plt.ylabel("Actual - predicted")
    plt.title("XGBoost: held-out residuals")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_holdout_residuals.png", dpi=160)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=features.columns)
    importance = importance.sort_values().tail(15)
    plt.figure(figsize=(9, 6))
    importance.plot.barh()
    plt.xlabel("Feature importance")
    plt.title("XGBoost: top 15 feature importances")
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_feature_importance.png", dpi=160)
    plt.close()
    return metrics


def prepare_hmm_outputs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    required = (
        "metrics_overall.csv",
        "metrics_by_run.csv",
        "predictions_10_step.csv",
    )
    for name in required:
        source = HMM_OUTPUT_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"Required saved HMM output is missing: {source}")
        (output_dir / name).write_bytes(source.read_bytes())

    overall = pd.read_csv(HMM_OUTPUT_DIR / "metrics_overall.csv")
    metrics_plot_dir = output_dir / "plots"
    metrics_plot_dir.mkdir(parents=True, exist_ok=True)
    groups = overall[overall["group"].isin(["LBO", "RBO"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        pivot = groups.pivot(index="group", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axis, rot=0)
        axis.set_title(f"HMM forecast and persistence: {metric}")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model")
    fig.tight_layout()
    fig.savefig(metrics_plot_dir / "hmm_metrics_by_family.png", dpi=160)
    plt.close(fig)

    predictions = pd.read_csv(HMM_OUTPUT_DIR / "predictions_10_step.csv")
    runs = forecasting.load_runs({
        "TAI": REPO_DIR / "TAI",
        "RBO": HMM_DIR / "rbo_prediction_using_lbo" / "data" / "RBO",
        "LBO": XGB_DATA_DIR,
    })
    forecast_plots = forecasting.plot_family_forecasts(
        runs, predictions, output_dir=metrics_plot_dir
    )
    return {
        "source": "lbo_hmm/outputs/forecasting/",
        "fresh_fit": False,
        "metrics": overall[overall["group"].isin(["LBO", "RBO"])].to_dict("records"),
        "plots": [path.name for path in forecast_plots]
        + ["hmm_metrics_by_family.png"],
    }


def create_synthetic_demo(output_dir):
    rng = np.random.default_rng(20261107)
    signal = np.zeros(4000, dtype=float)
    noise = rng.normal(0, 0.5, len(signal))
    for index in range(1, len(signal)):
        signal[index] = 0.92 * signal[index - 1] + noise[index]

    horizon = 10
    split = int(len(signal) * 0.8)
    origins = np.arange(split - 1, len(signal) - horizon, horizon)
    actual = signal[origins + horizon]
    predicted = signal[origins]
    metrics = {
        "status": "SYNTHETIC DEMO ONLY - not experimental data or a project-model evaluation",
        "synthetic_process": "AR(1), coefficient 0.92, Gaussian noise standard deviation 0.5",
        "forecast": "10-step persistence baseline",
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "n_predictions": int(len(actual)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic_demo_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    first = origins[0]
    last = origins[-1] + horizon
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(first, last + 1), signal[first:last + 1], label="Synthetic signal")
    plt.scatter(origins + horizon, predicted, s=12, label="Synthetic persistence forecast")
    plt.title("SYNTHETIC DEMO ONLY — not experimental or model-evaluation results")
    plt.xlabel("Synthetic sample index")
    plt.ylabel("Synthetic signal")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_demo_forecast.png", dpi=160)
    plt.close()


def main():
    actual_dir = RESULTS_DIR / "actual"
    xgb_dir = actual_dir / "xgboost"
    hmm_dir = actual_dir / "hmm"
    synthetic_dir = RESULTS_DIR / "synthetic_demo"
    xgb_metrics = evaluate_xgb(xgb_dir)
    hmm_info = prepare_hmm_outputs(hmm_dir)
    create_synthetic_demo(synthetic_dir)
    hmm_table = "\n".join(
        f"| {row['group']} | {row['model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} |"
        for row in hmm_info["metrics"]
    )

    report = f"""# Model results package

Generated locally from the project data and configured model workflows.

## Actual project data

- **XGBoost:** freshly fit with the project's configured model parameters, holding out file `88.xlsx`. Each workbook's features are computed separately and the scaler is fitted only on training files. Metrics and sample predictions are in `actual/xgboost/`.
- The existing saved XGBoost checkpoint was not reused because its 30-feature input schema does not match the current 15-feature data pipeline.
- The holdout is one operating file, so these results are preliminary and do not establish generalization. The older leave-one-out CSV is not used: its file labels are all `xx` and it does not contain numeric actual/predicted Phi values.
- **HMM:** the HMM metrics and 10-step predictions in `actual/hmm/` are existing results copied from `lbo_hmm/outputs/forecasting/`; plots were regenerated from those saved results and project measurements. A fresh HMM fit was not performed because `hmmlearn` is not installed in the current Python environment.
- XGBoost holdout summary: mean prediction percent error {xgb_metrics['mean_prediction_percent_error']:.3f}% across {xgb_metrics['n_rows']} samples; the holdout represents one operating file.
- HMM output provenance: `{hmm_info['source']}`.

| Dataset family | Model | MAE | RMSE |
|---|---|---:|---:|
{hmm_table}

## Synthetic demo (separate; not experimental results)

`synthetic_demo/` contains a deterministic AR(1) simulation and a 10-step persistence-baseline illustration. These files are labeled synthetic and must not be reported as measurements or performance of the project models.

## Reproduce

From the repository root, run:

```powershell
.\\.venv\\Scripts\\python.exe .\\lbo_hmm\\results\\professor_results_2026_11_07\\generate_results.py
```

The script writes only to this results folder. It does not modify the source datasets, saved model checkpoints, or prior output folders.
"""
    (RESULTS_DIR / "README.md").write_text(report, encoding="utf-8")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

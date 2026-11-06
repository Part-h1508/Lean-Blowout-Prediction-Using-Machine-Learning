import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
HMM_DIR = ROOT_DIR / "lbo_hmm"
HMM_FORECAST_SOURCE = HMM_DIR / "outputs" / "forecasting"
sys.path.insert(0, str(HMM_DIR))

import forecasting


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required results in {directory}: {', '.join(missing)}"
        )


def prepare_hmm_forecasting():
    output_dir = RESULTS_DIR / "hmm_forecasting"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = (
        "metrics_overall.csv",
        "metrics_by_run.csv",
        "predictions_10_step.csv",
    )
    require_files(HMM_FORECAST_SOURCE, required)
    for name in required:
        shutil.copy2(HMM_FORECAST_SOURCE / name, output_dir / name)

    overall = pd.read_csv(output_dir / "metrics_overall.csv")
    family_metrics = overall[overall["group"].isin(["LBO", "RBO"])].copy()
    if family_metrics.empty:
        raise ValueError("Saved HMM forecast metrics have no LBO or RBO rows.")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        pivot = family_metrics.pivot(index="group", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axis, rot=0)
        axis.set_title(f"10-step forecast {metric}")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Model")
    fig.tight_layout()
    fig.savefig(plot_dir / "hmm_metrics_by_family.png", dpi=160)
    plt.close(fig)

    predictions = pd.read_csv(output_dir / "predictions_10_step.csv")
    data_dirs = {
        "TAI": ROOT_DIR / "TAI",
        "RBO": HMM_DIR / "rbo_prediction_using_lbo" / "data" / "RBO",
        "LBO": ROOT_DIR / "lbo_prediction" / "data",
    }
    runs = forecasting.load_runs(data_dirs)
    forecast_plots = forecasting.plot_family_forecasts(
        runs, predictions, output_dir=plot_dir
    )
    return family_metrics, len(predictions), [path.name for path in forecast_plots]


def create_report(forecast_metrics, forecast_count, forecast_plots):
    xgb_dir = RESULTS_DIR / "xgboost"
    generation_dir = RESULTS_DIR / "hmm_generation"
    require_files(
        xgb_dir,
        ("summary_metrics.json", "metrics_by_file.csv", "predictions_by_sample.csv"),
    )
    require_files(
        generation_dir,
        ("summary_metrics.json", "metrics_by_file.csv", "generated_vs_actual_by_file.csv"),
    )
    xgb = json.loads((xgb_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    generation = json.loads(
        (generation_dir / "summary_metrics.json").read_text(encoding="utf-8")
    )

    forecast_table = "\n".join(
        f"| {row.group} | {row.model} | {row.MAE:.4f} | {row.RMSE:.4f} | "
        f"{int(row.n_predictions)} |"
        for row in forecast_metrics.itertuples(index=False)
    )
    forecast_plot_list = "\n".join(f"- `hmm_forecasting/plots/{name}`" for name in forecast_plots)
    report = f"""# Model results

This folder contains results for the standalone XGBoost and HMM workflows. The
LBO-trained/RBO-tested cross-dataset model is intentionally excluded.

## XGBoost: LBO leave-one-workbook-out

- Evaluation: {xgb['evaluation']}
- Held-out workbooks: {xgb['n_files']}; sample predictions: {xgb['n_predictions']:,}
- Mean per-workbook MAE: {xgb['file_mean_mae']:.5f}
- Mean per-workbook RMSE: {xgb['file_mean_rmse']:.5f}
- Mean per-workbook prediction percent error: {xgb['mean_file_percent_error']:.3f}%
- Pooled sample MAE/RMSE: {xgb['pooled_sample_mae']:.5f} / {xgb['pooled_sample_rmse']:.5f}

See `xgboost/metrics_by_file.csv`, `xgboost/summary_metrics.json`, and the
figures in `xgboost/`. Each complete workbook is held out. These are condition
regression results, not evidence of prospective real-time prediction.
The evaluation artifacts were already present in this folder; the packaging
script does not retrain XGBoost or regenerate its per-sample predictions.

## HMM: LBO signal generation

- Evaluation: {generation['evaluation']}
- Held-out workbooks: {generation['n_files']}; signal samples compared: {generation['n_rows_compared']:,}
- Average mean absolute difference: {generation['mean_metrics_across_files']['mean_absolute_difference']:.4f}
- Average variance absolute difference: {generation['mean_metrics_across_files']['variance_absolute_difference']:.4f}
- ACF MAE: {generation['mean_metrics_across_files']['acf_mae']:.4f}
- Normalized PSD L1: {generation['mean_metrics_across_files']['normalized_psd_l1']:.4f}
- Wasserstein distance: {generation['mean_metrics_across_files']['wasserstein_distance']:.4f}
- KS statistic: {generation['mean_metrics_across_files']['ks_statistic']:.4f}

See `hmm_generation/summary_metrics.json`, `hmm_generation/metrics_by_file.csv`,
and its plots. These are descriptive held-out diagnostics; the results do not
show that the generated signals faithfully reproduce the physical process.
These saved generation-evaluation artifacts are not refit by the packaging
script.

## HMM: within-run 10-step forecasting

The forecast metrics and prediction rows are copied from the existing
`lbo_hmm/outputs/forecasting/` evaluation. The charts below were regenerated
from those saved results and project measurements. This script does not refit
the HMM; `hmmlearn` is not installed in the current Python environment.
Forecasts use a chronological split within each run and are compared with a
persistence baseline. TAI is omitted because no TAI dataset folder is present.
The RBO rows below are within-run forecasts, not the excluded LBO-trained/RBO-tested
cross-dataset evaluation.

| Family | Model | MAE | RMSE | Predictions |
|---|---|---:|---:|---:|
{forecast_table}

Saved forecast rows: {forecast_count:,}.

{forecast_plot_list}

## Data handling and release

The LBO dataset notice identifies the experimental data as confidential.
Sample-level CSVs and signal plots in this folder are local evaluation outputs;
review data-sharing permissions before committing or publishing them.

GitHub Release `v1.2` was published from the pre-results repository state as
requested. This results package was added afterward and is not part of that
release tag. The package has not been committed or pushed.

## Rebuild forecast package

From the repository root, run:

```powershell
.\\.venv\\Scripts\\python.exe .\\final_results\\generate_results.py
```

The script packages the existing XGBoost and HMM-generation results without
overwriting them, copies the saved HMM forecast data, regenerates forecast
plots, and refreshes this summary.
"""
    (RESULTS_DIR / "README.md").write_text(report, encoding="utf-8")


def main():
    forecast_metrics, forecast_count, forecast_plots = prepare_hmm_forecasting()
    create_report(forecast_metrics, forecast_count, forecast_plots)
    print(f"Results summary written to {RESULTS_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
# Model results

This folder contains results for the standalone XGBoost and HMM workflows. The
LBO-trained/RBO-tested cross-dataset model is intentionally excluded.

## XGBoost: LBO leave-one-workbook-out

- Evaluation: leave one complete operating-condition workbook out
- Held-out workbooks: 10; sample predictions: 400,000
- Mean per-workbook MAE: 0.03264
- Mean per-workbook RMSE: 0.04532
- Mean per-workbook prediction percent error: 2.693%
- Pooled sample MAE/RMSE: 0.03866 / 0.05183

See `xgboost/metrics_by_file.csv`, `xgboost/summary_metrics.json`, and the
figures in `xgboost/`. Each complete workbook is held out. These are condition
regression results, not evidence of prospective real-time prediction.
The evaluation artifacts were already present in this folder; the packaging
script does not retrain XGBoost or regenerate its per-sample predictions.

## HMM: LBO signal generation

- Evaluation: leave one complete LBO workbook out
- Held-out workbooks: 10; signal samples compared: 400,000
- Average mean absolute difference: 1.0444
- Average variance absolute difference: 0.9788
- ACF MAE: 0.4730
- Normalized PSD L1: 1.0048
- Wasserstein distance: 1.1221
- KS statistic: 0.6449

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
| LBO | HMM | 0.1187 | 0.2258 | 8000 |
| LBO | Persistence | 0.1196 | 0.2258 | 8000 |
| RBO | HMM | 0.6216 | 0.7931 | 8800 |
| RBO | Persistence | 0.6678 | 0.8507 | 8800 |

Saved forecast rows: 16,800.

- `hmm_forecasting/plots/LBO_timeseries.png`
- `hmm_forecasting/plots/RBO_timeseries.png`

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
.\.venv\Scripts\python.exe .\final_results\generate_results.py
```

The script packages the existing XGBoost and HMM-generation results without
overwriting them, copies the saved HMM forecast data, regenerates forecast
plots, and refreshes this summary.

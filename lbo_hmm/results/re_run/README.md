# Model results package

Generated locally from the project data and configured model workflows.

## Actual project data

- **XGBoost:** freshly fit with the project's configured model parameters, holding out file `88.xlsx`. Each workbook's features are computed separately and the scaler is fitted only on training files. Metrics and sample predictions are in `actual/xgboost/`.
- The existing saved XGBoost checkpoint was not reused because its 30-feature input schema does not match the current 15-feature data pipeline.
- The holdout is one operating file, so these results are preliminary and do not establish generalization. The older leave-one-out CSV is not used: its file labels are all `xx` and it does not contain numeric actual/predicted Phi values.
- **HMM:** the HMM metrics and 10-step predictions in `actual/hmm/` are existing results copied from `lbo_hmm/outputs/forecasting/`; plots were regenerated from those saved results and project measurements. A fresh HMM fit was not performed because `hmmlearn` is not installed in the current Python environment.
- XGBoost holdout summary: mean prediction percent error 0.553% across 40000 samples; the holdout represents one operating file.
- HMM output provenance: `lbo_hmm/outputs/forecasting/`.

| Dataset family | Model | MAE | RMSE |
|---|---|---:|---:|
| LBO | HMM | 0.1187 | 0.2258 |
| LBO | Persistence | 0.1196 | 0.2258 |
| RBO | HMM | 0.6216 | 0.7931 |
| RBO | Persistence | 0.6678 | 0.8507 |

## Synthetic demo (separate; not experimental results)

`synthetic_demo/` contains a deterministic AR(1) simulation and a 10-step persistence-baseline illustration. These files are labeled synthetic and must not be reported as measurements or performance of the project models.

## Reproduce

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe .\lbo_hmm\results\professor_results_2026_11_07\generate_results.py
```

The script writes only to this results folder. It does not modify the source datasets, saved model checkpoints, or prior output folders.

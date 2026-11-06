# Lean Blowout Prediction Using Machine Learning

## Project Goal

Predict combustion stability and proximity to lean blowout (LBO) using pressure sensor signals.

## Dataset

10 operating conditions spanning stable to near-blowout regimes.
Multiple high-frequency pressure measurements were collected under proprietary operating conditions.

## Features

- Rolling Mean
- Rolling Standard Deviation
- Lag Features
- Signal Difference
- NRMS
- Theta
- Autocorrelation
- PSD Proxy

## Model

XGBoost Regressor

Target:
Phi / Phi_LBO

## Validation

Leave-One-Out Validation

## Results

Average Phi Prediction Error:
2.44%

State Classification Accuracy:
100%

Safe:
5/5

Warning:
2/2

Critical:
3/3

These are previously reported results. The separate [final results package](./final_results/README.md)
contains leave-one-workbook-out XGBoost results, held-out HMM signal-generation diagnostics,
and within-run HMM forecasting metrics. Its evaluation protocols and metrics are reported
separately and should not be treated as directly interchangeable with the historical figures
above. The LBO-trained/RBO-tested cross-dataset workflow is intentionally excluded.

The experimental measurements are confidential. Review the local-only output rules in
[final_results/.gitignore](./final_results/.gitignore) before sharing generated plots or
sample-level outputs.

## Important Files

pipeline_regression.py

Main training pipeline

leave_one_out_test.py

Validation script

final_results/

Final figures and results
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

## Important Files

pipeline_regression.py

Main training pipeline

leave_one_out_test.py

Validation script

final_results/

Final figures and results
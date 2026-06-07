# Lean Blowout Prediction using Machine Learning

## Objective

Predict combustion stability and lean blowout proximity using pressure sensor signals.

---

## Dataset

10 operating conditions spanning stable to near-blowout regimes.
Multiple high-frequency pressure measurements were collected under proprietary operating conditions.

---

## Features Used

Time-domain Features

- Rolling Mean
- Rolling Standard Deviation
- Lag Features
- Signal Difference

Physics-Based Features

- NRMS
- Theta
- Autocorrelation
- PSD Proxy

---

## Machine Learning Model

Model:

XGBoost Regressor

Target:

Phi/Phi_LBO

---

## Results

Average Prediction Error:

2.44 %

State Classification Accuracy:

100 %

Safe Conditions:

5/5 Correct

Warning Conditions:

2/2 Correct

Critical Conditions:

3/3 Correct

---

## Important Features

1. Rolling Mean (50)
2. Rolling Mean (100)
3. Rolling Mean (500)
4. Autocorrelation
5. PSD Proxy

---

## Conclusion

The proposed machine learning model successfully predicts combustion stability and correctly classifies operating conditions into Safe, Warning, and Critical states.

The model demonstrates strong agreement with experimentally derived lean blowout indicators.
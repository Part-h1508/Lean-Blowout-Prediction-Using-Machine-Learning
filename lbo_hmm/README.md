# LBO Signal HMM

This folder fits a Gaussian HMM to the LBO sensor signal and tests generated sequences against held-out files. It does not modify the existing LBO project or save raw/generated signals.

## Model

The observation is the single sensor amplitude, standardized using the training files only. A univariate Gaussian emission is a compact baseline for continuous signals; state persistence models temporal dependence. Hidden states are statistical states and are not labeled as stable, precursor, or blowout regimes.

The default uses four states. This is a documented starting choice, not a validated optimum. Each training file is passed as a separate sequence so transitions are not learned across file boundaries.

## Evaluation

Leave one complete workbook out at a time. For each fold, fit the standardizer and HMM on the other files, generate a sequence equal in length to the held-out file, and compare mean, variance, autocorrelation, normalized Welch PSD, Wasserstein distance, and KS statistic. Metrics are saved locally as JSON; signal values are not printed or saved.

Column 0 is time and column 1 is the sensor signal. Missing/non-numeric rows are dropped; the sampling check rejects non-increasing timestamps, irregular gaps, or files whose median intervals differ by more than the configured tolerance. The HMM uses sample steps; no resampling is performed.

## Dependency and run

`hmmlearn` was not installed in the workspace. Install dependencies with:

```powershell
python -m pip install -r .\lbo_hmm\requirements.txt
```

In the current Windows Python 3.14 environment, installing `hmmlearn` requires Microsoft C++ Build Tools because pip must compile its extension. A compatible Python environment with an available wheel may avoid that requirement.

Run synthetic tests first:

```powershell
python -m unittest discover -s .\lbo_hmm -p "test_*.py"
```

Run against confidential LBO workbooks only when ready:

```powershell
python .\lbo_hmm\evaluation.py
```

Metrics are diagnostics, not proof that samples faithfully replicate the physical process. One held-out file per fold limits leakage but cannot establish generalization beyond the available operating conditions.
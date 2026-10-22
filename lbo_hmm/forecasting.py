import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def _read_run(path, family):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, header=None)
    elif path.suffix.lower() == ".txt":
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    else:
        return None

    if family == "TAI":
        signal_column = 0
        time = None
    else:
        if frame.shape[1] < 2:
            raise ValueError(f"{path.name}: expected time and signal columns.")
        time = pd.to_numeric(frame.iloc[:, config.TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
        signal_column = config.SIGNAL_COLUMN

    signal = pd.to_numeric(frame.iloc[:, signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(signal)
    if time is not None:
        valid &= np.isfinite(time)
        time = time[valid]
    signal = signal[valid]
    return {
        "family": family,
        "run_id": f"{family}/{path.stem}",
        "file_name": path.name,
        "path": str(path),
        "time": time,
        "signal": signal,
        "original_rows": len(frame),
        "valid_rows": len(signal),
        "dropped_rows": len(frame) - len(signal),
    }


def load_runs(data_dirs=None):
    """Load independent TAI, RBO, and LBO measurement files."""
    data_dirs = config.FORECAST_DATA_DIRS if data_dirs is None else data_dirs
    runs = []
    for family in ("TAI", "RBO", "LBO"):
        folder = Path(data_dirs[family])
        if not folder.is_dir():
            print(f"Skipping {family}: dataset folder not found at {folder}.")
            continue
        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".txt"}
            and not path.stem.lower().startswith("data details")
        )
        for path in paths:
            run = _read_run(path, family)
            if run is not None:
                runs.append(run)
    if not runs:
        raise FileNotFoundError("No supported measurement files found in the configured family folders.")
    return runs


def robust_difference_scale(training_differences):
    median = np.median(training_differences)
    mad = np.median(np.abs(training_differences - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(training_differences))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return float(scale)


def select_training_segments(values, max_samples=None, chunk_length=None):
    """Subsample evenly spaced contiguous chunks without bridging their boundaries."""
    max_samples = config.FORECAST_MAX_FIT_SAMPLES if max_samples is None else max_samples
    chunk_length = config.FORECAST_CHUNK_LENGTH if chunk_length is None else chunk_length
    values = np.asarray(values, dtype=float)
    if len(values) <= max_samples or len(values) < chunk_length:
        return values.reshape(-1, 1), [len(values)]

    chunk_count = max(1, max_samples // chunk_length)
    starts = np.linspace(0, len(values) - chunk_length, chunk_count).round().astype(int)
    starts = np.unique(starts)
    chunks = [values[start:start + chunk_length] for start in starts]
    lengths = [len(chunk) for chunk in chunks]
    return np.concatenate(chunks).reshape(-1, 1), lengths


def causal_state_filter(model, observations):
    """Return filtered state probabilities using observations only through each row."""
    observations = np.asarray(observations, dtype=float).reshape(-1, 1)
    if not len(observations):
        return np.empty((0, model.n_components), dtype=float)

    log_emission = model._compute_log_likelihood(observations)
    log_transition = np.log(np.clip(model.transmat_, 1e-300, 1.0))
    log_alpha = np.log(np.clip(model.startprob_, 1e-300, 1.0)) + log_emission[0]
    log_alpha -= logsumexp(log_alpha)
    filtered = np.empty((len(observations), model.n_components), dtype=float)
    filtered[0] = np.exp(log_alpha)
    for index in range(1, len(observations)):
        log_alpha = log_emission[index] + logsumexp(
            log_alpha[:, None] + log_transition, axis=0
        )
        log_alpha -= logsumexp(log_alpha)
        filtered[index] = np.exp(log_alpha)
    return filtered


def forecast_from_state(model, state_probabilities, last_signal, scale, horizon=None):
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    probabilities = np.asarray(state_probabilities, dtype=float).copy()
    state_delta_means = model.means_[:, 0] * scale
    forecast = np.empty(horizon, dtype=float)
    current = float(last_signal)
    for step in range(horizon):
        probabilities = probabilities @ model.transmat_
        current += float(probabilities @ state_delta_means)
        forecast[step] = current
    return forecast


def safe_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else np.nan,
        "n_predictions": int(len(actual)),
    }


def evaluate_run(run, horizon=None, train_fraction=None, n_states=None, random_state=None):
    """Fit on the chronological training prefix and score non-overlapping test origins."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as error:
        raise RuntimeError("hmmlearn is required; install lbo_hmm/requirements.txt.") from error

    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    train_fraction = config.FORECAST_TRAIN_FRACTION if train_fraction is None else train_fraction
    n_states = config.FORECAST_N_STATES if n_states is None else n_states
    random_state = config.RANDOM_STATE if random_state is None else random_state
    signal = np.asarray(run["signal"], dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError(f"{run['run_id']}: signal must contain only finite values.")

    sample_count = len(signal)
    split = int(np.floor(train_fraction * sample_count))
    if split < config.FORECAST_MIN_TRAIN_ROWS or sample_count - split <= horizon:
        raise ValueError(f"{run['run_id']}: not enough rows for the requested split and horizon.")

    differences = np.diff(signal)
    training_differences = differences[:split - 1]
    scale = robust_difference_scale(training_differences)
    fit_values, lengths = select_training_segments(training_differences / scale)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=config.FORECAST_N_ITERATIONS,
        tol=1e-3,
        random_state=random_state,
        min_covar=1e-4,
        implementation="log",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(fit_values, lengths=lengths)

    filtered = causal_state_filter(model, differences / scale)
    origins = np.arange(
        split - 1,
        sample_count - horizon,
        config.FORECAST_ORIGIN_STRIDE,
        dtype=int,
    )
    rows = []
    for origin in origins:
        path = forecast_from_state(
            model, filtered[origin - 1], signal[origin], scale, horizon
        )
        target_index = origin + horizon
        rows.append({
            "family": run["family"],
            "run_id": run["run_id"],
            "file_name": run["file_name"],
            "origin_index": int(origin),
            "target_index": int(target_index),
            "horizon": horizon,
            "actual": float(signal[target_index]),
            "hmm_prediction": float(path[-1]),
            "persistence_prediction": float(signal[origin]),
        })

    predictions = pd.DataFrame(rows)
    metrics = []
    for name, column in (("HMM", "hmm_prediction"), ("Persistence", "persistence_prediction")):
        metrics.append({
            "family": run["family"],
            "run_id": run["run_id"],
            "model": name,
            **safe_metrics(predictions["actual"], predictions[column]),
            "train_rows": split,
            "test_rows": sample_count - split,
            "fit_samples": len(fit_values),
            "fit_segments": len(lengths),
            "n_states": n_states,
            "horizon": horizon,
            "converged": bool(model.monitor_.converged),
        })
    fitted = {"model": model, "scale": scale, "split_index": split}
    return predictions, metrics, fitted


def evaluate_runs(runs):
    prediction_frames = []
    metric_rows = []
    fitted_models = {}
    for run_number, run in enumerate(runs, start=1):
        prediction, metrics, fitted = evaluate_run(
            run, random_state=config.RANDOM_STATE + run_number - 1
        )
        prediction_frames.append(prediction)
        metric_rows.extend(metrics)
        fitted_models[run["run_id"]] = {
            **fitted,
            "family": run["family"],
            "file_name": run["file_name"],
        }
        print(
            f"[{run_number}/{len(runs)}] {run['run_id']}: "
            f"n={len(run['signal']):,}, forecasts={len(prediction):,}, "
            f"converged={fitted['model'].monitor_.converged}"
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_by_run = pd.DataFrame(metric_rows)
    return predictions, metrics_by_run, fitted_models


def aggregate_metrics(predictions):
    rows = []
    groups = list(predictions.groupby("family")) + [("ALL_DATASETS", predictions)]
    for group_name, group in groups:
        for model_name, column in (
            ("HMM", "hmm_prediction"),
            ("Persistence", "persistence_prediction"),
        ):
            rows.append({
                "group": group_name,
                "model": model_name,
                **safe_metrics(group["actual"], group[column]),
                "horizon": config.FORECAST_HORIZON,
                "origin_stride": config.FORECAST_ORIGIN_STRIDE,
            })
    return pd.DataFrame(rows)


def plot_family_forecasts(runs, predictions, output_dir=None, max_forecasts=300):
    import matplotlib.pyplot as plt

    output_dir = config.FORECAST_RESULTS_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_by_id = {run["run_id"]: run for run in runs}
    saved_paths = []

    for family in ("LBO", "RBO", "TAI"):
        family_predictions = predictions[predictions["family"] == family]
        if family_predictions.empty:
            print(f"Skipping {family} plot: no forecast rows are available.")
            continue

        run_id = family_predictions["run_id"].iloc[0]
        run = runs_by_id[run_id]
        sample = family_predictions[family_predictions["run_id"] == run_id].head(max_forecasts)
        first_origin = int(sample["origin_index"].iloc[0])
        final_target = int(sample["target_index"].iloc[-1])
        indices = np.arange(first_origin, final_target + 1)
        target_indices = sample["target_index"].to_numpy()
        signal = run["signal"]

        fig, axis = plt.subplots(figsize=(12, 4))
        axis.plot(indices, signal[indices], label="Actual", linewidth=1.2)
        axis.plot(
            target_indices,
            sample["hmm_prediction"],
            label="HMM 10-step forecast",
            marker=".",
            linewidth=1.0,
        )
        axis.plot(
            target_indices,
            sample["persistence_prediction"],
            label="Persistence baseline",
            marker=".",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.set_title(f"{family}: held-out 10-step forecasts ({run_id})")
        axis.set_xlabel("Sample index within run")
        axis.set_ylabel("Signal")
        axis.legend()
        axis.grid(True, alpha=0.25)
        fig.tight_layout()

        path = output_dir / f"{family}_timeseries.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def main():
    import joblib

    runs = load_runs()
    audit = pd.DataFrame([
        {key: value for key, value in run.items() if key not in {"time", "signal"}}
        for run in runs
    ])
    config.FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(config.FORECAST_OUTPUT_DIR / "data_audit.csv", index=False)
    predictions, metrics_by_run, models = evaluate_runs(runs)
    predictions.to_csv(config.FORECAST_OUTPUT_DIR / "predictions_10_step.csv", index=False)
    metrics_by_run.to_csv(config.FORECAST_OUTPUT_DIR / "metrics_by_run.csv", index=False)
    aggregate_metrics(predictions).to_csv(
        config.FORECAST_OUTPUT_DIR / "metrics_overall.csv", index=False
    )
    joblib.dump(models, config.FORECAST_OUTPUT_DIR / "hmm_forecast_models.joblib")
    plot_paths = plot_family_forecasts(runs, predictions)
    print(f"Forecast results saved to {config.FORECAST_OUTPUT_DIR}.")
    if plot_paths:
        print(f"Forecast plots saved to {config.FORECAST_RESULTS_DIR}.")


if __name__ == "__main__":
    main()
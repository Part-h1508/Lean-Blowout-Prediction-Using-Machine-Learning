from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate

import config


@dataclass
class Sequence:
    name: str
    time: np.ndarray
    signal: np.ndarray


def _is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_sequences(data_dir=None):
    data_dir = config.DATA_DIR if data_dir is None else Path(data_dir)
    paths = sorted(
        (path for path in data_dir.glob("*.xlsx") if _is_numeric(path.stem)),
        key=lambda path: float(path.stem)
    )
    if not paths:
        raise FileNotFoundError("No numeric-named LBO Excel files found.")

    sequences = []
    for path in paths:
        try:
            frame = pd.read_excel(
                path,
                header=None,
                usecols=[config.TIME_COLUMN, config.SIGNAL_COLUMN]
            )
        except (OSError, ValueError, ImportError) as error:
            raise ValueError("LBO files must contain the configured time and signal columns.") from error

        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) < 2:
            raise ValueError(f"{path.name} has fewer than two usable time/signal rows.")
        sequences.append(Sequence(path.name, values[:, 0], values[:, 1]))
    return sequences


def check_sampling(sequences):
    if not sequences:
        raise ValueError("No LBO sequences were supplied.")

    intervals = []
    for sequence in sequences:
        differences = np.diff(sequence.time)
        if not np.isfinite(differences).all() or (differences <= 0).any():
            raise ValueError(f"{sequence.name}: time values must be finite and increasing.")
        median = float(np.median(differences))
        variation = np.max(np.abs(differences - median)) / median
        if variation > config.SAMPLING_TOLERANCE:
            raise ValueError(f"{sequence.name}: irregular sampling or missing-time gap detected.")
        intervals.append(median)

    reference = float(np.median(intervals))
    if any(abs(value - reference) / reference > config.SAMPLING_TOLERANCE for value in intervals):
        raise ValueError("LBO files have incompatible sampling intervals.")
    return reference


def fit_standardizer(sequences):
    values = np.concatenate([sequence.signal for sequence in sequences])
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training signals must have nonzero finite variance.")
    return mean, scale


def standardize(sequence, mean, scale):
    return ((sequence.signal - mean) / scale).reshape(-1, 1)


def fit_hmm(sequences, mean, scale, n_states=None, random_state=None):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as error:
        raise RuntimeError(
            "hmmlearn is required; install lbo_hmm/requirements.txt. "
            "On Windows/Python 3.14, Microsoft C++ Build Tools may be required."
        ) from error

    sequences_scaled = [standardize(sequence, mean, scale) for sequence in sequences]
    lengths = [len(values) for values in sequences_scaled]
    observations = np.concatenate(sequences_scaled)
    model = GaussianHMM(
        n_components=config.N_STATES if n_states is None else n_states,
        covariance_type="diag",
        n_iter=config.N_ITERATIONS,
        min_covar=1e-3,
        random_state=config.RANDOM_STATE if random_state is None else random_state
    )
    model.fit(observations, lengths)
    return model


def generate_sequence(model, length, mean, scale, random_state=None):
    observations, _ = model.sample(length, random_state=random_state)
    return observations[:, 0] * scale + mean


def autocorrelation(values, max_lag=None):
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    limit = min(config.MAX_ACF_LAG if max_lag is None else max_lag, len(values) - 1)
    if limit <= 0:
        return np.zeros(0)
    correlation = correlate(centered, centered, mode="full", method="fft")[len(values) - 1:]
    if correlation[0] <= 0:
        return np.zeros(limit)
    return correlation[1:limit + 1] / correlation[0]
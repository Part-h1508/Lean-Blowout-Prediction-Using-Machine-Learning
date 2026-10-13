import json

import numpy as np
from scipy.signal import welch
from scipy.stats import ks_2samp, wasserstein_distance

import config
from model import check_sampling, fit_hmm, fit_standardizer, generate_sequence, autocorrelation


def compare_sequences(real, generated, sampling_interval):
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    real_acf = autocorrelation(real)
    generated_acf = autocorrelation(generated)
    acf_size = min(len(real_acf), len(generated_acf))

    sampling_frequency = 1.0 / sampling_interval
    segment_length = min(config.PSD_SEGMENT_LENGTH, len(real), len(generated))
    real_frequency, real_psd = welch(real, fs=sampling_frequency, nperseg=segment_length)
    generated_frequency, generated_psd = welch(generated, fs=sampling_frequency, nperseg=segment_length)
    if not np.allclose(real_frequency, generated_frequency):
        raise ValueError("PSD frequency grids do not match.")
    real_psd = real_psd / (np.sum(real_psd) + 1e-12)
    generated_psd = generated_psd / (np.sum(generated_psd) + 1e-12)
    ks = ks_2samp(real, generated)

    return {
        "real_mean": float(np.mean(real)),
        "generated_mean": float(np.mean(generated)),
        "mean_absolute_difference": float(abs(np.mean(real) - np.mean(generated))),
        "real_variance": float(np.var(real)),
        "generated_variance": float(np.var(generated)),
        "variance_absolute_difference": float(abs(np.var(real) - np.var(generated))),
        "acf_mae": float(np.mean(np.abs(real_acf[:acf_size] - generated_acf[:acf_size]))) if acf_size else 0.0,
        "normalized_psd_l1": float(np.sum(np.abs(real_psd - generated_psd))),
        "wasserstein_distance": float(wasserstein_distance(real, generated)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue)
    }


def evaluate_leave_one_file_out(sequences):
    if len(sequences) < 2:
        raise ValueError("At least two files are required for leave-one-file-out evaluation.")
    sampling_interval = check_sampling(sequences)
    results = []

    for index, held_out in enumerate(sequences):
        training = [sequence for sequence in sequences if sequence.name != held_out.name]
        mean, scale = fit_standardizer(training)
        model = fit_hmm(training, mean, scale, random_state=config.RANDOM_STATE + index)
        generated = generate_sequence(
            model,
            len(held_out.signal),
            mean,
            scale,
            random_state=config.RANDOM_STATE + index
        )
        results.append({
            "file": held_out.name,
            "rows": len(held_out.signal),
            "metrics": compare_sequences(held_out.signal, generated, sampling_interval)
        })

    metric_names = results[0]["metrics"].keys()
    summary = {
        metric: float(np.mean([row["metrics"][metric] for row in results]))
        for metric in metric_names
    }
    return {"split": "leave one complete file out", "summary": summary, "files": results}


def main():
    from model import load_sequences

    sequences = load_sequences()
    result = evaluate_leave_one_file_out(sequences)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = config.OUTPUT_DIR / "evaluation.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation metrics saved to {report}.")


if __name__ == "__main__":
    main()
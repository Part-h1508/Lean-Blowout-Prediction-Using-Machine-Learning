import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from model import (
    Sequence,
    check_sampling,
    fit_hmm,
    fit_standardizer,
    generate_sequence,
    load_sequences,
)


AIR_COLUMN = "Air (SLPM)"
PHI_COLUMN = "Equivalence ratio"
SELECTED_AIR_FLOWS = (90, 80, 65)
RANDOM_STATE = config.RANDOM_STATE


def select_phi_conditions(sequences, metadata_path, air_flows=SELECTED_AIR_FLOWS):
    metadata = pd.read_excel(metadata_path)
    missing_columns = {AIR_COLUMN, PHI_COLUMN} - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"{metadata_path} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    metadata = metadata.copy()
    metadata[AIR_COLUMN] = pd.to_numeric(metadata[AIR_COLUMN], errors="coerce")
    metadata[PHI_COLUMN] = pd.to_numeric(metadata[PHI_COLUMN], errors="coerce")

    sequences_by_flow = {}
    for sequence in sequences:
        try:
            flow = int(Path(sequence.name).stem)
        except ValueError:
            continue
        if flow in sequences_by_flow:
            raise ValueError(f"Multiple LBO sequences found for {flow} SLPM.")
        sequences_by_flow[flow] = sequence

    conditions = []
    for flow in air_flows:
        rows = metadata.loc[metadata[AIR_COLUMN] == flow, PHI_COLUMN].dropna()
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one finite equivalence ratio for {flow} SLPM "
                f"in {metadata_path}; found {len(rows)}."
            )
        if flow not in sequences_by_flow:
            raise FileNotFoundError(f"No LBO signal workbook found for {flow} SLPM.")
        conditions.append((flow, float(rows.iloc[0]), sequences_by_flow[flow]))

    return conditions


def generate_condition_signals(conditions, random_state=RANDOM_STATE):
    check_sampling([sequence for _, _, sequence in conditions])
    generated_conditions = []

    for index, (flow, phi, sequence) in enumerate(conditions):
        mean, scale = fit_standardizer([sequence])
        model = fit_hmm(
            [sequence], mean, scale, random_state=random_state + index
        )
        generated = generate_sequence(
            model,
            len(sequence.signal),
            mean,
            scale,
            random_state=random_state + index,
        )
        if not np.isfinite(generated).all():
            raise ValueError(f"HMM generated non-finite amplitudes for {flow} SLPM.")

        generated_conditions.append(
            (phi, sequence.time - sequence.time[0], generated)
        )

    return generated_conditions


def save_phi_figure(generated_conditions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, (phi, time_s, signal) in zip(axes, generated_conditions):
        axis.plot(time_s, signal, linewidth=0.7)
        axis.set_title(f"Φ = {phi:.3f}", fontsize=18)
        axis.set_ylabel("Amplitude", fontsize=16)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_phi_timeseries(
    data_dir=config.DATA_DIR,
    metadata_path=None,
    output_path=None,
    random_state=RANDOM_STATE,
):
    data_dir = Path(data_dir)
    metadata_path = (
        data_dir / "Data details.xlsx"
        if metadata_path is None
        else Path(metadata_path)
    )
    output_path = (
        config.FORECAST_RESULTS_DIR / "LBO_phi_generated_timeseries.png"
        if output_path is None
        else Path(output_path)
    )

    sequences = load_sequences(data_dir)
    conditions = select_phi_conditions(sequences, metadata_path)
    generated = generate_condition_signals(conditions, random_state=random_state)
    return save_phi_figure(generated, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate full-length LBO HMM signals at selected equivalence ratios."
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    output_path = generate_phi_timeseries(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        output_path=args.output,
        random_state=args.random_state,
    )
    print(f"HMM-generated LBO time-series figure saved to {output_path}.")


if __name__ == "__main__":
    main()

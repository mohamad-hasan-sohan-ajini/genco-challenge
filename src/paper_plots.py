"""Replot saved impedance-estimation results without rerunning the experiment.

Run: venv/bin/python src/paper_plots.py --seed 42
Each plotting method saves a title-free PDF beside its source result, adding
_replot before the extension. Add calls to further plotting methods in main().
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "local_predictions"
KEY_COLUMNS = ["dataset_index", "scenario_id", "bus_index"]
VOLTAGE_COLUMNS = {
    "Vm": ("vm_gt_pu", "vm_pred_pu", "p.u."),
    "Va": ("va_gt_deg", "va_pred_deg", "degrees"),
}
RUNS = {
    "Nominal impedance": ("original", "red"),
    "Aged impedance": ("true", "green"),
    "Estimated impedance": ("estimated", "blue"),
}


@dataclass
class ExperimentResults:
    directory: Path
    checkpoint: dict
    predictions: dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed (default: 42)")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return args


def validate_prediction_tables(tables):
    """Compare identical scenario/bus rows and ground truth for all curves."""
    reference = tables["Aged impedance"]
    if reference.empty:
        raise ValueError("No saved predictions are available")
    for label, table in tables.items():
        if table.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate scenario/bus rows for {label}")
        if not table[KEY_COLUMNS].equals(reference[KEY_COLUMNS]):
            raise ValueError(f"Prediction row keys differ for {label}")
        for gt_column, _, _ in VOLTAGE_COLUMNS.values():
            if not np.allclose(table[gt_column], reference[gt_column], equal_nan=True):
                raise ValueError(f"Ground truths differ for {label}: {gt_column}")


def load_experiment_results(seed):
    """Load the saved estimator checkpoint and its existing PF predictions.

    The estimator currently saves its final state as denoised_admittance.pt;
    it does not save a separately ranked best checkpoint. Predictions are the
    three comparison tables exported by that estimation run.
    """
    directory = RESULTS_DIR / f"seed{seed}"
    checkpoint_path = directory / "denoised_admittance.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing saved estimator: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed", seed) != seed:
        raise ValueError(f"Checkpoint seed does not match requested seed {seed}")
    columns = KEY_COLUMNS + [
        name for gt, prediction, _ in VOLTAGE_COLUMNS.values() for name in (gt, prediction)
    ]
    predictions = {}
    for label, (run, _) in RUNS.items():
        path = directory / f"aged_comparison_{run}_pf_ground_truth_predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved predictions: {path}")
        predictions[label] = (
            pd.read_parquet(path, columns=columns)
            .sort_values(KEY_COLUMNS)
            .reset_index(drop=True)
        )
    validate_prediction_tables(predictions)
    return ExperimentResults(directory, checkpoint, predictions)


def shared_histogram_bins(errors):
    """Match the original plot's combined per-curve 0.5–99.5% range."""
    if any(values.size == 0 for values in errors.values()):
        raise ValueError("Cannot plot a curve without finite prediction errors")
    lower = min(np.quantile(values, 0.005) for values in errors.values())
    upper = max(np.quantile(values, 0.995) for values in errors.values())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    return np.linspace(lower, upper, 101)


def plot_estimated_impedance_pf_error_histograms(results):
    """Plot Vm and Va signed errors in physical units, with a legend per panel.

    As in the source plot, this includes all buses, including known PF inputs.
    Fractions sum to one within the displayed histogram range for each curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, (variable, (gt, prediction, unit)) in zip(axes, VOLTAGE_COLUMNS.items()):
        errors = {}
        for label, table in results.predictions.items():
            values = table[prediction].to_numpy(dtype=float) - table[gt].to_numpy(dtype=float)
            errors[label] = values[np.isfinite(values)]
        bins = shared_histogram_bins(errors)
        for label, (_, color) in RUNS.items():
            counts, _ = np.histogram(errors[label], bins=bins)
            if not counts.sum():
                raise ValueError(f"No displayed errors for {label}: {variable}")
            ax.stairs(counts / counts.sum(), bins, label=label, color=color, linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(f"{variable} prediction error ({unit})")
        ax.set_ylabel("Fraction of observations")
        ax.grid(alpha=0.2)
        ax.legend()

    source_path = results.directory / "estimated_impedance_pf_error_histograms.pdf"
    output_path = source_path.with_name(f"{source_path.stem}_replot{source_path.suffix}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    return fig


def main(argv=None):
    args = parse_args(argv)
    results = load_experiment_results(args.seed)

    # Call each paper plotting method here; load the experiment only once.
    figure = plot_estimated_impedance_pf_error_histograms(results)
    plt.close(figure)


if __name__ == "__main__":
    main()

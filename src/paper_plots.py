"""Replot saved impedance-estimation results without rerunning the experiment.

Run: venv/bin/python src/paper_plots.py --seed 42
Each plotting method saves a title-free PDF beside its source result, adding
_replot before the extension. Add calls to further plotting methods in main().
"""

import argparse
import json
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
CASE_STYLES = {
    "Nominal impedance": ("Nominal impedance + aged PF", "red"),
    "Aged impedance": ("Aged impedance + aged PF", "green"),
    "Estimated impedance": ("Estimated impedance + aged PF", "blue"),
    "Nominal PF": ("Nominal impedance + nominal PF", "tab:orange"),
}
RUNS = {
    "Nominal impedance": "original",
    "Aged impedance": "true",
    "Estimated impedance": "estimated",
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
    for label, run in RUNS.items():
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
    return _plot_pf_error_comparison(
        results, results.predictions, "estimated_impedance_pf_error_histograms_replot.pdf"
    )


def load_nominal_pf_predictions(results):
    """Load or evaluate original-network PF with its own nominal impedance.

    Unlike aged_comparison_original, this uses original PF observations and
    original targets, not aged observations with substituted impedance.
    """
    path = results.directory / "case118_original_pf_ground_truth_predictions.parquet"
    if path.is_file():
        return pd.read_parquet(path)

    # Lazy imports: existing replots do not need model/data loading.
    import aged_impedance_estimation as experiment
    from torch_geometric.data import Batch
    from tqdm.auto import tqdm

    config = json.loads((results.directory / "experiment_config.json").read_text())
    root = Path(config["dataset_roots"]["case118_original"])
    artifacts = {
        name: experiment.hf_hub_download(
            experiment.MODEL_REPO_ID, f"{experiment.RUN_DIR}/{filename}",
            revision=experiment.MODEL_REVISION,
        )
        for name, filename in {
            "config": "case118_250000.yaml",
            "normalizer": "normalizer_stats.pt",
            "checkpoint": "best_model_state_dict.pt",
        }.items()
    }
    with open(artifacts["config"], encoding="utf-8") as handle:
        args = experiment.NestedNamespace(**experiment.yaml.safe_load(handle))
    normalizer = experiment.load_normalizer(args)
    stats = torch.load(artifacts["normalizer"], map_location="cpu", weights_only=True)
    normalizer.fit_from_dict(stats[experiment.NORMALIZER_NETWORK])
    dataset = experiment.HeteroGridDatasetDisk(
        root=str(root), data_normalizer=normalizer, transform=None,
        stream_partitions=getattr(args.data, "stream_partitions", "auto"),
    )
    transform = experiment.get_task_transforms(args)
    model = experiment.load_model(args)
    state = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
    state = experiment._normalize_loaded_state_dict_keys(state)
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    model.to(experiment.DEVICE).eval()
    indices = sorted(int(p.stem.removeprefix("data_index_"))
                     for p in Path(dataset.processed_dir).glob("data_index_*.pt"))
    if not indices:
        raise ValueError(f"No original PF scenarios found in {root}")
    frames = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(indices), experiment.BATCH_SIZE), desc="Nominal PF baseline"):
            batch_indices = indices[start:start + experiment.BATCH_SIZE]
            graphs = [transform(dataset[index]) for index in batch_indices]
            batch = Batch.from_data_list(graphs).to(experiment.DEVICE)
            output = model(batch)
            normalizer.inverse_output(output, batch)
            normalizer.inverse_transform(batch)
            prediction = output["bus"]
            target = batch["bus"].y
            ptr = batch["bus"].ptr.cpu().tolist()
            for j, index in enumerate(batch_indices):
                a, b = ptr[j:j + 2]
                frames.append(pd.DataFrame({
                    "dataset_index": index,
                    "scenario_id": int(graphs[j].scenario_id.item()),
                    "bus_index": np.arange(b - a),
                    "vm_gt_pu": target[a:b, experiment.VM_H].cpu().numpy(),
                    "vm_pred_pu": prediction[a:b, experiment.VM_OUT].cpu().numpy(),
                    "va_gt_deg": torch.rad2deg(target[a:b, experiment.VA_H]).cpu().numpy(),
                    "va_pred_deg": torch.rad2deg(prediction[a:b, experiment.VA_OUT]).cpu().numpy(),
                }))
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(path, index=False)
    print(f"Saved nominal PF baseline to {path}")
    return table


def plot_nominal_vs_aged_pf_error_histograms(results):
    """Compare correctly specified original PF and aged PF error distributions."""
    cases = {
        "Nominal PF": load_nominal_pf_predictions(results),
        "Aged impedance": results.predictions["Aged impedance"],
    }
    return _plot_pf_error_comparison(
        results, cases, "nominal_vs_aged_pf_error_histograms_replot.pdf"
    )


def plot_hidden_aging_error_histograms(results):
    """Notebook's second plot: aged PF with nominal versus aged impedance."""
    cases = {
        "Nominal impedance": results.predictions["Nominal impedance"],
        "Aged impedance": results.predictions["Aged impedance"],
    }
    return _plot_pf_error_comparison(
        results, cases, "hidden_aging_error_histograms_replot.pdf"
    )


def _plot_pf_error_comparison(results, cases, filename):
    """Shared two-panel physical-unit errors, with one legend per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    statistics = []
    for ax, (variable, (gt, prediction, unit)) in zip(axes, VOLTAGE_COLUMNS.items()):
        errors = {}
        for case, table in cases.items():
            label, _ = CASE_STYLES[case]
            if table.duplicated(KEY_COLUMNS).any():
                raise ValueError(f"Duplicate prediction rows for {label}")
            values = table[prediction].to_numpy(dtype=float) - table[gt].to_numpy(dtype=float)
            errors[case] = values[np.isfinite(values)]
            statistics.append({
                "Variable": variable,
                "Case": label,
                "Unit": unit,
                "Count": errors[case].size,
                "Mean": errors[case].mean(),
                "Std": errors[case].std(ddof=0),
            })
        bins = shared_histogram_bins(errors)
        for case in cases:
            label, color = CASE_STYLES[case]
            counts, _ = np.histogram(errors[case], bins=bins)
            if not counts.sum():
                raise ValueError(f"No displayed errors for {label}: {variable}")
            ax.stairs(counts / counts.sum(), bins, label=label, color=color, linewidth=1.5)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(f"{variable} prediction error ({unit})")
        ax.set_ylabel("Fraction of observations")
        ax.grid(alpha=0.2)
        ax.legend()
    output_path = results.directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    print("\nSigned prediction-error statistics (all finite errors; population std, ddof=0):")
    print(pd.DataFrame(statistics).to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    return fig


def main(argv=None):
    args = parse_args(argv)
    results = load_experiment_results(args.seed)

    # Call each paper plotting method here; load the experiment only once.
    figure = plot_estimated_impedance_pf_error_histograms(results)
    plt.close(figure)
    print("\n" + "-" * 100 + "\n")
    figure = plot_nominal_vs_aged_pf_error_histograms(results)
    plt.close(figure)
    print("\n" + "-" * 100 + "\n")
    figure = plot_hidden_aging_error_histograms(results)
    plt.close(figure)


if __name__ == "__main__":
    main()

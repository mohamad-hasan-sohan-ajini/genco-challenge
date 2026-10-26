"""Replot saved impedance-estimation results without rerunning the experiment.

Run: venv/bin/python src/paper_plots.py --case 57 --seed 0
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

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "local_predictions"  # Legacy IEEE-118 results.
TB_DIR = PROJECT_DIR / "tb"
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
    case: int = 118


def results_root(case):
    """Prefer the case-specific layout; retain legacy IEEE-118 access."""
    root = PROJECT_DIR / "results" / f"case{case}"
    if case == 118 and not root.exists() and RESULTS_DIR.is_dir():
        return RESULTS_DIR
    return root


def experiment_directory(seed, case):
    directory = PROJECT_DIR / "results" / f"case{case}" / f"seed{seed}"
    legacy = RESULTS_DIR / f"seed{seed}"
    if case == 118 and not directory.exists() and legacy.is_dir():
        return legacy
    return directory


def tensorboard_directory(seed, case):
    directory = TB_DIR / f"case{case}" / f"seed{seed}"
    legacy = TB_DIR / f"seed{seed}"
    if case == 118 and not directory.exists() and legacy.is_dir():
        return legacy
    return directory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(14, 30, 57, 118), default=118,
                        help="IEEE bus system (default: 118)")
    parser.add_argument(
        "--seed", type=int, default=0, help="Experiment seed (default: 0)"
    )
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


def load_experiment_results(seed, case=118):
    """Load the saved estimator checkpoint and its existing PF predictions.

    The estimator currently saves its final state as denoised_admittance.pt;
    it does not save a separately ranked best checkpoint. Predictions are the
    three comparison tables exported by that estimation run.
    """
    directory = experiment_directory(seed, case)
    checkpoint_path = directory / "denoised_admittance.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing saved estimator: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed", seed) != seed:
        raise ValueError(f"Checkpoint seed does not match requested seed {seed}")
    if checkpoint.get("case", 118) != case:
        raise ValueError(f"Checkpoint case does not match requested IEEE-{case}: {checkpoint_path}")
    columns = KEY_COLUMNS + [
        name
        for gt, prediction, _ in VOLTAGE_COLUMNS.values()
        for name in (gt, prediction)
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
    return ExperimentResults(directory, checkpoint, predictions, case=case)


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
        results,
        results.predictions,
        "estimated_impedance_pf_error_histograms_replot.pdf",
    )


def load_nominal_pf_predictions(results):
    """Load or evaluate original-network PF with its own nominal impedance.

    Unlike aged_comparison_original, this uses original PF observations and
    original targets, not aged observations with substituted impedance.
    """
    original_name = f"case{results.case}_original"
    path = results.directory / f"{original_name}_pf_ground_truth_predictions.parquet"
    if path.is_file():
        return pd.read_parquet(path)

    # Lazy imports: existing replots do not need model/data loading.
    import aged_impedance_estimation as experiment
    from torch_geometric.data import Batch
    from tqdm.auto import tqdm

    config = json.loads((results.directory / "experiment_config.json").read_text())
    root = Path(config["dataset_roots"][original_name])
    model, normalizer, args, _ = experiment.load_model_bundle(results.case)
    dataset = experiment.HeteroGridDatasetDisk(
        root=str(root),
        data_normalizer=normalizer,
        transform=None,
        stream_partitions=getattr(args.data, "stream_partitions", "auto"),
    )
    transform = experiment.get_task_transforms(args)
    indices = sorted(
        int(p.stem.removeprefix("data_index_"))
        for p in Path(dataset.processed_dir).glob("data_index_*.pt")
    )
    if not indices:
        raise ValueError(f"No original PF scenarios found in {root}")
    frames = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(indices), experiment.BATCH_SIZE), desc="Nominal PF baseline"
        ):
            batch_indices = indices[start : start + experiment.BATCH_SIZE]
            graphs = [transform(dataset[index]) for index in batch_indices]
            batch = Batch.from_data_list(graphs).to(experiment.DEVICE)
            output = model(batch)
            if output["bus"].size(1) > max(experiment.PG_OUT, experiment.QG_OUT):
                normalizer.inverse_output(output, batch)
            normalizer.inverse_transform(batch)
            prediction = output["bus"]
            target = batch["bus"].y
            ptr = batch["bus"].ptr.cpu().tolist()
            for j, index in enumerate(batch_indices):
                a, b = ptr[j : j + 2]
                frames.append(
                    pd.DataFrame(
                        {
                            "dataset_index": index,
                            "scenario_id": int(graphs[j].scenario_id.item()),
                            "bus_index": np.arange(b - a),
                            "vm_gt_pu": target[a:b, experiment.VM_H].cpu().numpy(),
                            "vm_pred_pu": prediction[a:b, experiment.VM_OUT]
                            .cpu()
                            .numpy(),
                            "va_gt_deg": torch.rad2deg(target[a:b, experiment.VA_H])
                            .cpu()
                            .numpy(),
                            "va_pred_deg": torch.rad2deg(
                                prediction[a:b, experiment.VA_OUT]
                            )
                            .cpu()
                            .numpy(),
                        }
                    )
                )
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
            values = table[prediction].to_numpy(dtype=float) - table[gt].to_numpy(
                dtype=float
            )
            errors[case] = values[np.isfinite(values)]
            statistics.append(
                {
                    "Variable": variable,
                    "Case": label,
                    "Unit": unit,
                    "Count": errors[case].size,
                    "Mean": errors[case].mean(),
                    "Std": errors[case].std(ddof=0),
                }
            )
        bins = shared_histogram_bins(errors)
        for case in cases:
            label, color = CASE_STYLES[case]
            counts, _ = np.histogram(errors[case], bins=bins)
            if not counts.sum():
                raise ValueError(f"No displayed errors for {label}: {variable}")
            ax.stairs(
                counts / counts.sum(), bins, label=label, color=color, linewidth=1.5
            )
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(f"{variable} prediction error ({unit})")
        ax.set_ylabel("Fraction of observations")
        ax.grid(alpha=0.2)
        ax.legend()
    output_path = results.directory / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    print(
        "\nSigned prediction-error statistics (all finite errors; population std, ddof=0):"
    )
    print(
        pd.DataFrame(statistics).to_string(
            index=False, float_format=lambda value: f"{value:.8g}"
        )
    )
    return fig


def load_tensorboard_impedance_rmse(seed, case=118):
    """Read every scalar point from the latest run containing both RMSE tags."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags = ("epoch/reactance_mismatch_rmse", "epoch/resistance_mismatch_rmse")
    seed_dir = tensorboard_directory(seed, case)
    run_dirs = sorted(
        {path.parent for path in seed_dir.rglob("events.out.tfevents.*")}, reverse=True
    )
    for run_dir in run_dirs:
        events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        events.Reload()
        if not all(tag in events.Tags().get("scalars", []) for tag in tags):
            continue
        curves = {}
        for tag in tags:
            table = pd.DataFrame(
                [
                    {
                        "epoch": event.step,
                        "value": event.value,
                        "wall_time": event.wall_time,
                    }
                    for event in events.Scalars(tag)
                ]
            )
            if table.empty:
                raise ValueError(f"Empty TensorBoard series: {run_dir}: {tag}")
            # In case a resumed run logs the same step again, retain its latest value.
            table = (
                table.sort_values("wall_time", kind="stable")
                .drop_duplicates("epoch", keep="last")
                .sort_values("epoch")
            )
            if not np.isfinite(table["value"]).all():
                raise ValueError(f"Non-finite TensorBoard values: {run_dir}: {tag}")
            curves[tag] = table
        print(f"TensorBoard IEEE-{case}, seed {seed}: {run_dir}")
        return curves
    raise FileNotFoundError(
        f"No TensorBoard run with both impedance RMSE tags in {seed_dir}"
    )


def print_impedance_error_reduction(curves, case=118):
    """Compare nominal and final estimated impedances to the same aged truth."""
    absolute_rows, relative_rows = [], []
    for seed, series in curves.items():
        path = experiment_directory(seed, case) / "denoised_admittance.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("seed", seed) != seed:
            raise ValueError(f"Checkpoint seed mismatch: {path}")
        if checkpoint.get("case", 118) != case:
            raise ValueError(f"Checkpoint case mismatch: {path}")
        mask = checkpoint["learnable_line_mask"].bool()
        for component, symbol in (("reactance", "X"), ("resistance", "R")):
            actual, nominal, estimated = [
                checkpoint[f"{prefix}_{component}_pu"][mask]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                for prefix in ("actual_aged", "initial", "estimated")
            ]
            if not actual.size or not all(
                np.isfinite(v).all() for v in (actual, nominal, estimated)
            ):
                raise ValueError(
                    f"Invalid impedance values for seed {seed}: {component}"
                )
            baseline_error = nominal - actual
            final_error = estimated - actual
            baseline_rmse = np.sqrt(np.mean(baseline_error**2))
            final_rmse = np.sqrt(np.mean(final_error**2))
            last = series[f"epoch/{component}_mismatch_rmse"].iloc[-1]
            if int(last.epoch) != checkpoint["epochs"] or not np.isclose(
                final_rmse, last.value, rtol=1e-5, atol=1e-9
            ):
                raise ValueError(
                    f"Checkpoint does not match the final TensorBoard epoch: {path}: {component}"
                )
            reduction = lambda before, after: (
                100 * (1 - after / before) if before > 0 else np.nan
            )
            absolute_rows.append(
                {
                    "Seed": seed,
                    "Parameter": symbol,
                    "Epoch": int(last.epoch),
                    "Lines": actual.size,
                    "Nominal RMSE (p.u.)": baseline_rmse,
                    "Final RMSE (p.u.)": final_rmse,
                    "RMSE reduction (p.u.)": baseline_rmse - final_rmse,
                    "Reduction (%)": reduction(baseline_rmse, final_rmse),
                }
            )
            # Normalize each line's signed error by its own real aged value.
            valid = np.abs(actual) > 1e-12
            baseline_relative = (
                np.sqrt(np.mean((baseline_error[valid] / actual[valid]) ** 2))
                if valid.any()
                else np.nan
            )
            final_relative = (
                np.sqrt(np.mean((final_error[valid] / actual[valid]) ** 2))
                if valid.any()
                else np.nan
            )
            relative_rows.append(
                {
                    "Seed": seed,
                    "Parameter": symbol,
                    "Lines": int(valid.sum()),
                    "Nominal relative RMSE (%)": 100 * baseline_relative,
                    "Final relative RMSE (%)": 100 * final_relative,
                    "Relative RMSE reduction (%)": reduction(
                        baseline_relative, final_relative
                    ),
                }
            )
    print(
        "\nImpedance errors against real aged values (eligible lines; final logged epoch):"
    )
    print(
        pd.DataFrame(absolute_rows).to_string(
            index=False, float_format=lambda v: f"{v:.6g}"
        )
    )
    print(
        "\nPer-line relative error = (impedance - real aged impedance) / real aged impedance."
    )
    print("Relative RMSE excludes real aged values with absolute magnitude <= 1e-12.")
    print(
        pd.DataFrame(relative_rows).to_string(
            index=False, float_format=lambda v: f"{v:.6g}"
        )
    )
    print(
        "Reduction (%) = 100 * (nominal error - final estimated error) / nominal error."
    )
    print(
        "Positive reduction indicates improvement; undefined zero-baseline reductions are NaN."
    )


def plot_tensorboard_impedance_rmse(case=118):
    """Overlay available seeds 0–4 for one case in separate R/X PDFs."""
    curves = {}
    for seed in (0, 1, 2, 3, 4):
        if not (experiment_directory(seed, case) / "denoised_admittance.pt").is_file():
            print(f"Skipping IEEE-{case}, seed {seed}: no saved impedance checkpoint")
            continue
        try:
            curves[seed] = load_tensorboard_impedance_rmse(seed, case)
        except FileNotFoundError as error:
            print(f"Skipping IEEE-{case}, seed {seed}: {error}")
    if not curves:
        print(f"No TensorBoard curves available for IEEE-{case}; skipping convergence plots.")
        return []
    seeds = tuple(curves)
    seed_label = "0-4" if seeds == (0, 1, 2, 3, 4) else "-".join(map(str, seeds))
    panels = (
        ("epoch/reactance_mismatch_rmse", "Reactance mismatch RMSE (p.u.)"),
        ("epoch/resistance_mismatch_rmse", "Resistance mismatch RMSE (p.u.)"),
    )
    output_root = results_root(case)
    output_root.mkdir(parents=True, exist_ok=True)
    figures = []
    for tag, ylabel in panels:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for seed in seeds:
            table = curves[seed][tag]
            ax.plot(
                table["epoch"],
                table["value"],
                color=f"C{seed}",
                linewidth=1.5,
                label=f"Seed {seed}",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        ax.margins(x=0)
        component = tag.split("/")[-1].removesuffix("_mismatch_rmse")
        output_path = output_root / f"{component}_mismatch_rmse_seeds{seed_label}_replot.pdf"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {output_path}")
        figures.append(fig)
    print_impedance_error_reduction(curves, case)
    return figures


def plot_impedance_parity(results, component, symbol, zoom_max=None):
    """Preserve the notebook parity plot and add a bottom-right near-zero inset.

    By default the inset covers the lowest quarter of lines by the larger of
    their actual/estimated impedance. Pass zoom_max to set its upper limit.
    """
    from matplotlib.ticker import MaxNLocator

    checkpoint = results.checkpoint
    mask = checkpoint["learnable_line_mask"].bool()
    aged = checkpoint[f"actual_aged_{component}_pu"][mask].detach().cpu().numpy()
    estimated = checkpoint[f"estimated_{component}_pu"][mask].detach().cpu().numpy()
    valid = np.isfinite(aged) & np.isfinite(estimated)
    if not valid.any():
        raise ValueError(f"No finite {symbol} impedance values to plot")
    aged, estimated = aged[valid], estimated[valid]
    lower = min(aged.min(), estimated.min())
    upper = max(aged.max(), estimated.max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-12)
        lower, upper = lower - padding, upper + padding

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(
        estimated, aged, s=22, color="tab:blue", alpha=0.8, label="Learnable lines"
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Perfect estimate (y = x)",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Estimated {symbol} (p.u.)")
    ax.set_ylabel(f"Aged ground-truth {symbol} (p.u.)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    if zoom_max is None:
        zoom_max = float(np.quantile(np.maximum(np.abs(aged), np.abs(estimated)), 0.25))
        if zoom_max <= 0:
            zoom_max = max(abs(lower), abs(upper), 1e-12) * 0.1
    if not np.isfinite(zoom_max) or zoom_max <= 0:
        raise ValueError("zoom_max must be finite and positive")
    inset = ax.inset_axes([0.56, 0.10, 0.39, 0.39])
    inset.scatter(estimated, aged, s=15, color="tab:blue", alpha=0.8)
    inset.plot(
        [0, zoom_max], [0, zoom_max], color="black", linestyle="--", linewidth=1.0
    )
    inset.set_xlim(0, zoom_max)
    inset.set_ylim(0, zoom_max)
    inset.set_aspect("equal", adjustable="box")
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.25)
    ax.indicate_inset_zoom(inset, edgecolor="grey", alpha=0.6)

    output_path = results.directory / f"{component}_estimated_vs_aged_replot.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}; near-zero inset: 0–{zoom_max:.6g} p.u.")
    return fig


def main(argv=None):
    args = parse_args(argv)
    results = load_experiment_results(args.seed, args.case)

    # Call each paper plotting method here; load the experiment only once.
    figure = plot_estimated_impedance_pf_error_histograms(results)
    plt.close(figure)
    print("\n" + "-" * 100 + "\n")
    figure = plot_nominal_vs_aged_pf_error_histograms(results)
    plt.close(figure)
    print("\n" + "-" * 100 + "\n")
    figure = plot_hidden_aging_error_histograms(results)
    plt.close(figure)
    print("\n" + "-" * 100 + "\n")
    for figure in plot_tensorboard_impedance_rmse(args.case):
        plt.close(figure)
    figure = plot_impedance_parity(results, "resistance", "R")
    plt.close(figure)
    figure = plot_impedance_parity(results, "reactance", "X")
    plt.close(figure)


if __name__ == "__main__":
    main()

"""Plot GENCO error distributions for matched and mismatched grid conditions.

Run from the repository root:
    venv/bin/python src/new_error_distribution.py --case 57 --seed 0

The matched figure compares nominal PF + nominal impedance with aged PF + aged
impedance.  The mismatched figure keeps the aged PF observations and targets,
but replaces the model's branch-admittance inputs with values reconstructed
from the unmodified nominal MATPOWER network.
A third figure overlays all three conditions using shared histogram bins.

Inputs:
    src/datakit_out_new/caseN/seedS/caseN_original/
    src/datakit_out_new/caseN/seedS/caseN_aged_seedS_k2p5_s40_d50_a60_logr/

Outputs are kept in a case/seed-specific directory below
``src/results_new/new_error_distribution`` and have condition-specific filenames.
PDFs and reports start with caseN-seedS-. Defaults to all cases and seed 0.
Each condition also exports one numeric .npy error array per variable (Vm, Va,
P, Q). Errors are prediction minus actual in p.u., degrees, MW, and MVAr,
respectively. Arrays include all errors in table row order, without histogram
clipping or filtering of non-finite values. For example:
    errors = np.load("src/results_new/new_error_distribution/case57/seed0/"
                     "matched_nominal_Vm_errors.npy", allow_pickle=False)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from gridfm_graphkit.datasets.globals import (
    B_ON,
    PG_H,
    PG_OUT,
    QG_H,
    QG_OUT,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
    YFF_TT_R,
    YFT_TF_I,
)
from gridfm_graphkit.io.param_handler import get_task_transforms

from error_evaluation import (
    DEFAULT_DATA_CACHE,
    DEVICE,
    HuggingFaceModel,
    _evaluate_pair,
    load_data_source,
    load_huggingface_model,
)
from aged_impedance_estimation import read_matpower_branch_rows
import resistance_analysis as resistance


MATCHED_NOMINAL = "Matched nominal"
MATCHED_AGED = "Matched aged"
MISMATCHED = "Mismatched"

# Keep these condition colours stable across all figures.
CONDITION_COLORS = {
    MATCHED_NOMINAL: "green",
    MATCHED_AGED: "orange",
    MISMATCHED: "red",
}

EDGE_TYPE = ("bus", "connects", "bus")
ADMITTANCE_COLUMNS = slice(YFF_TT_R, YFT_TF_I + 1)
VARIABLES = (("Vm", "p.u."), ("Va", "degrees"), ("P", "MW"), ("Q", "MVAr"))


def parse_args(argv=None):
    parser = resistance.parser(__doc__, single_seed_option=True)
    parser.add_argument("--recompute", action="store_true", help="Re-evaluate complete baselines")
    return resistance.select_cases_and_seeds(parser.parse_args(argv))


def experiment_paths(case, seed):
    """Return the two input datasets and the isolated output directory."""
    data_root = resistance.DATA / f"case{case}" / f"seed{seed}"
    return (
        data_root / f"case{case}_original",
        data_root / resistance.ageing_name(case, seed),
        resistance.BASELINES / f"case{case}" / f"seed{seed}",
    )


def model_spec(case):
    """Select the case-matched GENCO model while retaining training seed 0."""
    return HuggingFaceModel(
        "gridfm/genco-pf-datakit",
        f"case{case}_ieee/base/seed0",
        config_filename=f"case{case}_ieee_base_seed0.yaml",
        normalizer_network=f"case{case}_ieee",
    )


def nominal_admittance(case, normalizer):
    """Reconstruct normalized directed-edge Y from the nominal MATPOWER case."""
    case_path = resistance.GRIDS / f"case{case}_original.m"
    if not case_path.is_file():
        raise FileNotFoundError(f"Nominal MATPOWER network not found: {case_path}")
    rows = read_matpower_branch_rows(case_path)
    resistance = torch.tensor([row[2] for row in rows], dtype=torch.float32)
    reactance = torch.tensor([row[3] for row in rows], dtype=torch.float32)
    impedance = torch.complex(resistance, reactance)
    if (impedance.abs() == 0).any():
        raise ValueError(f"Zero branch impedance in {case_path}")

    charging = torch.tensor([row[4] for row in rows], dtype=torch.float32)
    tap_magnitude = torch.tensor(
        [row[8] if row[8] != 0.0 else 1.0 for row in rows], dtype=torch.float32
    )
    tap_shift = torch.deg2rad(
        torch.tensor([row[9] for row in rows], dtype=torch.float32)
    )
    status = torch.tensor([row[10] > 0.0 for row in rows], dtype=torch.float32)
    scale = float(normalizer.baseMVA_orig / normalizer.baseMVA)
    tap = torch.polar(tap_magnitude, tap_shift)
    series_y = status * scale / impedance
    half_charging_y = torch.complex(
        torch.zeros_like(charging), status * scale * charging / 2.0
    )
    self_y = series_y + half_charging_y
    forward = torch.stack(
        [
            (self_y / tap.abs().square()).real,
            (self_y / tap.abs().square()).imag,
            (-series_y / tap.conj()).real,
            (-series_y / tap.conj()).imag,
        ],
        dim=1,
    )
    reverse = torch.stack(
        [
            self_y.real,
            self_y.imag,
            (-series_y / tap).real,
            (-series_y / tap).imag,
        ],
        dim=1,
    )
    return torch.cat([forward, reverse])


def use_nominal_impedance(aged_graph, nominal_y):
    """Copy nominal Y inputs into an aged-PF graph after ordering checks.

    Branch PF/QF features, bus/generator PF observations, and ground-truth
    targets all remain aged.  Only the four directed-branch admittance columns
    derived from impedance are replaced.
    """
    aged_edges = aged_graph[EDGE_TYPE]
    if aged_edges.edge_index.size(1) != nominal_y.size(0):
        raise ValueError("MATPOWER branch count differs from the aged graph")
    n_lines = nominal_y.size(0) // 2
    if not torch.equal(
        aged_edges.edge_index[:, :n_lines],
        aged_edges.edge_index.flip(0)[:, n_lines:],
    ):
        raise ValueError("Forward and reverse branch ordering is not aligned")

    mismatched = aged_graph.clone()
    edge_attr = mismatched[EDGE_TYPE].edge_attr.clone()
    active = edge_attr[:, B_ON] > 0.5
    # Keep out-of-service branch Y at zero while making every active branch
    # use the unaged nominal-network value.
    edge_attr[:, ADMITTANCE_COLUMNS] = 0.0
    edge_attr[active, ADMITTANCE_COLUMNS] = nominal_y.to(edge_attr)[active]
    mismatched[EDGE_TYPE].edge_attr = edge_attr
    return mismatched


def prediction_frame(graph, output, dataset_index, label, normalizer):
    """Convert one normalized model result to the standard tidy error table."""
    prediction = output["bus"]
    if prediction.ndim != 2 or prediction.size(1) <= max(
        VM_OUT, VA_OUT, PG_OUT, QG_OUT
    ):
        raise ValueError("Model bus output does not contain Vm, Va, P, and Q")

    # These operations mutate the output and batch tensors in place.
    batch = graph
    normalizer.inverse_output(output, batch)
    normalizer.inverse_transform(batch)

    target = batch["bus"].y
    generator_target = batch["gen"].y
    generator_to_bus = batch["gen", "connected_to", "bus"].edge_index[1]
    bus_count = target.size(0)
    bus_index = np.arange(bus_count, dtype=np.int32)

    actual_p = torch.zeros(bus_count, device=DEVICE, dtype=generator_target.dtype)
    actual_p.index_add_(0, generator_to_bus, generator_target[:, 0])
    p_mask = torch.zeros(bus_count, device=DEVICE, dtype=generator_target.dtype)
    p_mask.index_add_(
        0,
        generator_to_bus,
        batch.mask_dict["gen"][:, PG_H].to(generator_target.dtype),
    )
    p_mask = p_mask > 0
    q_mask = batch.mask_dict["bus"][:, QG_H]

    values = (
        (
            "Vm",
            "p.u.",
            bus_index,
            target[:, VM_H],
            prediction[:, VM_OUT],
        ),
        (
            "Va",
            "degrees",
            bus_index,
            torch.rad2deg(target[:, VA_H]),
            torch.rad2deg(prediction[:, VA_OUT]),
        ),
        (
            "P",
            "MW",
            bus_index[p_mask.detach().cpu().numpy()],
            actual_p[p_mask],
            prediction[p_mask, PG_OUT],
        ),
        (
            "Q",
            "MVAr",
            bus_index[q_mask.detach().cpu().numpy()],
            target[q_mask, QG_H],
            prediction[q_mask, QG_OUT],
        ),
    )
    scenario_id = int(batch.scenario_id.item())
    frames = []
    for variable, unit, indices, actual, predicted in values:
        actual = actual.detach().cpu().numpy()
        predicted = predicted.detach().cpu().numpy()
        frames.append(
            pd.DataFrame(
                {
                    "pair": label,
                    "dataset_index": dataset_index,
                    "scenario_id": scenario_id,
                    "bus_index": indices,
                    "variable": variable,
                    "unit": unit,
                    "actual": actual,
                    "prediction": predicted,
                    "error": predicted - actual,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def evaluate_mismatched(
    aged_root,
    nominal_y,
    model_bundle,
):
    """Evaluate aged PF with nominal impedance for every aged scenario."""
    model, normalizer, args = model_bundle
    stream_partitions = getattr(args.data, "stream_partitions", "auto")
    aged_dataset = load_data_source(
        aged_root,
        normalizer,
        stream_partitions=stream_partitions,
        cache_dir=DEFAULT_DATA_CACHE,
    )
    aged_indices = sorted(
        int(path.stem.removeprefix("data_index_"))
        for path in Path(aged_dataset.processed_dir).glob("data_index_*.pt")
    )
    if not aged_indices:
        raise ValueError(f"{MISMATCHED} has no samples to evaluate")

    transform = get_task_transforms(args)
    frames = []
    with torch.inference_mode():
        for aged_index in tqdm(aged_indices, desc=f"Evaluating {MISMATCHED}"):
            aged_graph = aged_dataset[aged_index]
            graph = transform(use_nominal_impedance(aged_graph, nominal_y))
            batch = Batch.from_data_list([graph]).to(DEVICE)
            output = model(batch)
            frames.append(
                prediction_frame(batch, output, aged_index, MISMATCHED, normalizer)
            )
    return pd.concat(frames, ignore_index=True)


def save_numpy_errors(table, condition, output_dir):
    """Save separate numeric error arrays so variables with different units stay separate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable, _ in VARIABLES:
        errors = table.loc[table["variable"] == variable, "error"].to_numpy(
            dtype=np.float64
        )
        path = output_dir / f"{condition}_{variable}_errors.npy"
        np.save(path, errors, allow_pickle=False)
        print(f"Saved {path}")


def place_top_legend(axis, renderer):
    """Choose the top corner with the least overlap with the plotted paths."""
    paths = [
        artist.get_path().transformed(artist.get_transform())
        for artist in (*axis.patches, *axis.lines)
    ]
    scores = {}
    for location in ("upper right", "upper left"):
        legend = axis.legend(loc=location, fontsize=7)
        bounds = legend.get_window_extent(renderer)
        scores[location] = (
            any(bounds.overlaps(child.get_window_extent(renderer)) for child in axis.child_axes),
            sum(
                bounds.count_contains(path.vertices)
                + int(path.intersects_bbox(bounds, filled=False))
                for path in paths
            ),
        )
        legend.remove()
    axis.legend(loc=min(scores, key=scores.get), fontsize=7)


def plot_error_distributions(tables, output_path, *, colors=None):
    """Save a four-panel signed-error histogram and return its statistics."""
    errors_by_variable = {}
    for variable, _ in VARIABLES:
        errors = errors_by_variable[variable] = {}
        for label, table in tables.items():
            rows = table.loc[table["variable"] == variable]
            finite = np.isfinite(rows["actual"]) & np.isfinite(rows["error"])
            errors[label] = rows.loc[finite, "error"].to_numpy(dtype=float)
    return plot_error_arrays(errors_by_variable, output_path, colors=colors)


def plot_error_arrays(errors_by_variable, output_path, *, colors=None, decorate_axis=None):
    """Plot finite error arrays using shared bins for each variable."""
    if colors is None:
        colors = CONDITION_COLORS
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    statistics = []
    for axis, (variable, unit) in zip(axes.flat, VARIABLES):
        errors = errors_by_variable[variable]
        for label, values in errors.items():
            if values.size == 0:
                raise ValueError(f"No finite {variable} values for {label}")
            statistics.append(
                {
                    "Variable": variable,
                    "Condition": label,
                    "Unit": unit,
                    "Count": values.size,
                    "Mean": values.mean(),
                    "Std": values.std(ddof=0),
                }
            )

        if any(values.size < 3 for values in errors.values()):
            # With one or two errors, percentile trimming can exclude every point.
            lower = min(values.min() for values in errors.values())
            upper = max(values.max() for values in errors.values())
        else:
            lower = min(np.quantile(values, 0.005) for values in errors.values())
            upper = max(np.quantile(values, 0.995) for values in errors.values())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 101)
        for label, values in errors.items():
            counts, _ = np.histogram(values, bins=bins)
            if not counts.sum():
                raise ValueError(f"No displayed {variable} errors for {label}")
            axis.stairs(
                counts / counts.sum(),
                bins,
                color=colors[label],
                label=label,
                linewidth=1.5,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xlabel(f"{variable} prediction error ({unit})")
        axis.set_ylabel("Fraction of observations")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=7)
        if decorate_axis is not None:
            decorate_axis(axis, variable)

    # Resolve constrained layout before comparing legend boxes with the curves.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in axes.flat:
        place_top_legend(axis, renderer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")
    return pd.DataFrame(statistics)


def main(argv=None):
    from rx_common import evaluate_baselines, plot_groups, write_manifest

    options = parse_args(argv)
    models = {}
    for seed in options.seeds:
        for case in options.case:
            if case not in models:
                models[case] = load_huggingface_model(model_spec(case))
            nominal, aged, directory = experiment_paths(case, seed)
            tables = evaluate_baselines(
                case, models[case], nominal, aged, resistance.GRIDS, directory,
                recompute=options.recompute,
            )
            plot_groups(tables, directory, case, seed)
            write_manifest(
                directory / f"case{case}-seed{seed}-baseline_manifest.json",
                case=case, seed=seed, ageing=aged.name, nominal_dataset=str(nominal),
                aged_dataset=str(aged), all_eligible_scenarios=True,
            )
            print(f"Saved baseline analysis to {directory}")


if __name__ == "__main__":
    main()

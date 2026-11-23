"""Report resistance recovery, learning curves, and deployed PF errors.

Defaults to all IEEE cases and seed 0; --all-seeds selects seeds 0–4.
Reads the latest run per case/seed from tb_new and its checkpoint from
results_new. Measures recovery against the aged MATPOWER network and keeps
TensorBoard diagnostics in separate logged_* columns.

Outputs under results_new/new_resistance_error_reduction/AGEING_TAG/OBJECTIVE:
seed-labelled CSVs and curves, then per-case/seed/run parity plots, PF error
distributions, Parquet/NumPy exports and a checkpoint provenance manifest.
--skip-error-distributions writes the recovery CSV and curves without inference.
"""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import resistance_analysis as resistance

METRICS = {
    "loss/power_flow_mse": ("power_flow_mse", "Power-flow MSE (normalized)"),
    "epoch/impedance_mismatch_mae": ("impedance_mismatch_mae", "Resistance mismatch MAE (p.u.)"),
    "epoch/impedance_mismatch_rmse": ("impedance_mismatch_rmse", "Resistance mismatch RMSE (p.u.)"),
    "epoch/resistance_mismatch_rmse": ("resistance_mismatch_rmse", "Resistance mismatch RMSE (p.u.)"),
    "epoch/reactance_mismatch_rmse": ("reactance_mismatch_rmse", "Reactance mismatch RMSE (p.u.)"),
}
ESTIMATED_RESISTANCE = "Estimated Resistance"


def parse_args(argv=None):
    parser = resistance.parser(__doc__)
    parser.add_argument("--ageing-tag", default=resistance.TAG)
    parser.add_argument("--objective", choices=("pf_only", "kl_physics_informed"), default="pf_only")
    parser.add_argument("--log-dir", type=Path, default=resistance.TB)
    parser.add_argument("--data-root", type=Path, default=resistance.DATA)
    parser.add_argument("--grid-dir", type=Path, default=resistance.GRIDS)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=resistance.RESULTS,
        help="Root containing caseN/seedS/AGEING_CASE/OBJECTIVE/resistance_only/RUN checkpoints",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=resistance.BASELINES,
        help="Reuse existing baseline prediction Parquets for the default ageing tag",
    )
    parser.add_argument(
        "--skip-error-distributions", action="store_true",
        help="Generate recovery CSV and training curves without deployment inference",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=resistance.DEPLOYMENT,
    )
    options = parser.parse_args(argv)
    if Path(options.ageing_tag).name != options.ageing_tag or options.ageing_tag in {".", ".."}:
        parser.error("--ageing-tag must be a directory name, not a path")
    return resistance.select_cases_and_seeds(options)


def load_run(log_root, case, seed, ageing_tag, objective):
    """Read the latest resistance-only run, requiring a consistent scalar export."""
    experiment_dir = (
        Path(log_root) / f"case{case}" / f"seed{seed}"
        / ageing_tag / objective / "resistance_only"
    )
    run_dirs = sorted(
        {path.parent for path in experiment_dir.glob("*/events.out.tfevents.*")},
        key=lambda path: path.name, reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No resistance-only TensorBoard runs in {experiment_dir}")
    run_dir = run_dirs[0]
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    missing = set(METRICS) - set(accumulator.Tags().get("scalars", []))
    if missing:
        raise ValueError(f"Latest run {run_dir} is missing tags: {sorted(missing)}")

    curves = {}
    for tag in METRICS:
        table = pd.DataFrame([
            {"epoch": event.step, "value": event.value, "wall_time": event.wall_time}
            for event in accumulator.Scalars(tag)
        ])
        if table.empty:
            raise ValueError(f"Empty TensorBoard series: {run_dir}: {tag}")
        table = (
            table.sort_values("wall_time", kind="stable")
            .drop_duplicates("epoch", keep="last")
            .sort_values("epoch").reset_index(drop=True)
        )
        if not np.isfinite(table["value"]).all() or (table["value"] < 0).any():
            raise ValueError(f"Invalid error/loss values: {run_dir}: {tag}")
        if table.iloc[0]["epoch"] != 0 or len(table) < 2:
            raise ValueError(f"Need epoch 0 and a later epoch: {run_dir}: {tag}")
        curves[tag] = table
    reference_epochs = curves[next(iter(METRICS))]["epoch"]
    if any(not table["epoch"].equals(reference_epochs) for table in curves.values()):
        raise ValueError(f"Requested tags have different epochs: {run_dir}")
    return run_dir, curves


def recovery_row(case, seed, checkpoint_info, curves, options):
    from rx_common import recovery_components, recovery_metrics, reduction

    path, checkpoint = checkpoint_info
    components = recovery_components(
        options.grid_dir, case, checkpoint["ageing"], checkpoint, fixed_reactance=True,
    )
    reference = curves[next(iter(METRICS))]
    row = {
        "case": case, "seed": seed, "n_seeds": 1,
        "eligible_lines": len(components["resistance"][0]),
        "initial_epoch": int(reference.iloc[0]["epoch"]),
        "final_epoch": int(reference.iloc[-1]["epoch"]),
        **recovery_metrics(components),
    }
    for tag, (metric, _) in METRICS.items():
        initial = float(curves[tag].iloc[0]["value"])
        final = float(curves[tag].iloc[-1]["value"])
        row[f"logged_{metric}_initial"] = initial
        row[f"logged_{metric}_final"] = final
        row[f"logged_{metric}_reduction_pct"] = reduction(initial, final)
    row["checkpoint"] = str(path)
    return row


def case_mean_row(case, rows):
    """Average seed-level values, including reductions, rather than recomputing ratios."""
    table = pd.DataFrame(rows)
    numeric_columns = [column for column in table if column not in {"case", "seed", "n_seeds", "checkpoint"}]
    return {
        "case": case, "seed": "mean", "n_seeds": len(rows),
        **{column: table[column].mean() for column in numeric_columns},
        "checkpoint": "",
    }


def plot_case(case, runs, output_dir, seed_label):
    """Overlay seeds in separate title-free PDFs, with legends at the top."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tables = []
    for tag, (metric, ylabel) in METRICS.items():
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for seed, (_, curves) in runs.items():
            table = curves[tag]
            axis.plot(
                table["epoch"], table["value"], color=f"C{seed % 10}",
                linewidth=1.5, label=f"Seed {seed}",
            )
            raw_tables.append(table.assign(case=case, seed=seed, tag=tag))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
        axis.margins(x=0)
        path = output_dir / f"case{case}-{metric}_seeds{seed_label}.pdf"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path}")
    pd.concat(raw_tables, ignore_index=True).to_csv(
        output_dir / f"case{case}-tensorboard_curves_seeds{seed_label}.csv", index=False
    )


def load_final_checkpoint(case, seed, run_dir, curves, options):
    """Resolve the final saved R state belonging to this exact TensorBoard run."""
    import torch

    run_id = run_dir.name.removeprefix("resistance-multiplier-denoising-")
    ageing = f"case{case}_aged_seed{seed}_{options.ageing_tag}"
    path = run_dir / "denoised_admittance.pt"
    if not path.is_file():
        path = (
            options.checkpoint_dir / f"case{case}" / f"seed{seed}" / ageing
            / options.objective / "resistance_only" / run_id / "denoised_admittance.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Final checkpoint for TensorBoard run {run_dir} not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"case": case, "seed": seed, "ageing": ageing,
                "run_id": run_id, "estimation_target": "resistance"}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match selected run: {path}")
    if bool(checkpoint.get("kl_enabled", False)) != (options.objective == "kl_physics_informed"):
        raise ValueError(f"Checkpoint objective differs from selected run: {path}")
    last_epoch = int(curves["loss/power_flow_mse"].epoch.iloc[-1])
    if "epochs" in checkpoint and checkpoint["epochs"] != last_epoch:
        raise ValueError(f"Checkpoint is not from the final logged epoch: {path}")
    mask = checkpoint["learnable_line_mask"].bool()
    if not mask.any():
        raise ValueError(f"Checkpoint has no learnable lines: {path}")
    if not torch.equal(checkpoint["estimated_reactance_pu"], checkpoint["initial_reactance_pu"]):
        raise ValueError(f"Reactance changed in a resistance-only checkpoint: {path}")
    errors = (checkpoint["estimated_resistance_pu"] - checkpoint["actual_aged_resistance_pu"])[mask]
    final_metrics = {
        "epoch/impedance_mismatch_mae": errors.abs().mean().item(),
        "epoch/impedance_mismatch_rmse": errors.square().mean().sqrt().item(),
    }
    for tag, value in final_metrics.items():
        if not np.isfinite(value) or not np.isclose(
            value, curves[tag].iloc[-1]["value"], rtol=1e-5, atol=1e-9
        ):
            raise ValueError(f"Checkpoint resistance differs from the final logged epoch: {path}: {tag}")
    return path, checkpoint


def use_estimated_resistance(graph, checkpoint):
    """Deploy final estimated R on eligible active branches, retaining aged PF."""
    import torch
    import new_error_distribution as distributions

    edge_type = distributions.EDGE_TYPE
    if not torch.equal(graph[edge_type].edge_index, checkpoint["edge_index"]):
        raise ValueError("Aged graph branch ordering differs from the resistance checkpoint")
    graph = graph.clone()
    attributes = graph[edge_type].edge_attr.clone()
    mask = checkpoint["learnable_line_mask"].bool()
    directed_mask = torch.cat([mask, mask]).to(attributes.device)
    active = attributes[:, distributions.B_ON] > 0.5
    estimated_y = checkpoint["estimated_admittance_normalized"].to(attributes)
    if estimated_y.shape != attributes[:, distributions.ADMITTANCE_COLUMNS].shape:
        raise ValueError("Checkpoint admittance shape differs from the aged graph")
    attributes[directed_mask, distributions.ADMITTANCE_COLUMNS] = 0
    selected = directed_mask & active
    attributes[selected, distributions.ADMITTANCE_COLUMNS] = estimated_y[selected]
    graph[edge_type].edge_attr = attributes
    return graph


def evaluate_estimated_resistance(aged_root, checkpoint, model_bundle):
    """Predict aged PF targets using the final resistance-only state."""
    import torch
    from torch_geometric.data import Batch
    from tqdm.auto import tqdm
    import new_error_distribution as distributions

    model, normalizer, args = model_bundle
    dataset = distributions.load_data_source(
        aged_root, normalizer,
        stream_partitions=getattr(args.data, "stream_partitions", "auto"),
        cache_dir=distributions.DEFAULT_DATA_CACHE,
    )
    indices = sorted(
        int(path.stem.removeprefix("data_index_"))
        for path in Path(dataset.processed_dir).glob("data_index_*.pt")
    )
    if not indices:
        raise ValueError(f"No aged PF scenarios in {aged_root}")
    transform = distributions.get_task_transforms(args)
    frames = []
    with torch.inference_mode():
        for index in tqdm(indices, desc="Evaluating aged PF + estimated resistance"):
            graph = transform(use_estimated_resistance(dataset[index], checkpoint))
            batch = Batch.from_data_list([graph]).to(distributions.DEVICE)
            frames.append(distributions.prediction_frame(
                batch, model(batch), index, ESTIMATED_RESISTANCE, normalizer,
            ))
    return pd.concat(frames, ignore_index=True)


def validate_aged_prediction_rows(reference, candidate):
    """Ensure deployment comparisons use identical aged targets and scenario/bus rows."""
    keys = ["dataset_index", "scenario_id", "bus_index", "variable"]
    left = reference.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate scenario/bus/variable rows in aged PF predictions")
    if not left[keys].equals(right[keys]) or not np.allclose(
        left["actual"], right["actual"], rtol=1e-6, atol=1e-8, equal_nan=True
    ):
        raise ValueError("Aged PF scenario rows or targets differ; regenerate baseline predictions")


def plot_deployment_errors(case, seed, run_dir, checkpoint_info, model_bundle, options, output_dir):
    """Reproduce baseline plots and compare the effect of deploying final estimated R."""
    import torch
    import new_error_distribution as distributions
    import rx_common as analysis

    checkpoint_path, checkpoint = checkpoint_info
    _, normalizer, _ = model_bundle
    spec = distributions.model_spec(case)
    expected_artifacts = {"repo_id": spec.repo_id, "run_dir": spec.subfolder}
    if any(checkpoint["model_artifacts"].get(key) != value for key, value in expected_artifacts.items()):
        raise ValueError("Estimator and PF evaluator use different pretrained GENCO models")
    nominal_y = analysis.admittance(options.grid_dir / f"case{case}_original.m", normalizer)
    if not torch.allclose(nominal_y, checkpoint["initial_admittance_normalized"], rtol=1e-5, atol=1e-6):
        raise ValueError("Checkpoint nominal admittance differs from this network/normalizer")
    # Confirm the saved deployment inputs encode the saved final R, with X fixed.
    mask = checkpoint["learnable_line_mask"].bool()
    n_lines = mask.numel()
    saved_y = checkpoint["estimated_admittance_normalized"]
    cross = torch.complex(saved_y[:n_lines, 2], saved_y[:n_lines, 3])[mask]
    impedance = float(normalizer.baseMVA_orig / normalizer.baseMVA) / (-cross)
    if not torch.allclose(impedance.real, checkpoint["estimated_resistance_pu"][mask], rtol=1e-5, atol=1e-7):
        raise ValueError("Saved admittance does not encode the checkpoint's estimated resistance")
    if not torch.allclose(impedance.imag, checkpoint["initial_reactance_pu"][mask], rtol=1e-5, atol=1e-7):
        raise ValueError("Saved deployment inputs do not retain nominal reactance")

    data_root = options.data_root / f"case{case}" / f"seed{seed}"
    nominal_root = data_root / f"case{case}_original"
    aged_root = data_root / checkpoint["ageing"]
    cached_dir = options.baseline_dir / f"case{case}" / f"seed{seed}"
    tables = analysis.evaluate_baselines(
        case, model_bundle, nominal_root, aged_root, options.grid_dir, cached_dir,
        recompute=options.ageing_tag != resistance.TAG, save=False,
    )
    estimated = evaluate_estimated_resistance(aged_root, checkpoint, model_bundle)
    for condition in analysis.BASE_CONDITIONS[1:]:
        validate_aged_prediction_rows(tables[condition], estimated)
    estimated_condition = "aged_pf_estimated_resistance"
    tables[estimated_condition] = estimated
    output_dir = output_dir / f"seed{seed}" / checkpoint["run_id"]
    prefix = f"case{case}-seed{seed}-"
    analysis.save_errors(estimated, output_dir, estimated_condition, prefix)
    analysis.plot_groups(tables, output_dir, case, seed, estimated=True, target="resistance")
    components = analysis.recovery_components(
        options.grid_dir, case, checkpoint["ageing"], checkpoint, fixed_reactance=True,
    )
    analysis.plot_recovery_parity(case, seed, components, output_dir)
    analysis.write_manifest(
        output_dir / f"{prefix}deployment_manifest.json",
        case=case, seed=seed, run_id=checkpoint["run_id"], checkpoint=str(checkpoint_path),
        tensorboard_run=str(run_dir), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        all_eligible_scenarios=True, estimation_target="resistance",
        model=expected_artifacts, aged_dataset=str(aged_root), nominal_dataset=str(nominal_root),
        scenario_counts={condition: int(table.scenario_id.nunique()) for condition, table in tables.items()},
    )


def main(argv=None):
    options = parse_args(argv)
    # Validate all requested runs before writing reports or plotting a partial set.
    runs = {case: {} for case in options.case}
    checkpoints = {}
    for seed in options.seeds:
        for case in options.case:
            run_dir, curves = load_run(
                options.log_dir, case, seed, options.ageing_tag, options.objective
            )
            runs[case][seed] = (run_dir, curves)
            checkpoints[case, seed] = load_final_checkpoint(case, seed, run_dir, curves, options)
            print(f"IEEE-{case}, seed {seed}: {run_dir}")

    output_dir = options.output_dir / options.ageing_tag / options.objective
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "-".join(map(str, options.seeds))
    case_label = "-".join(map(str, options.case))
    rows = []
    for case, case_runs in runs.items():
        seed_rows = [
            recovery_row(case, seed, checkpoints[case, seed], curves, options)
            for seed, (_, curves) in case_runs.items()
        ]
        rows.extend(seed_rows)
        rows.append(case_mean_row(case, seed_rows))
        plot_case(case, case_runs, output_dir / f"case{case}", seed_label)

    table = pd.DataFrame(rows)
    path = output_dir / f"case{case_label}-resistance_error_reduction_seeds{seed_label}.csv"
    table.to_csv(path, index=False, na_rep="NaN")
    print(f"Saved {path}")
    if not options.skip_error_distributions:
        import new_error_distribution as distributions

        model_bundles = {}
        for seed in options.seeds:
            for case in options.case:
                run_dir, _ = runs[case][seed]
                if case not in model_bundles:
                    model_bundles[case] = distributions.load_huggingface_model(distributions.model_spec(case))
                print(f"Evaluating all eligible scenarios: IEEE-{case}, seed {seed}")
                plot_deployment_errors(
                    case, seed, run_dir, checkpoints[case, seed], model_bundles[case],
                    options, output_dir / f"case{case}",
                )
    return table


if __name__ == "__main__":
    main()

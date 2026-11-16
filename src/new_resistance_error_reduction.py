"""Report resistance-only estimation improvements and plot TensorBoard curves.

Run all IEEE cases and seeds 0-4 from any working directory:
    venv/bin/python src/new_resistance_error_reduction.py
Select a subset:
    venv/bin/python src/new_resistance_error_reduction.py --case 57 --seeds 0 1

Reads the latest run in each exact TensorBoard experiment directory:
    src/tb/caseN/seedS/AGEING_TAG/OBJECTIVE/resistance_only/RUN/
Defaults: AGEING_TAG=k2p5_s40_d50_a60_logr, OBJECTIVE=pf_only.
Joint R/X runs and other objectives are never mixed into the selected runs.

Outputs below src/results/new_resistance_error_reduction/AGEING_TAG/OBJECTIVE/:
    caseN-...-resistance_error_reduction_seedsS-....csv
    caseN/caseN-*_seedsS-....pdf (one plot per requested metric)
    caseN/caseN-tensorboard_curves_seedsS-....csv (all plotted scalar points)
    caseN/seedS/RUN/caseN-seedS-*_prediction_error_distribution.pdf
    caseN/seedS/RUN/caseN-seedS-aged_pf_estimated_resistance_* (Parquet/NumPy errors)

PF distributions reuse available baseline Parquets from new_error_distribution
and evaluate aged PF using the final resistance-only checkpoint matching each
TensorBoard run ID. Five distributions are saved per seed: the three previous
plots, nominal versus estimated resistance on aged PF, and all four conditions
together. Colors: nominal green, aged yellow, mismatched red, estimated blue.
Legends choose the upper-left or upper-right corner. A manifest records the
checkpoint path and SHA256. --skip-error-distributions retains the log-only
workflow. Every eligible scenario is evaluated, with no sample cap. Processing
finishes all selected cases for seed 0 before seed 1, and so forth.

The main CSV has one row per seed and a mean row at the end of each case.
Initial means epoch 0; final means the last logged epoch, not the best epoch.
Reduction (%) = 100 * (initial - final) / initial. Zero initial values give
NaN percentages. Mean rows average the individual seed values, including
the per-seed percentage reductions, with equal weight and excluding NaNs.

Resistance MAE and RMSE are in p.u.; power-flow MSE is the normalized training
loss. Only R is estimated in these runs, so the impedance-mismatch tags describe
resistance errors. Curves use every logged point without smoothing. Resumed
runs keep the most recent scalar at each step. All three tags must have the
same epochs, including epoch 0 and at least one later epoch.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_DIR = Path(__file__).resolve().parent
CASES = (14, 30, 57, 118)
METRICS = {
    "loss/power_flow_mse": ("power_flow_mse", "Power-flow MSE (normalized)"),
    "epoch/impedance_mismatch_mae": ("resistance_mae", "Resistance mismatch MAE (p.u.)"),
    "epoch/impedance_mismatch_rmse": ("resistance_rmse", "Resistance mismatch RMSE (p.u.)"),
}
ESTIMATED_RESISTANCE = "Estimated Resistance"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--ageing-tag", default="k2p5_s40_d50_a60_logr")
    parser.add_argument("--objective", choices=("pf_only", "kl_physics_informed"), default="pf_only")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_DIR / "tb")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_DIR / "results",
        help="Root containing caseN/seedS/AGEING_CASE/OBJECTIVE/resistance_only/RUN checkpoints",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=PROJECT_DIR / "results" / "new_error_distribution",
        help="Reuse existing baseline prediction Parquets for the default ageing tag",
    )
    parser.add_argument(
        "--skip-error-distributions", action="store_true",
        help="Generate only the TensorBoard report and training curves",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "results" / "new_resistance_error_reduction",
    )
    options = parser.parse_args(argv)
    if any(not 0 <= seed < 2**32 for seed in options.seeds):
        parser.error("--seeds must be between 0 and 2**32 - 1")
    if Path(options.ageing_tag).name != options.ageing_tag or options.ageing_tag in {".", ".."}:
        parser.error("--ageing-tag must be a directory name, not a path")
    options.case = sorted(set(options.case))
    options.seeds = sorted(set(options.seeds))
    return options


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


def improvement_row(case, seed, run_dir, curves):
    reference = curves[next(iter(METRICS))]
    row = {
        "case": case, "seed": seed, "n_seeds": 1,
        "initial_epoch": int(reference.iloc[0]["epoch"]),
        "final_epoch": int(reference.iloc[-1]["epoch"]),
    }
    for tag, (metric, _) in METRICS.items():
        initial = float(curves[tag].iloc[0]["value"])
        final = float(curves[tag].iloc[-1]["value"])
        row[f"{metric}_initial"] = initial
        row[f"{metric}_final"] = final
        row[f"{metric}_reduction_pct"] = 100 * (1 - final / initial) if initial > 0 else np.nan
    row["run_dir"] = str(run_dir)
    return row


def case_mean_row(case, rows):
    """Average seed-level values, including reductions, rather than recomputing ratios."""
    table = pd.DataFrame(rows)
    numeric_columns = [column for column in table if column not in {"case", "seed", "n_seeds", "run_dir"}]
    return {
        "case": case, "seed": "mean", "n_seeds": len(rows),
        **{column: table[column].mean() for column in numeric_columns},
        "run_dir": "",
    }


def plot_case(case, runs, output_dir, seed_label):
    """Overlay seeds in three separate title-free PDFs, with legends at the top."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_tables = []
    for tag, (metric, ylabel) in METRICS.items():
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for seed, (run_dir, curves) in runs.items():
            table = curves[tag]
            axis.plot(
                table["epoch"], table["value"], color=f"C{seed % 10}",
                linewidth=1.5, label=f"Seed {seed}",
            )
            raw_tables.append(table.assign(case=case, seed=seed, tag=tag, run_dir=str(run_dir)))
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

    checkpoint_path, checkpoint = checkpoint_info
    _, normalizer, _ = model_bundle
    spec = distributions.model_spec(case)
    expected_artifacts = {"repo_id": spec.repo_id, "run_dir": spec.subfolder}
    if any(checkpoint["model_artifacts"].get(key) != value for key, value in expected_artifacts.items()):
        raise ValueError("Estimator and PF evaluator use different pretrained GENCO models")
    nominal_y = distributions.nominal_admittance(case, normalizer)
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

    data_root = PROJECT_DIR / "datakit_out" / f"case{case}" / f"seed{seed}"
    nominal_root = data_root / f"case{case}_original"
    aged_root = Path(checkpoint["dataset_roots"][f"case{case}_aged"]).expanduser()
    if not aged_root.is_absolute():
        aged_root = PROJECT_DIR / aged_root
    cached_dir = options.baseline_dir / f"case{case}" / f"seed{seed}"
    tables = {}
    for condition, label, source in (
        ("matched_nominal", distributions.MATCHED_NOMINAL, nominal_root),
        ("matched_aged", distributions.MATCHED_AGED, aged_root),
        ("mismatched_aged_pf_nominal_impedance", distributions.MISMATCHED, aged_root),
    ):
        cached_path = cached_dir / f"{condition}_prediction_errors.parquet"
        can_reuse_baseline = (
            options.ageing_tag == "k2p5_s40_d50_a60_logr"
            and aged_root.resolve() == (data_root / checkpoint["ageing"]).resolve()
        )
        if can_reuse_baseline and cached_path.is_file():
            table = pd.read_parquet(cached_path)
            dataset = distributions.load_data_source(
                source, normalizer,
                stream_partitions=getattr(model_bundle[2].data, "stream_partitions", "auto"),
                cache_dir=distributions.DEFAULT_DATA_CACHE,
            )
            eligible_indices = {
                int(path.stem.removeprefix("data_index_"))
                for path in Path(dataset.processed_dir).glob("data_index_*.pt")
            }
            can_reuse_baseline = bool(eligible_indices) and set(table["dataset_index"].unique()) == eligible_indices
            if can_reuse_baseline:
                print(f"Reusing complete baseline {cached_path} ({len(eligible_indices):,} scenarios)")
            else:
                print(f"Re-evaluating incomplete baseline {cached_path}")
        else:
            can_reuse_baseline = False
        if can_reuse_baseline:
            tables[label] = table
            continue
        if label == distributions.MISMATCHED:
            table = distributions.evaluate_mismatched(
                aged_root, nominal_y, model_bundle,
            )
        else:
            table = distributions._evaluate_pair(
                (source, spec), label, max_samples=None,
                data_cache_dir=distributions.DEFAULT_DATA_CACHE, model_bundle=model_bundle,
            )
        tables[label] = table
    estimated = evaluate_estimated_resistance(aged_root, checkpoint, model_bundle)
    validate_aged_prediction_rows(tables[distributions.MATCHED_AGED], estimated)
    validate_aged_prediction_rows(tables[distributions.MISMATCHED], estimated)
    tables[ESTIMATED_RESISTANCE] = estimated
    output_dir = output_dir / f"seed{seed}" / checkpoint["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"case{case}-seed{seed}"
    estimated.to_parquet(output_dir / f"{prefix}-aged_pf_estimated_resistance_prediction_errors.parquet", index=False)
    distributions.save_numpy_errors(estimated, f"{prefix}-aged_pf_estimated_resistance", output_dir)
    colors = {**distributions.CONDITION_COLORS, ESTIMATED_RESISTANCE: "blue"}
    groups = (
        ("matched_nominal_and_aged", (distributions.MATCHED_NOMINAL, distributions.MATCHED_AGED)),
        ("mismatched_aged_pf_nominal_impedance", (distributions.MISMATCHED,)),
        ("matched_and_mismatched", tuple(tables)[:3]),
        ("nominal_vs_estimated_resistance_aged_pf", (distributions.MISMATCHED, ESTIMATED_RESISTANCE)),
        ("all_conditions_with_estimated_resistance", tuple(tables)),
    )
    for name, labels in groups:
        statistics = distributions.plot_error_distributions(
            {label: tables[label] for label in labels},
            output_dir / f"{prefix}-{name}_prediction_error_distribution.pdf", colors=colors,
        )
    statistics.to_csv(output_dir / f"{prefix}-pf_error_statistics.csv", index=False)
    (output_dir / f"{prefix}-deployment_manifest.json").write_text(json.dumps({
        "case": case, "seed": seed, "tensorboard_run": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "run_id": checkpoint["run_id"], "all_eligible_scenarios": True,
        "reactance": "fixed at nominal", "model": expected_artifacts,
        "aged_dataset": str(aged_root), "nominal_dataset": str(nominal_root),
        "scenario_counts": {label: int(table["scenario_id"].nunique()) for label, table in tables.items()},
    }, indent=2) + "\n")


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
            if not options.skip_error_distributions:
                checkpoints[case, seed] = load_final_checkpoint(
                    case, seed, run_dir, curves, options,
                )
            print(f"IEEE-{case}, seed {seed}: {run_dir}")

    output_dir = options.output_dir / options.ageing_tag / options.objective
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "-".join(map(str, options.seeds))
    case_label = "-".join(map(str, options.case))
    rows = []
    for case, case_runs in runs.items():
        seed_rows = [
            improvement_row(case, seed, run_dir, curves)
            for seed, (run_dir, curves) in case_runs.items()
        ]
        rows.extend(seed_rows)
        rows.append(case_mean_row(case, seed_rows))
        plot_case(case, case_runs, output_dir / f"case{case}", seed_label)

    table = pd.DataFrame(rows)
    path = output_dir / f"case{case_label}-resistance_error_reduction_seeds{seed_label}.csv"
    table.to_csv(path, index=False, na_rep="NaN")
    print(f"Saved {path}")
    print(table.drop(columns="run_dir").to_string(index=False))
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

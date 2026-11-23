"""Report R/X recovery, training curves and deployment PF errors; seed 0 by default.

Use --all-seeds for seeds 0-4. Outputs are isolated under results_rx/.
Recovery percentages compare the nominal/final estimates with the actual aged
MATPOWER R/X, not the admittances averaged over perturbed PF scenarios. Logged
diagnostics are separately checked against the matching final checkpoint.
"""

import hashlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import rx_common as rx
from new_resistance_error_reduction import validate_aged_prediction_rows

METRICS = {
    "loss/power_flow_mse": ("power_flow_mse", "Power-flow MSE (normalized)"),
    "epoch/impedance_mismatch_mae": ("impedance_mismatch_mae", "Combined R/X mismatch MAE (p.u.)"),
    "epoch/impedance_mismatch_rmse": ("impedance_mismatch_rmse", "Combined R/X mismatch RMSE (p.u.)"),
    "epoch/resistance_mismatch_rmse": ("resistance_mismatch_rmse", "Resistance mismatch RMSE (p.u.)"),
    "epoch/reactance_mismatch_rmse": ("reactance_mismatch_rmse", "Reactance mismatch RMSE (p.u.)"),
}


def load_run(case, seed):
    root = rx.TB / f"case{case}" / f"seed{seed}" / rx.TAG / "pf_only"
    directories = sorted({path.parent for path in root.glob("*/events.out.tfevents.*")}, reverse=True)
    if not directories:
        raise FileNotFoundError(f"No joint RX logs in {root}; run run_all_rx_estimation.sh first")
    run = directories[0]
    reader = EventAccumulator(str(run), size_guidance={"scalars": 0})
    reader.Reload()
    missing = set(METRICS) - set(reader.Tags().get("scalars", []))
    if missing:
        raise ValueError(f"Latest RX run is missing tags {sorted(missing)}: {run}")
    curves = {}
    for tag in METRICS:
        table = pd.DataFrame([{"epoch": event.step, "value": event.value, "wall_time": event.wall_time}
                              for event in reader.Scalars(tag)])
        if table.empty:
            raise ValueError(f"No scalar values for {tag}: {run}")
        table = (table.sort_values("wall_time", kind="stable").drop_duplicates("epoch", keep="last")
                 .sort_values("epoch").reset_index(drop=True))
        if len(table) < 2 or table.epoch.iloc[0] != 0 or not np.isfinite(table.value).all() or (table.value < 0).any():
            raise ValueError(f"Invalid or incomplete learning curve {tag}: {run}")
        curves[tag] = table
    reference = curves[next(iter(METRICS))].epoch
    if any(not table.epoch.equals(reference) for table in curves.values()):
        raise ValueError(f"RX learning tags cover different epochs: {run}")
    return run, curves


def load_checkpoint(case, seed, run, curves):
    run_id = run.name.removeprefix("impedance-multiplier-denoising-")
    path = (rx.RESULTS / f"case{case}" / f"seed{seed}" / rx.aged_name(case, seed)
            / "pf_only" / run_id / "denoised_admittance.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint for selected RX log: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in {"case": case, "seed": seed, "run_id": run_id,
                          "ageing": rx.aged_name(case, seed), "estimation_target": "impedance"}.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Checkpoint {key} does not match the RX run: {path}")
    if checkpoint.get("epochs") != int(curves["loss/power_flow_mse"].epoch.iloc[-1]):
        raise ValueError(f"Checkpoint is not from the last logged epoch: {path}")
    if checkpoint.get("kl_enabled", False):
        raise ValueError("The RX comparison uses PF-only estimation, matching the resistance-only baseline")
    mask = checkpoint["learnable_line_mask"].bool()
    if not mask.any():
        raise ValueError("No eligible RX lines in checkpoint")
    errors = []
    for component in ("resistance", "reactance"):
        error = (checkpoint[f"estimated_{component}_pu"] - checkpoint[f"actual_aged_{component}_pu"])[mask]
        expected = curves[f"epoch/{component}_mismatch_rmse"].value.iloc[-1]
        if not np.isclose(error.square().mean().sqrt().item(), expected, rtol=1e-5, atol=1e-9):
            raise ValueError(f"Checkpoint {component} does not match final logged diagnostics")
        errors.append(error)
    combined = torch.cat(errors)
    for tag, value in (("epoch/impedance_mismatch_mae", combined.abs().mean().item()),
                       ("epoch/impedance_mismatch_rmse", combined.square().mean().sqrt().item())):
        if not np.isclose(value, curves[tag].value.iloc[-1], rtol=1e-5, atol=1e-9):
            raise ValueError(f"Checkpoint does not match final logged {tag}")
    return path, checkpoint


def recovery_row(case, seed, checkpoint, curves):
    """Measure recovery against actual MATPOWER parameters."""
    components = rx.recovery_components(rx.GRIDS, case, rx.aged_name(case, seed), checkpoint)
    result = {"case": case, "seed": seed, "n_seeds": 1,
              "eligible_lines": len(components["resistance"][0]), "initial_epoch": 0,
              "final_epoch": int(curves["loss/power_flow_mse"].epoch.iloc[-1]),
              **rx.recovery_metrics(components)}
    for tag, (metric, _) in METRICS.items():
        before, after = float(curves[tag].value.iloc[0]), float(curves[tag].value.iloc[-1])
        result.update({f"logged_{metric}_initial": before, f"logged_{metric}_final": after,
                       f"logged_{metric}_reduction_pct": rx.reduction(before, after)})
    return result


def plot_learning(case, runs, directory, seed_label):
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for tag, (metric, ylabel) in METRICS.items():
        figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for seed, (_, curves, _, _) in runs.items():
            table = curves[tag]
            axis.plot(table.epoch, table.value, label=f"Seed {seed}", color=f"C{seed}")
            frames.append(table.assign(case=case, seed=seed, tag=tag))
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend(loc="upper right")
        axis.margins(x=0)
        figure.savefig(directory / f"case{case}-{metric}_seeds{seed_label}.pdf", dpi=300)
        plt.close(figure)
    pd.concat(frames, ignore_index=True).to_csv(directory / f"case{case}-tensorboard_curves_seeds{seed_label}.csv", index=False)


def plot_parity(case, seed, checkpoint, directory):
    components = rx.recovery_components(rx.GRIDS, case, rx.aged_name(case, seed), checkpoint)
    rx.plot_recovery_parity(case, seed, components, directory)


def deployment(case, seed, run, checkpoint_path, checkpoint, bundle):
    nominal_y = rx.admittance(rx.GRIDS / f"case{case}_original.m", bundle[1])
    if not torch.allclose(nominal_y, checkpoint["initial_admittance_normalized"], rtol=1e-5, atol=1e-6):
        raise ValueError("Checkpoint uses a different nominal network or normalizer")
    expected_model = checkpoint["model_artifacts"]
    if expected_model["run_dir"] != rx.pf.model_spec(case).subfolder:
        raise ValueError("Estimator and PF evaluator use different GENCO checkpoints")
    mask = checkpoint["learnable_line_mask"].bool()
    saved = checkpoint["estimated_admittance_normalized"][:mask.numel()]
    cross = torch.complex(saved[:, 2], saved[:, 3])[mask]
    impedance = float(bundle[1].baseMVA_orig / bundle[1].baseMVA) / (-cross)
    for component, values in (("resistance", impedance.real), ("reactance", impedance.imag)):
        if not torch.allclose(values, checkpoint[f"estimated_{component}_pu"][mask], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Saved deployment inputs do not encode final estimated {component}")
    tables = rx.baseline_tables(case, seed, bundle)
    _, aged = rx.dataset_roots(case, seed)
    estimated = rx.evaluate_deployed(rx.load_dataset(aged, bundle), checkpoint, bundle)
    for condition in ("matched_aged", "mismatched_aged_pf_nominal_impedance"):
        validate_aged_prediction_rows(tables[condition], estimated)
    tables[rx.ESTIMATED] = estimated
    directory = rx.DEPLOYMENT / f"case{case}" / f"seed{seed}" / checkpoint["run_id"]
    prefix = f"case{case}-seed{seed}-"
    rx.save_errors(estimated, directory, rx.ESTIMATED, prefix)
    rx.plot_groups(tables, directory, case, seed, estimated=True)
    plot_parity(case, seed, checkpoint, directory)
    rx.write_manifest(directory / f"{prefix}deployment_manifest.json", case=case, seed=seed,
                      run_id=checkpoint["run_id"], checkpoint=str(checkpoint_path), tensorboard_run=str(run),
                      checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      all_eligible_scenarios=True, estimation_target="impedance",
                      scenario_counts={name: int(table.scenario_id.nunique()) for name, table in tables.items()})


def main(argv=None):
    parser = rx.parser(__doc__)
    parser.add_argument("--skip-error-distributions", action="store_true")
    options = parser.parse_args(argv)
    cases, seeds = rx.selected(options)
    runs = {case: {} for case in cases}
    rows = []
    for seed in seeds:
        for case in cases:
            run, curves = load_run(case, seed)
            checkpoint_path, checkpoint = load_checkpoint(case, seed, run, curves)
            runs[case][seed] = (run, curves, checkpoint_path, checkpoint)
            row = recovery_row(case, seed, checkpoint, curves)
            row["checkpoint"] = str(checkpoint_path)
            rows.append(row)
    seed_label = "-".join(map(str, seeds))
    report_rows = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        report_rows.extend(case_rows)
        frame = pd.DataFrame(case_rows)
        mean = {column: frame[column].mean() for column in frame.select_dtypes(include="number")
                if column not in {"case", "seed", "n_seeds"}}
        report_rows.append({**mean, "case": case, "seed": "mean", "n_seeds": len(seeds), "checkpoint": ""})
        plot_learning(case, runs[case], rx.DEPLOYMENT / f"case{case}", seed_label)
    path = rx.DEPLOYMENT / f"case{'-'.join(map(str, cases))}-impedance_error_reduction_seeds{seed_label}.csv"
    pd.DataFrame(report_rows).to_csv(path, index=False, na_rep="NaN")
    print(f"Saved {path}")
    if not options.skip_error_distributions:
        models = {}
        for seed in seeds:
            for case in cases:
                print(f"RX deployment: case{case}, seed{seed}; all eligible scenarios")
                if case not in models:
                    models[case] = rx.pf.load_huggingface_model(rx.pf.model_spec(case))
                run, _, checkpoint_path, checkpoint = runs[case][seed]
                deployment(case, seed, run, checkpoint_path, checkpoint, models[case])


if __name__ == "__main__":
    main()

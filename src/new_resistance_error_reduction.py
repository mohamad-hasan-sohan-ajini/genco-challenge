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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--ageing-tag", default="k2p5_s40_d50_a60_logr")
    parser.add_argument("--objective", choices=("pf_only", "kl_physics_informed"), default="pf_only")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_DIR / "tb")
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


def main(argv=None):
    options = parse_args(argv)
    # Validate all requested runs before writing reports or plotting a partial set.
    runs = {}
    for case in options.case:
        runs[case] = {}
        for seed in options.seeds:
            run_dir, curves = load_run(
                options.log_dir, case, seed, options.ageing_tag, options.objective
            )
            runs[case][seed] = (run_dir, curves)
            print(f"IEEE-{case}, seed {seed}: {run_dir}")

    output_dir = options.output_dir / options.ageing_tag / options.objective
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
    return table


if __name__ == "__main__":
    main()

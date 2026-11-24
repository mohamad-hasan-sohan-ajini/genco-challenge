"""Replot the resistance-only overlays with zoom insets from inset_config.yaml.

Run from the repository root with venv/bin/python src/insets.py.
Defaults to all four IEEE cases and seed 0. Outputs are written alongside the
deployment PDFs in the latest run directory for each selected case and seed.
Each configured panel uses the same inset bounds for both overlays. Panels
omitted from the YAML, such as IEEE-118 Va and P, have no inset.
"""

from functools import partial
from pathlib import Path

import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator

import new_error_distribution as plots
import new_statistical_tests as errors
import resistance_analysis as resistance


ESTIMATED_LABEL = "Estimated Resistance"
LABELS = dict(zip(errors.CONDITIONS, (
    plots.MATCHED_NOMINAL, plots.MATCHED_AGED, plots.MISMATCHED, ESTIMATED_LABEL,
)))
OVERLAYS = (
    ("matched_and_mismatched", errors.BASELINE_CONDITIONS),
    ("all_conditions_with_estimated_resistance", errors.CONDITIONS),
)


def add_inset(axis, variable, *, panels):
    panel = panels.get(variable)
    if panel is None:
        return

    left, right = sorted(panel["inset"]["x"])
    bottom, top = sorted(panel["inset"]["y"])
    xmin, xmax = panel["zoom"]["x"]
    ymin, ymax = panel["zoom"]["y"]
    if (not np.isfinite([left, top, right, bottom, xmin, xmax, ymin, ymax]).all()
            or not (left < right and bottom < top and xmin < xmax and ymin < ymax)):
        raise ValueError(f"Invalid inset bounds for {variable}")

    # The config uses parent data coordinates. Include the box in the parent
    # limits if a selected seed has a different histogram range.
    x0, x1 = axis.get_xlim()
    y0, y1 = axis.get_ylim()
    axis.set_xlim(min(x0, left), max(x1, right))
    axis.set_ylim(min(y0, bottom), max(y1, top))
    inset = axis.inset_axes([left, bottom, right - left, top - bottom], transform=axis.transData)
    inset.set_in_layout(False)
    for curve in axis.patches:
        values, edges, baseline = curve.get_data()
        inset.stairs(values, edges, baseline=baseline,
                     color=curve.get_edgecolor(), linewidth=curve.get_linewidth())
    inset.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    inset.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if left + right < x0 + x1:
        inset.yaxis.tick_right()
    inset.tick_params(labelsize=6, length=2, pad=1)
    inset.grid(alpha=0.2)
    axis.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.5, linewidth=0.7)


def load_arrays(case, seed, baseline_dir, run_dir):
    prefix = f"case{case}-seed{seed}-"
    arrays = {}
    for variable, _ in plots.VARIABLES:
        arrays[variable] = {}
        counts = {}
        for condition in errors.CONDITIONS:
            filename = f"{condition}_{variable}_errors.npy"
            path = (run_dir / f"{prefix}{filename}" if condition == errors.ESTIMATED_CONDITION
                    else baseline_dir / filename)
            values, excluded = errors.load_errors(path)
            arrays[variable][condition] = values
            counts[condition] = values.size + excluded
        if counts[errors.ESTIMATED_CONDITION] != counts["matched_aged"]:
            raise ValueError(f"Matched-aged and estimated {variable} error counts differ for {run_dir}")
    return arrays


def main(argv=None):
    parser = resistance.parser(__doc__)
    parser.add_argument("--config", type=Path, default=resistance.ROOT / "inset_config.yaml")
    parser.add_argument("--baseline-dir", type=Path, default=resistance.BASELINES)
    parser.add_argument("--estimated-dir", type=Path,
                        default=resistance.DEPLOYMENT / resistance.TAG / "pf_only")
    options = resistance.select_cases_and_seeds(parser.parse_args(argv))
    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    colors = {**plots.CONDITION_COLORS, ESTIMATED_LABEL: "blue"}

    for case in options.case:
        panels = config[f"case{case}"]
        baselines = errors.discover_seed_directories(options.baseline_dir / f"case{case}", options.seeds)
        runs, _ = errors.discover_estimated_directories(options.estimated_dir, case, options.seeds)
        if missing := sorted(baselines.keys() - runs.keys()):
            raise FileNotFoundError(f"No complete resistance deployment for case{case}, seeds {missing}")
        for seed in options.seeds:
            run = runs[seed]
            arrays = load_arrays(case, seed, baselines[seed], run)
            for name, conditions in OVERLAYS:
                data = {
                    variable: {LABELS[condition]: values[condition] for condition in conditions}
                    for variable, values in arrays.items()
                }
                output = run / f"case{case}-seed{seed}-{name}_prediction_error_distribution_insets.pdf"
                plots.plot_error_arrays(
                    data, output, colors=colors,
                    decorate_axis=partial(add_inset, panels=panels),
                )


if __name__ == "__main__":
    main()

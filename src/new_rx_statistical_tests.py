"""Mean signed error, zero-centered skewness, and all six same-seed Wasserstein-1 pairs.

RX data only; all four cases and seed 0 by default. --all-seeds selects 0-4.
Reads full error exports, performs no inference, and writes per-seed and
aggregated CSVs. Seeds receive equal weight. Sample std uses ddof=1 and is NaN
with one seed. No formatted mean +/- std column is generated.
"""

from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import new_statistical_tests as statistics
import rx_common as rx

CONDITIONS = (*rx.BASE_CONDITIONS, rx.ESTIMATED)


def error_sources(case, seeds):
    sources, records = {}, []
    for seed in seeds:
        baseline = rx.BASELINES / f"case{case}" / f"seed{seed}"
        for condition in rx.BASE_CONDITIONS:
            for variable, _ in rx.pf.VARIABLES:
                path = baseline / f"{condition}_{variable}_errors.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing RX baseline {path}; run new_rx_error_distribution.py first")
                sources[condition, seed, variable] = path
        runs = sorted(path for path in (rx.DEPLOYMENT / f"case{case}" / f"seed{seed}").glob("run-*") if path.is_dir())
        if not runs:
            raise FileNotFoundError(f"No RX deployment exports for case{case}, seed{seed}; run new_rx_impedance_error_reduction.py")
        run = runs[-1]
        prefix = f"case{case}-seed{seed}-"
        manifest_path = run / f"{prefix}deployment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Latest RX deployment has not completed: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        expected = {"case": case, "seed": seed, "run_id": run.name,
                    "all_eligible_scenarios": True, "estimation_target": "impedance"}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Invalid RX deployment provenance: {manifest_path}")
        for variable, _ in rx.pf.VARIABLES:
            path = run / f"{prefix}{rx.ESTIMATED}_{variable}_errors.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete RX estimated-error export: {path}")
            sources[rx.ESTIMATED, seed, variable] = path
        records.append({"case": case, "seed": seed, "run_id": run.name,
                        "manifest": str(manifest_path), "checkpoint": manifest["checkpoint"],
                        "checkpoint_sha256": manifest["checkpoint_sha256"]})
    return sources, pd.DataFrame(records)


def calculate(case, seeds, sources):
    moment_rows, distance_rows, summaries = [], [], []
    for variable, unit in rx.pf.VARIABLES:
        arrays, counts = {}, {}
        for condition in CONDITIONS:
            means, skews = [], []
            for seed in seeds:
                errors, excluded = statistics.load_errors(sources[condition, seed, variable])
                arrays[condition, seed] = errors
                counts[condition, seed] = len(errors) + excluded
                if condition == rx.ESTIMATED and counts[condition, seed] != counts["matched_aged", seed]:
                    raise ValueError(f"RX estimated/matched-aged counts differ: case{case}, seed{seed}, {variable}")
                mean, skew = float(errors.mean()), statistics.skewness_around_zero(errors)
                means.append(mean)
                skews.append(skew)
                moment_rows.append({"case": case, "seed": seed, "variable": variable, "unit": unit,
                                    "condition": condition, "mean_signed_error": mean, "skewness_around_zero": skew,
                                    "n_errors": len(errors) + excluded, "n_finite": len(errors), "n_excluded": excluded})
            for metric, units, values in (("mean_signed_error", unit, means),
                                          ("skewness_around_zero", "dimensionless", skews)):
                summaries.append(statistics.summarize(case, variable, units, metric, condition, values))
        for left, right in combinations(CONDITIONS, 2):
            comparison = f"{left}_vs_{right}"
            values = []
            for seed in seeds:
                value = float(wasserstein_distance(arrays[left, seed], arrays[right, seed]))
                values.append(value)
                distance_rows.append({"case": case, "variable": variable, "unit": unit,
                                      "condition_or_comparison": comparison, "reference_seed": seed,
                                      "comparison_seed": seed, "wasserstein_1": value})
            summaries.append(statistics.summarize(case, variable, unit, "wasserstein_1", comparison, values))
    return pd.DataFrame(moment_rows), pd.DataFrame(distance_rows), pd.DataFrame(summaries)


def main(argv=None):
    options = rx.parser(__doc__).parse_args(argv)
    cases, seeds = rx.selected(options)
    label = "-".join(map(str, seeds))
    for case in cases:
        sources, provenance = error_sources(case, seeds)
        moments, distances, aggregate = calculate(case, seeds, sources)
        individual = statistics.per_seed_report(moments, distances)
        directory = rx.RESULTS / "new_rx_statistical_tests" / f"case{case}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in (("statistical_summary", individual), ("statistical_summary_aggregated", aggregate),
                            ("moments_by_seed", moments), ("wasserstein_seed_pairs", distances),
                            ("estimated_error_sources", provenance)):
            path = directory / f"case{case}-{name}_seeds{label}.csv"
            table.to_csv(path, index=False, na_rep="NaN")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()

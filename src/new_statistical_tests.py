"""Summarize saved GENCO errors for an IEEE case without rerunning inference.

Run from the repository root:
    venv/bin/python src/new_statistical_tests.py --case 57

Inputs: src/results/new_error_distribution/caseN/seedS/*_errors.npy
All available seed directories are used and must contain all twelve arrays.
Outputs: src/results/new_statistical_tests/caseN/
    skewness_by_seed.csv
    wasserstein_seed_pairs.csv
    statistical_summary.csv

Skewness about zero is E[e**3] / E[e**2]**1.5, without centering errors.
It is calculated separately for all three conditions and all four variables.
Wasserstein-1 distances compare each matched-aged seed with every mismatched
seed, including equal seeds (5 x 5 = 25 pairs for seeds 0-4). Reverse duplicates
are omitted because the distance is symmetric. Variables remain in physical
units: Vm in p.u., Va in degrees, P in MW, Q in MVAr.

The summary gives equally weighted means and sample standard deviations
(ddof=1) over seed skewness values or over all seed-pair distances. It includes
numeric mean/std columns and a formatted mean +/- std column. These are
descriptive statistics; pairs sharing seeds are not independent replicates.
Non-finite errors are excluded and their counts recorded in skewness_by_seed.
All-zero errors have undefined skewness (NaN); undefined metric values are
counted and excluded from summaries. A single defined value has undefined std.
"""

import argparse
from itertools import product
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


PROJECT_DIR = Path(__file__).resolve().parent
VARIABLES = (("Vm", "p.u."), ("Va", "degrees"), ("P", "MW"), ("Q", "MVAr"))
CONDITIONS = (
    "matched_nominal",
    "matched_aged",
    "mismatched_aged_pf_nominal_impedance",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(14, 30, 57, 118), required=True)
    return parser.parse_args(argv)


def discover_seed_directories(input_dir):
    """Require complete exports for every discovered seed, avoiding partial summaries."""
    directories = {}
    for path in input_dir.glob("seed*"):
        match = re.fullmatch(r"seed(\d+)", path.name)
        if path.is_dir() and match:
            seed = int(match.group(1))
            if seed in directories:
                raise ValueError(f"Multiple directories for seed {seed}: {input_dir}")
            directories[seed] = path
    if not directories:
        raise FileNotFoundError(
            f"No seed directories in {input_dir}. Run new_error_distribution.py first."
        )
    missing = [
        directory / f"{condition}_{variable}_errors.npy"
        for _, directory in sorted(directories.items())
        for condition in CONDITIONS
        for variable, _ in VARIABLES
        if not (directory / f"{condition}_{variable}_errors.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete NumPy error exports; run new_error_distribution.py for "
            "these case/seed combinations:\n" + "\n".join(map(str, missing))
        )
    return dict(sorted(directories.items()))


def load_errors(path):
    """Load numeric error values, retaining full tails and reporting exclusions."""
    errors = np.load(path, allow_pickle=False)
    if errors.ndim != 1 or errors.dtype.kind not in "fiu":
        raise ValueError(f"Expected a one-dimensional real numeric error array: {path}")
    errors = np.asarray(errors, dtype=np.float64)
    finite = errors[np.isfinite(errors)]
    if not finite.size:
        raise ValueError(f"No finite errors in {path}")
    return finite, errors.size - finite.size


def skewness_around_zero(errors):
    """Compute normalized third moment about zero with stable scaling."""
    scale = np.max(np.abs(errors))
    if scale == 0:
        return np.nan
    scaled = errors / scale
    return float(np.mean(scaled**3) / np.mean(scaled**2)**1.5)


def summarize(case, variable, unit, metric, condition, values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if finite.size else np.nan
    std = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
    return {
        "case": case,
        "variable": variable,
        "unit": unit,
        "metric": metric,
        "condition_or_comparison": condition,
        "n_values": values.size,
        "n_defined": finite.size,
        "n_undefined": values.size - finite.size,
        "mean": mean,
        "std": std,
        "mean_plus_minus_std": f"{mean:.8g} +/- {std:.8g}",
    }


def calculate_statistics(case, seed_directories):
    skewness_rows, distance_rows, summary_rows = [], [], []
    seeds = list(seed_directories)
    for variable, unit in VARIABLES:
        arrays = {}
        for condition in CONDITIONS:
            skewness_values = []
            for seed, directory in seed_directories.items():
                path = directory / f"{condition}_{variable}_errors.npy"
                errors, n_excluded = load_errors(path)
                arrays[condition, seed] = errors
                skewness = skewness_around_zero(errors)
                skewness_values.append(skewness)
                skewness_rows.append({
                    "case": case,
                    "variable": variable,
                    "condition": condition,
                    "seed": seed,
                    "n_errors": errors.size + n_excluded,
                    "n_finite": errors.size,
                    "n_excluded": n_excluded,
                    "skewness_around_zero": skewness,
                })
            summary_rows.append(summarize(
                case, variable, "dimensionless", "skewness_around_zero",
                condition, skewness_values,
            ))

        distances = []
        for aged_seed, mismatched_seed in product(seeds, repeat=2):
            distance = float(wasserstein_distance(
                arrays["matched_aged", aged_seed],
                arrays["mismatched_aged_pf_nominal_impedance", mismatched_seed],
            ))
            distances.append(distance)
            distance_rows.append({
                "case": case,
                "variable": variable,
                "unit": unit,
                "matched_aged_seed": aged_seed,
                "mismatched_seed": mismatched_seed,
                "wasserstein_1": distance,
            })
        summary_rows.append(summarize(
            case, variable, unit, "wasserstein_1",
            "matched_aged_vs_mismatched_aged_pf_nominal_impedance", distances,
        ))
    return (
        pd.DataFrame(skewness_rows),
        pd.DataFrame(distance_rows),
        pd.DataFrame(summary_rows),
    )


def main(argv=None):
    options = parse_args(argv)
    input_dir = PROJECT_DIR / "results" / "new_error_distribution" / f"case{options.case}"
    output_dir = PROJECT_DIR / "results" / "new_statistical_tests" / f"case{options.case}"
    seed_directories = discover_seed_directories(input_dir)
    print(f"IEEE-{options.case}: seeds {list(seed_directories)}")
    print(f"Wasserstein-1: {len(seed_directories)**2} seed pairs per variable")
    skewness, distances, summary = calculate_statistics(options.case, seed_directories)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("skewness_by_seed.csv", skewness),
        ("wasserstein_seed_pairs.csv", distances),
        ("statistical_summary.csv", summary),
    ):
        path = output_dir / filename
        table.to_csv(path, index=False, na_rep="NaN")
        print(f"Saved {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

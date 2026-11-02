"""Compare signed GENCO errors across local perturbation and ageing datasets.

Run from the repository root:
    venv/bin/python src/error_evaluation_2.py --case 57 --seed 0

Inputs: src/datakit_out/caseN/seedS/<dataset>/raw/
Outputs: src/results/error_evaluation_2/caseN/seedS/
The data seed selects datasets; the pretrained model always uses training seed 0.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from error_evaluation import (
    DEFAULT_DATA_CACHE,
    PROJECT_DIR,
    HuggingFaceModel,
    _evaluate_pair,
    _plot_error_tables,
    _validate_local_dataset_root,
    load_huggingface_model,
)


def dataset_names(case, seed):
    """Name the five nominal and four aged conditions for one experiment."""
    nominal = f"case{case}"
    aged = f"case{case}_aged_seed{seed}_k2p5_s40"
    return {
        "normal": f"{nominal}_normal",
        "one_out_topo": f"{nominal}_one_out_topo",
        "one_out_gen": f"{nominal}_one_out_gen",
        "one_out_adm": f"{nominal}_one_out_adm",
        "all_out": f"{nominal}_all_out",
        "d50_normal": f"{aged}_d50_a60_g2_normal",
        "d10_normal": f"{aged}_d10_a60_g2_normal",
        "d50_all_out": f"{aged}_d50_a60_g2_all_out",
        "d10_all_out": f"{aged}_d10_a60_g2_all_out",
    }


COMPARISONS = (
    ("normal", "one_out_topo"),
    ("normal", "one_out_gen"),
    ("normal", "one_out_adm"),
    ("normal", "all_out"),
    ("all_out", "one_out_topo"),
    ("all_out", "one_out_gen"),
    ("all_out", "one_out_adm"),
    ("normal", "d50_normal"),
    ("normal", "d10_normal"),
    ("d10_normal", "d50_normal"),
    ("all_out", "d50_all_out"),
    ("all_out", "d10_all_out"),
    ("d10_all_out", "d50_all_out"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=int,
        choices=(14, 30, 57, 118),
        default=57,
        help="IEEE bus system (default: 57)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Dataset seed (default: 0); model training seed stays 0",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap on saved scenarios per dataset; default evaluates all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default results/error_evaluation_2/caseN/seedS directory",
    )
    options = parser.parse_args(argv)
    if not 0 <= options.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if options.max_samples is not None and options.max_samples <= 0:
        parser.error("--max-samples must be a positive integer")
    return options


def evaluate_conditions(case, seed, *, max_samples=None):
    """Load one model and evaluate each local condition exactly once."""
    names = dataset_names(case, seed)
    data_root = PROJECT_DIR / "datakit_out" / f"case{case}" / f"seed{seed}"
    missing = []
    for name in names.values():
        try:
            _validate_local_dataset_root(data_root / name)
        except FileNotFoundError as error:
            missing.append(str(error))
    if missing:
        raise FileNotFoundError(
            "Missing required evaluation data:\n" + "\n".join(missing)
        )

    model = HuggingFaceModel(
        "gridfm/genco-pf-datakit",
        f"case{case}_ieee/base/seed0",
        config_filename=f"case{case}_ieee_base_seed0.yaml",
        normalizer_network=f"case{case}_ieee",
    )
    bundle = load_huggingface_model(model)
    tables = {}
    for name in names.values():
        tables[name] = _evaluate_pair(
            (data_root / name, model),
            name,
            max_samples=max_samples,
            data_cache_dir=DEFAULT_DATA_CACHE,
            model_bundle=bundle,
        )
    return tables


def plot_comparisons(tables, case, seed, output_dir):
    """Write thirteen pairwise PDFs and one overlay of all nine conditions."""
    names = dataset_names(case, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the color for a condition identical across all fourteen figures.
    palette = plt.get_cmap("tab10")
    colors = {name: palette(index) for index, name in enumerate(names.values())}
    paths = []
    for left, right in COMPARISONS:
        labels = (names[left], names[right])
        output_path = output_dir / f"{labels[0]}_vs_{labels[1]}.pdf"
        figure = _plot_error_tables(
            {label: tables[label] for label in labels},
            output_path=output_path,
            colors=colors,
        )
        plt.close(figure)
        paths.append(output_path)
    combined_path = output_dir / "all_conditions.pdf"
    figure = _plot_error_tables(
        {name: tables[name] for name in names.values()},
        output_path=combined_path,
        colors=colors,
    )
    plt.close(figure)
    paths.append(combined_path)
    return paths


def load_perturbation_types(dataset_root):
    """Read the last generation's actual settings rather than infer from names."""
    sections = {
        "topo pert": "topology_perturbation",
        "gen pert": "generation_perturbation",
        "adm pert": "admittance_perturbation",
    }
    log_path = Path(dataset_root) / "raw" / "args.log"
    if not log_path.is_file():
        return {column: "unknown" for column in sections}
    text = log_path.read_text()
    _, marker, latest_generation = text.rpartition("New generation started at")
    if marker:
        text = latest_generation.partition("\n")[2]
    config = yaml.safe_load(text) or {}
    return {
        column: config.get(section, {}).get("type", "unknown")
        for column, section in sections.items()
    }


def error_statistics(errors):
    """Compute signed mean, mean/population std, and raw skewness about zero."""
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        raise ValueError("Cannot summarize an error series without finite values")
    mean = errors.mean()
    std = errors.std(ddof=0)
    # Scaling cancels in mean(e**3) / mean(e**2)**1.5 and avoids overflowing
    # or underflowing the raw powers. Do not subtract the error mean.
    scale = np.abs(errors).max()
    if scale > 0:
        scaled = errors / scale
        skewness = np.mean(scaled**3) / np.mean(scaled**2) ** 1.5
    else:
        skewness = np.nan
    return {
        "mean error": mean,
        "mean error/error std": mean / std if std > 0 else np.nan,
        "skewness around zero": skewness,
    }


def report_error_statistics(tables, case, seed, output_dir):
    """Print and save a seven-column dataset summary for each physical quantity."""
    data_root = PROJECT_DIR / "datakit_out" / f"case{case}" / f"seed{seed}"
    settings = {name: load_perturbation_types(data_root / name) for name in tables}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    variables = (("Vm", "p.u."), ("Va", "degrees"), ("P", "MW"), ("Q", "MVAr"))
    for variable, unit in variables:
        rows = []
        for name, table in tables.items():
            values = table.loc[table["variable"] == variable]
            finite = np.isfinite(values["actual"]) & np.isfinite(values["error"])
            errors = values.loc[finite, "error"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "exp name": name,
                    **settings[name],
                    **error_statistics(errors),
                }
            )
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / f"{variable}_error_summary.csv", index=False)
        print(f"\n{variable} error summary (mean error in {unit}; population std, ddof=0):")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
        summaries[variable] = summary
    return summaries


def main(argv=None):
    options = parse_args(argv)
    output_dir = options.output_dir or (
        PROJECT_DIR
        / "results"
        / "error_evaluation_2"
        / f"case{options.case}"
        / f"seed{options.seed}"
    )
    tables = evaluate_conditions(
        options.case,
        options.seed,
        max_samples=options.max_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}_errors.parquet", index=False)
    paths = plot_comparisons(tables, options.case, options.seed, output_dir)
    report_error_statistics(tables, options.case, options.seed, output_dir)
    print(
        f"Saved {len(paths)} PDF comparisons, {len(tables)} error tables, "
        f"and 4 summary CSVs to {output_dir}"
    )


if __name__ == "__main__":
    main()

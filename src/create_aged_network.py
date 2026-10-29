"""Age only transmission-line R/X on ('bus', 'connects', 'bus') edges.

MATPOWER's mpc.branch table defines bus-to-bus branches. Generator-to-bus
and bus-to-generator edges describe generator attachments, not line
impedances; their source data in mpc.gen are never modified here.
Transformer branches and out-of-service branches retain their parameters.
Every eligible line receives a right-truncated Weibull age and a shared
age-dependent degradation multiplier for resistance and reactance.
"""

import argparse
import json
import math
import random
import re
from decimal import Decimal
from pathlib import Path

from gridfm_datakit.network import get_pglib_source_path

BRANCH_RE = re.compile(r"(mpc\.branch\s*=\s*\[)(.*?)(\];)", re.DOTALL)
NUMBER_RE = re.compile(r"[^\s;]+")
FUNCTION_RE = re.compile(r"(?m)^(\s*function\s+mpc\s*=\s*)[A-Za-z]\w*")
AGED_EDGE_TYPE = ("bus", "connects", "bus")


def set_matpower_function_name(contents, name):
    """Set the MATPOWER function name to match its .m filename."""
    updated, replacements = FUNCTION_RE.subn(
        lambda match: match.group(1) + name,
        contents,
        count=1,
    )
    if replacements != 1:
        raise ValueError("Could not find MATPOWER function declaration")
    return updated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=int,
        choices=(14, 30, 57, 118),
        default=118,
        help="IEEE bus system to age (default: 118)",
    )
    parser.add_argument(
        "--weibull-shape",
        type=float,
        default=2.5,
        help="Weibull shape k (default: 2.5)",
    )
    parser.add_argument(
        "--weibull-scale",
        type=float,
        default=40.0,
        help="Weibull scale in years (default: 40)",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=60.0,
        help="Right truncation age in years; minimum is zero (default: 60)",
    )
    parser.add_argument(
        "--max-degradation",
        type=float,
        default=0.5,
        help="Fractional R/X increase at maximum age; 0.5 means 50%% (default: 0.5)",
    )
    parser.add_argument(
        "--degradation-exponent",
        type=float,
        default=2.0,
        help="Exponent gamma in D = max_degradation * (age / max_age)^gamma (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling line ages (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Base output filename (default: grids/case<case>_aged.m); seed, k, s, degradation percent d, maximum age a and exponent g are appended",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("grids") / f"case{args.case}_aged.m"
    for name in ("weibull_shape", "weibull_scale", "max_age", "degradation_exponent"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.max_degradation) or args.max_degradation < 0:
        parser.error("--max-degradation must be finite and nonnegative")
    try:
        truncation_cdf = -math.expm1(
            -((args.max_age / args.weibull_scale) ** args.weibull_shape)
        )
    except OverflowError:
        parser.error("Weibull parameters exceed the supported numerical range")
    if truncation_cdf == 0:
        parser.error("Weibull truncation probability is too small to represent")
    return args


def sample_truncated_weibull(rng, shape, scale, max_age):
    """Sample A | 0 <= A <= max_age by inverting the conditional CDF.

    F(a) = 1 - exp(-(a / scale)^shape). Draw p uniformly in
    [0, F(max_age)), then return F^-1(p). No clipping or boundary mass.
    expm1/log1p avoid cancellation for small truncation probabilities.
    """
    truncation_cdf = -math.expm1(-((max_age / scale) ** shape))
    probability = rng.random() * truncation_cdf
    return scale * (-math.log1p(-probability)) ** (1.0 / shape)


def parameterized_output_path(
    output, seed, shape, scale, max_degradation=0.5, max_age=60.0,
    degradation_exponent=2.0,
):
    """Append seed/k/s/d/a/g tags; d is a percentage, s and a are years.

    Encode decimals as ``p`` to keep the MATPOWER function name valid.
    Include every age/degradation parameter so changing the degradation cap,
    truncation age or exponent creates a distinct file and corrected-file cache.
    """

    def tag(value):
        value = format(Decimal(str(value)), "f")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        return (
            value
            .replace(".", "p")
            .replace("-", "m")
            .replace("+", "")
        )

    degradation_percent = format(Decimal(str(max_degradation)) * 100, "f")
    return output.with_name(
        f"{output.stem}_seed{tag(seed)}_k{tag(shape)}_s{tag(scale)}"
        f"_d{tag(degradation_percent)}_a{tag(max_age)}"
        f"_g{tag(degradation_exponent)}{output.suffix}"
    )


def plot_age_distribution(changes, output_path):
    """Save a histogram of the actual sampled ages, one per eligible branch."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ages = [line["age_years"] for line in changes["affected_lines"]]
    minimum, maximum = changes["age_range_years"]
    fig = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.hist(
        ages,
        bins=20,
        range=(minimum, maximum),
        color="tab:blue",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlim(minimum, maximum)
    ax.set_xlabel("Line age (years)")
    ax.set_ylabel("Number of eligible bus-to-bus lines")
    ax.set_title(
        f"Sampled line ages — right-truncated Weibull\n"
        f"k={changes['weibull_shape']:g}, scale={changes['weibull_scale_years']:g} years, "
        f"seed={changes['seed']}; {len(ages)} lines"
    )
    if not ages:
        ax.text(0.5, 0.5, "No eligible lines", transform=ax.transAxes, ha="center")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(output_path, dpi=300)
    return fig


def plot_weibull_distribution(changes, output_path):
    """Plot the theoretical conditional age PDF, independent of sampled lines."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    shape = changes["weibull_shape"]
    scale = changes["weibull_scale_years"]
    maximum = changes["age_range_years"][1]
    probability = -math.expm1(-((maximum / scale) ** shape))
    ages = np.linspace(0, maximum, 2001)
    # For k < 1 the PDF diverges at zero; plot positive ages rather than cap it.
    if shape < 1:
        ages = ages[1:]
    density = (shape / scale) * (ages / scale) ** (shape - 1) * np.exp(
        -((ages / scale) ** shape)
    ) / probability

    fig = Figure(figsize=(7, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(ages, density, color="tab:blue", linewidth=2)
    ax.set_xlim(0, maximum)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Probability density (1/year)")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)
    return fig


def main():
    args = parse_args()
    original_network_name = f"case{args.case}_original"
    source_network_name = f"case{args.case}_ieee"
    output_path = parameterized_output_path(
        args.output, args.seed, args.weibull_shape, args.weibull_scale,
        args.max_degradation, args.max_age, args.degradation_exponent,
    )

    source = Path(get_pglib_source_path(source_network_name))

    original = source.read_text()
    # Only this table maps to AGED_EDGE_TYPE. Do not edit mpc.bus or mpc.gen.
    match = BRANCH_RE.search(original)
    if match is None:
        raise ValueError(f"Could not find mpc.branch table in {source}")

    lines = match.group(2).splitlines(keepends=True)
    eligible = []
    branch_rows = {}
    branch_row = 0
    for idx, line in enumerate(lines):
        data = line.partition("%")[0]
        fields = NUMBER_RE.findall(data)
        if not fields:
            continue
        branch_row += 1
        branch_rows[idx] = branch_row
        if len(fields) < 11:
            raise ValueError(
                f"Malformed branch row {idx + 1}: expected at least 11 columns, "
                f"found {len(fields)}"
            )

        # MATPOWER branch columns (zero-based): 2=R, 3=X, 8=TAP, 9=SHIFT, 10=STATUS.
        tap, shift, status = float(fields[8]), float(fields[9]), float(fields[10])
        is_transformer = (tap != 0.0 and tap != 1.0) or shift != 0.0
        if status > 0 and not is_transformer:
            eligible.append(idx)

    rng = random.Random(args.seed)
    affected_lines = []
    for idx in eligible:
        line = lines[idx]
        data, marker, comment = line.partition("%")
        tokens = list(NUMBER_RE.finditer(data))
        fields = [t.group() for t in tokens]
        original_resistance = float(fields[2])
        original_reactance = float(fields[3])
        age = sample_truncated_weibull(
            rng, args.weibull_shape, args.weibull_scale, args.max_age
        )
        degradation = (
            args.max_degradation * (age / args.max_age) ** args.degradation_exponent
        )
        multiplier = 1.0 + degradation
        affected_resistance = float(f"{original_resistance * multiplier:.16g}")
        affected_reactance = float(f"{original_reactance * multiplier:.16g}")
        updated_values = {2: affected_resistance, 3: affected_reactance}
        # Replace only R and X, preserving all other MATPOWER fields, spacing and comments.
        # reverse order keeps earlier match offsets unchanged
        for column in (3, 2):
            token = tokens[column]
            updated = f"{updated_values[column]:.16g}"
            data = data[: token.start()] + updated + data[token.end() :]
        lines[idx] = data + (marker + comment if marker else "")
        affected_lines.append(
            {
                "branch_row": branch_rows[idx],
                "from_bus": int(float(fields[0])),
                "to_bus": int(float(fields[1])),
                "age_years": age,
                "degradation": degradation,
                "original_resistance": original_resistance,
                "affected_resistance": affected_resistance,
                "resistance_multiplier": multiplier,
                "original_reactance": original_reactance,
                "affected_reactance": affected_reactance,
                "reactance_multiplier": multiplier,
            }
        )

    # Splice only branch rows: all other tables and generator attachments are
    # copied verbatim. The function declaration is renamed to match the output.
    amended = original[: match.start(2)] + "".join(lines) + original[match.end(2) :]
    amended = set_matpower_function_name(amended, output_path.stem)

    original_output = args.output.with_name(f"{original_network_name}.m")
    if original_output == args.output:
        raise ValueError(
            f"The aged output cannot be named {original_network_name}.m because "
            "that filename is reserved for the original network"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(amended)
    original_output.write_text(
        set_matpower_function_name(original, original_network_name)
    )
    changes = {
        "case": args.case,
        "source_network": source_network_name,
        "network_file": output_path.name,
        "original_network_file": original_output.name,
        "affected_edge_type": list(AGED_EDGE_TYPE),
        "seed": args.seed,
        "age_distribution": "right_truncated_weibull",
        "weibull_shape": args.weibull_shape,
        "weibull_scale_years": args.weibull_scale,
        "age_range_years": [0.0, args.max_age],
        "max_degradation": args.max_degradation,
        "degradation_exponent": args.degradation_exponent,
        "affected_lines": affected_lines,
    }
    age_plot_path = output_path.with_name(f"{output_path.stem}_age_distribution.png")
    changes["age_distribution_plot"] = age_plot_path.name
    plot_age_distribution(changes, age_plot_path)
    weibull_plot_path = output_path.with_name(f"{output_path.stem}_weibull_distribution.pdf")
    changes["weibull_distribution_plot"] = weibull_plot_path.name
    plot_weibull_distribution(changes, weibull_plot_path)
    output_path.with_suffix(".json").write_text(json.dumps(changes, indent=2) + "\n")
    print(f"Saved aged network to {output_path}")
    print(f"Saved line age distribution to {age_plot_path}")
    print(f"Saved theoretical Weibull distribution to {weibull_plot_path}")


if __name__ == "__main__":
    # example usage:
    # python create_aged_network.py --seed 0 --output grids/case118_aged.m
    # python create_aged_network.py --case 57 --seed 0
    main()

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
from pathlib import Path

from gridfm_datakit.network import get_pglib_source_path

BRANCH_RE = re.compile(r"(mpc\.branch\s*=\s*\[)(.*?)(\];)", re.DOTALL)
NUMBER_RE = re.compile(r"[^\s;]+")
FUNCTION_RE = re.compile(r"(?m)^(\s*function\s+mpc\s*=\s*)[A-Za-z]\w*")
ORIGINAL_NETWORK_NAME = "case118_original"
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
        default=42,
        help="Random seed for sampling line ages (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .m file for the aged network",
    )
    args = parser.parse_args()
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


def main():
    args = parse_args()

    source = Path(get_pglib_source_path("case118_ieee"))

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
    amended = set_matpower_function_name(amended, args.output.stem)

    original_output = args.output.with_name(f"{ORIGINAL_NETWORK_NAME}.m")
    if original_output == args.output:
        raise ValueError(
            f"The aged output cannot be named {ORIGINAL_NETWORK_NAME}.m because "
            "that filename is reserved for the original network"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(amended)
    original_output.write_text(
        set_matpower_function_name(original, ORIGINAL_NETWORK_NAME)
    )
    changes = {
        "network_file": args.output.name,
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
    args.output.with_suffix(".json").write_text(json.dumps(changes, indent=2) + "\n")


if __name__ == "__main__":
    # example usage:
    # python create_aged_network.py --seed 42 --output grids/case118_aged.m
    main()

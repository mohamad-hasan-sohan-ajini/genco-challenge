"""Generate an IEEE 118 MATPOWER case with selected line impedances varied."""

import argparse
import json
import math
import random
import re
from pathlib import Path

from gridfm_datakit.network import get_pglib_source_path

BRANCH_RE = re.compile(r"(mpc\.branch\s*=\s*\[)(.*?)(\];)", re.DOTALL)
NUMBER_RE = re.compile(r"[^\s;]+")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aged-lines-percent",
        type=float,
        default=10.0,
        help="Percentage of eligible lines to age (default: 10)",
    )
    parser.add_argument(
        "--min-res",
        type=float,
        default=1.0,
        help="Minimum per-unit resistance multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--max-res",
        type=float,
        default=1.5,
        help="Maximum per-unit resistance multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--min-react",
        type=float,
        default=1.0,
        help="Minimum per-unit reactance multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--max-react",
        type=float,
        default=1.5,
        help="Maximum per-unit reactance multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting aged lines (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .m file for the aged network",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    source = Path(get_pglib_source_path("case118_ieee"))

    original = source.read_text()
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

    n_age = math.floor(len(eligible) * args.aged_lines_percent / 100 + 0.5)
    rng = random.Random(args.seed)
    chosen = set(rng.sample(eligible, n_age))
    affected_lines = []
    for idx in sorted(chosen):
        line = lines[idx]
        data, marker, comment = line.partition("%")
        tokens = list(NUMBER_RE.finditer(data))
        fields = [t.group() for t in tokens]
        original_resistance = float(fields[2])
        original_reactance = float(fields[3])
        multipliers = {
            2: rng.uniform(args.min_res, args.max_res),
            3: rng.uniform(args.min_react, args.max_react),
        }
        affected_resistance = float(f"{original_resistance * multipliers[2]:.16g}")
        affected_reactance = float(f"{original_reactance * multipliers[3]:.16g}")
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
                "original_resistance": original_resistance,
                "affected_resistance": affected_resistance,
                "resistance_multiplier": multipliers[2],
                "original_reactance": original_reactance,
                "affected_reactance": affected_reactance,
                "reactance_multiplier": multipliers[3],
            }
        )

    amended = original[: match.start(2)] + "".join(lines) + original[match.end(2) :]
    # MATPOWER expects the function name to match the .m filename.
    function_pattern = re.compile(r"(?m)^(\s*function\s+mpc\s*=\s*)[A-Za-z]\w*")
    amended, replacements = function_pattern.subn(
        lambda m: m.group(1) + args.output.stem,
        amended,
        count=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(amended)
    changes = {
        "network_file": args.output.name,
        "seed": args.seed,
        "aged_lines_percent": args.aged_lines_percent,
        "resistance_multiplier_range": [args.min_res, args.max_res],
        "reactance_multiplier_range": [args.min_react, args.max_react],
        "affected_lines": affected_lines,
    }
    args.output.with_suffix(".json").write_text(json.dumps(changes, indent=2) + "\n")


if __name__ == "__main__":
    # example usage:
    # python create_aged_network.py --aged-lines-percent 90 --min-res 1.01 --max-res 1.5 --min-react 1.01 --max-react 1.5 --seed 42 --output grids/case118_aged.m
    main()

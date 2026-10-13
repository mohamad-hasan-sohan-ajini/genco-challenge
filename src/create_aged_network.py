"""Generate an IEEE 118 MATPOWER case with selected line impedances increased."""

import argparse
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
        "--impedance-increase-percent",
        type=float,
        default=10.0,
        help="Percentage increase in R and X for aged lines (default: 10)",
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
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(get_pglib_source_path("case118_ieee"))

    original = source.read_text()
    match = BRANCH_RE.search(original)
    if match is None:
        raise ValueError(f"Could not find mpc.branch table in {source}")

    lines = match.group(2).splitlines(keepends=True)
    eligible = []
    for idx, line in enumerate(lines):
        data = line.partition("%")[0]
        fields = NUMBER_RE.findall(data)
        if not fields:
            continue
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
    chosen = set(random.Random(args.seed).sample(eligible, n_age))
    multiplier = 1 + args.impedance_increase_percent / 100
    aged_pairs = []
    for idx in sorted(chosen):
        line = lines[idx]
        data, marker, comment = line.partition("%")
        tokens = list(NUMBER_RE.finditer(data))
        fields = [t.group() for t in tokens]
        aged_pairs.append((int(float(fields[0])), int(float(fields[1]))))
        # Replace only R and X, preserving all other MATPOWER fields, spacing and comments.
        for column in (3, 2):  # reverse order keeps earlier match offsets unchanged
            token = tokens[column]
            updated = f"{float(fields[column]) * multiplier:.16g}"
            data = data[: token.start()] + updated + data[token.end() :]
        lines[idx] = data + (marker + comment if marker else "")

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


if __name__ == "__main__":
    # example usage:
    # python create_aged_network.py --aged-lines-percent 10 --impedance-increase-percent 5 --seed 42 --output grids/case118_aged05.m
    main()

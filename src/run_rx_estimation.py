"""Run joint R/X projected-Adam estimation on isolated logrx data (seed 0 by default)."""

import subprocess
import sys

import rx_common as rx


def estimation_command(case, seed, epochs):
    return [
        sys.executable, str(rx.ROOT / "aged_impedance_estimation_2.py"),
        "--case", str(case), "--seed", str(seed), "--ageing", rx.aged_name(case, seed),
        "--epochs", str(epochs), "--kl-weight", "0",
        "--data-root", str(rx.DATA), "--grid-dir", str(rx.GRIDS),
        "--results-dir", str(rx.RESULTS), "--tb-dir", str(rx.TB),
    ]


def main(argv=None):
    parser = rx.parser(__doc__)
    parser.add_argument("--epochs", type=int, default=60,
                        help="Optimization epochs (default: 60, matching resistance-only runs)")
    options = parser.parse_args(argv)
    if options.epochs <= 0:
        parser.error("--epochs must be positive")
    cases, seeds = rx.selected(options)
    for seed in seeds:
        for case in cases:
            _, aged = rx.dataset_roots(case, seed)
            for name in ("bus", "gen", "branch"):
                if not (aged / "raw" / f"{name}_data.parquet").exists():
                    raise FileNotFoundError(f"Generate the RX data first: {aged}")
    for seed in seeds:
        for case in cases:
            print(f"Joint R/X estimation: case{case}, seed{seed}", flush=True)
            subprocess.run(estimation_command(case, seed, options.epochs), check=True, cwd=rx.ROOT)


if __name__ == "__main__":
    main()

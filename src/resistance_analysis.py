"""Paths and case/seed selection for the resistance-only analysis."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = (14, 30, 57, 118)
TAG = "k2p5_s40_d50_a60_logr"
DATA = ROOT / "datakit_out_new"
GRIDS = ROOT / "grids_new"
RESULTS = ROOT / "results_new"
TB = ROOT / "tb_new"
BASELINES = RESULTS / "new_error_distribution"
DEPLOYMENT = RESULTS / "new_resistance_error_reduction"
STATISTICS = RESULTS / "new_statistical_tests"


def dataset_seed(value):
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("dataset seeds must be between 0 and 2**32 - 1")
    return seed


def parser(description, *, single_seed_option=False):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--case", type=int, nargs="+", choices=CASES, default=list(CASES))
    seeds = result.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=dataset_seed, nargs="+")
    seeds.add_argument("--all-seeds", action="store_true", help="Use seeds 0–4")
    if single_seed_option:
        seeds.add_argument("--seed", type=dataset_seed, help="Select one dataset seed")
    return result


def select_cases_and_seeds(options):
    seed = getattr(options, "seed", None)
    if options.all_seeds:
        options.seeds = list(range(5))
    else:
        options.seeds = sorted(set(options.seeds or [0 if seed is None else seed]))
    options.case = sorted(set(options.case))
    return options


def ageing_name(case, seed, tag=TAG):
    return f"case{case}_aged_seed{seed}_{tag}"

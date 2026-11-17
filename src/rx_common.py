"""Shared paths, complete-dataset inference, and exports for the R+X experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm.auto import tqdm

import new_error_distribution as pf
from aged_impedance_estimation import read_matpower_branch_rows

ROOT = Path(__file__).resolve().parent
CASES = (14, 30, 57, 118)
TAG = "k2p5_s40_d50_a60_logrx"
DATA = ROOT / "datakit_out_rx"
GRIDS = ROOT / "grids_rx"
RESULTS = ROOT / "results_rx"
TB = ROOT / "tb_rx"
BASELINES = RESULTS / "new_rx_error_distribution"
DEPLOYMENT = RESULTS / "new_rx_impedance_error_reduction" / TAG / "pf_only"
BASE_CONDITIONS = ("matched_nominal", "matched_aged", "mismatched_aged_pf_nominal_impedance")
LABELS = dict(zip(BASE_CONDITIONS, (pf.MATCHED_NOMINAL, pf.MATCHED_AGED, pf.MISMATCHED)))
ESTIMATED = "aged_pf_estimated_impedance"
LABELS[ESTIMATED] = "Estimated Impedance"
COLORS = {**pf.CONDITION_COLORS, LABELS[ESTIMATED]: "blue"}


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--case", type=int, nargs="+", choices=CASES, default=list(CASES))
    seeds = result.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+", choices=range(5))
    seeds.add_argument("--all-seeds", action="store_true", help="Use seeds 0-4 instead of seed 0 only")
    return result


def selected(options):
    seeds = list(range(5)) if options.all_seeds else (options.seeds or [0])
    return sorted(set(options.case)), sorted(set(seeds))


def aged_name(case, seed):
    return f"case{case}_aged_seed{seed}_{TAG}"


def dataset_roots(case, seed):
    root = DATA / f"case{case}" / f"seed{seed}"
    return root / f"case{case}_original", root / aged_name(case, seed)


def admittance(case_path, normalizer):
    """Rebuild all directed Y values, including transformer taps and line charging."""
    rows = read_matpower_branch_rows(case_path)
    values = torch.tensor(rows, dtype=torch.float32)
    impedance = torch.complex(values[:, 2], values[:, 3])
    if (impedance.abs() == 0).any():
        raise ValueError(f"Zero branch impedance: {case_path}")
    tap = torch.polar(torch.where(values[:, 8] == 0, 1., values[:, 8]), torch.deg2rad(values[:, 9]))
    scale = float(normalizer.baseMVA_orig / normalizer.baseMVA)
    status = (values[:, 10] > 0).float()
    series = status * scale / impedance
    own = series + 1j * status * scale * values[:, 4] / 2
    forward_self, reverse_self = own / tap.abs().square(), own
    forward_cross, reverse_cross = -series / tap.conj(), -series / tap
    return torch.cat([
        torch.stack([forward_self.real, forward_self.imag, forward_cross.real, forward_cross.imag], dim=1),
        torch.stack([reverse_self.real, reverse_self.imag, reverse_cross.real, reverse_cross.imag], dim=1),
    ])


def load_dataset(source, bundle):
    return pf.load_data_source(
        source, bundle[1], stream_partitions=getattr(bundle[2].data, "stream_partitions", "auto"),
        cache_dir=pf.DEFAULT_DATA_CACHE,
    )


def indices(dataset):
    result = sorted(int(path.stem.removeprefix("data_index_"))
                    for path in Path(dataset.processed_dir).glob("data_index_*.pt"))
    if not result:
        raise ValueError(f"No eligible scenarios: {dataset.processed_dir}")
    return result


def deploy(graph, checkpoint):
    """Insert saved estimated R+X-derived Y, preserving aged observations and outages."""
    edges = graph[pf.EDGE_TYPE]
    if not torch.equal(edges.edge_index, checkpoint["edge_index"]):
        raise ValueError("Checkpoint and graph directed-branch ordering differ")
    result = graph.clone()
    attributes = result[pf.EDGE_TYPE].edge_attr.clone()
    mask = checkpoint["learnable_line_mask"].bool()
    mask = torch.cat([mask, mask]).to(attributes.device)
    active = attributes[:, pf.B_ON] > .5
    saved_y = checkpoint["estimated_admittance_normalized"].to(attributes)
    if saved_y.shape != attributes[:, pf.ADMITTANCE_COLUMNS].shape:
        raise ValueError("Checkpoint and graph branch-feature counts differ")
    attributes[mask, pf.ADMITTANCE_COLUMNS] = 0
    attributes[mask & active, pf.ADMITTANCE_COLUMNS] = saved_y[mask & active]
    result[pf.EDGE_TYPE].edge_attr = attributes
    return result


def evaluate_deployed(dataset, checkpoint, bundle):
    model, normalizer, args = bundle
    transform = pf.get_task_transforms(args)
    frames = []
    with torch.inference_mode():
        for index in tqdm(indices(dataset), desc="Aged PF + estimated R/X"):
            graph = transform(deploy(dataset[index], checkpoint))
            batch = Batch.from_data_list([graph]).to(pf.DEVICE)
            frames.append(pf.prediction_frame(batch, model(batch), index, LABELS[ESTIMATED], normalizer))
    return pd.concat(frames, ignore_index=True)


def save_errors(table, directory, condition, prefix=""):
    directory.mkdir(parents=True, exist_ok=True)
    table.to_parquet(directory / f"{prefix}{condition}_prediction_errors.parquet", index=False)
    pf.save_numpy_errors(table, f"{prefix}{condition}", directory)


def baseline_tables(case, seed, bundle, recompute=False):
    """Reuse only complete RX baselines; never read resistance-only experiment results."""
    nominal, aged = dataset_roots(case, seed)
    sources = dict(zip(BASE_CONDITIONS, (nominal, aged, aged)))
    directory = BASELINES / f"case{case}" / f"seed{seed}"
    tables = {}
    for condition, source in sources.items():
        dataset = load_dataset(source, bundle)
        eligible = set(indices(dataset))
        path = directory / f"{condition}_prediction_errors.parquet"
        if path.is_file() and not recompute:
            table = pd.read_parquet(path)
            if set(table.dataset_index.unique()) == eligible:
                tables[condition] = table
                continue
        if condition == BASE_CONDITIONS[-1]:
            nominal_y = admittance(GRIDS / f"case{case}_original.m", bundle[1])
            table = pf.evaluate_mismatched(aged, nominal_y, bundle)
        else:
            table = pf._evaluate_pair(
                (source, pf.model_spec(case)), LABELS[condition], max_samples=None,
                data_cache_dir=pf.DEFAULT_DATA_CACHE, model_bundle=bundle,
            )
        if set(table.dataset_index.unique()) != eligible:
            raise ValueError("Baseline evaluation did not cover all eligible scenarios")
        save_errors(table, directory, condition)
        tables[condition] = table
    return tables


def plot_groups(tables, directory, case, seed, estimated=False):
    directory.mkdir(parents=True, exist_ok=True)
    groups = [
        ("matched_nominal_and_aged", BASE_CONDITIONS[:2]),
        ("mismatched", (BASE_CONDITIONS[-1],)),
        ("matched_and_mismatched", BASE_CONDITIONS),
    ]
    if estimated:
        groups.extend([
            ("nominal_vs_estimated_impedance_aged_pf", (BASE_CONDITIONS[-1], ESTIMATED)),
            ("all_conditions_with_estimated_impedance", (*BASE_CONDITIONS, ESTIMATED)),
        ])
    for name, conditions in groups:
        statistics = pf.plot_error_distributions(
            {LABELS[condition]: tables[condition] for condition in conditions},
            directory / f"case{case}-seed{seed}-{name}_prediction_error_distribution.pdf", colors=COLORS,
        )
    statistics.to_csv(directory / f"case{case}-seed{seed}-pf_error_statistics.csv", index=False)


def write_manifest(path, **values):
    path.write_text(json.dumps(values, indent=2) + "\n")

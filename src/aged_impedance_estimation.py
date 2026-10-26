"""Standalone Section 7 of 1_error_dist.ipynb (projected Adam only).

Run from any directory:
    /path/to/venv/bin/python /path/to/src/aged_impedance_estimation.py --case 57 --seed 0

Inputs: src/datakit_out/caseN/seedXX/{caseN_original,
        caseN_aged_seedXX_k2p5_scale40}/raw/
Outputs: src/results/caseN/seedXX/
TensorBoard: src/tb/caseN/seedXX/impedance-multiplier-denoising-TIMESTAMP/

IEEE-118 retains the transfer checkpoint used in the notebook. IEEE-14/30/57
use their matching Datakit Base checkpoint (model-training seed 0). The CLI
seed selects experimental data, independently of the pretrained model's seed.

GENCO is frozen. Independent R/X multipliers use the notebook's [1, 1.5]
bounds and one full-dataset Adam update per epoch. Aged admittances and
impedances are used only for diagnostics and the true-impedance baseline.
All Section 7 plots are title-free PDFs at 300 DPI. Section 8 is excluded.
"""

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.globals import (
    B_ON,
    C0_H,
    C2_H,
    G_ON,
    VA_H,
    VM_H,
    PG_OUT,
    QG_OUT,
    VA_OUT,
    VM_OUT,
    YFF_TT_R,
    YFT_TF_I,
    QG_H,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)
from gridfm_graphkit.io.config_version import resolve_config


from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

MODEL_REPO_ID = "gridfm/genco-pf-transfer"
MODEL_REVISION = "main"
RUN_DIR = Path("finetune/case118_250000")
NORMALIZER_NETWORK = "case118_ieee"
PROJECT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 16
EPOCHS = 250
LR = 3e-3
MULTIPLIER_MIN = 1.0
MULTIPLIER_MAX = 1.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_paths(seed, case=118):
    """Resolve the current Datakit directory layout for one experiment."""
    root = PROJECT_DIR / "datakit_out" / f"case{case}" / f"seed{seed}"
    datasets = {
        f"case{case}_original": root / f"case{case}_original",
        f"case{case}_aged": root / f"case{case}_aged_seed{seed}_k2p5_scale40",
    }
    return (
        datasets,
        PROJECT_DIR / "results" / f"case{case}" / f"seed{seed}",
        PROJECT_DIR / "tb" / f"case{case}" / f"seed{seed}",
    )


def model_artifact_spec(case):
    """Select one matched PF checkpoint/config/normalizer bundle per case."""
    if case == 118:
        return {
            "repo_id": MODEL_REPO_ID,
            "run_dir": str(RUN_DIR),
            "config_filename": "case118_250000.yaml",
            "normalizer_network": NORMALIZER_NETWORK,
        }
    if case not in (14, 30, 57):
        raise ValueError(f"Unsupported IEEE case: {case}")
    return {
        "repo_id": "gridfm/genco-pf-datakit",
        "run_dir": f"case{case}_ieee/base/seed0",
        "config_filename": f"case{case}_ieee_base_seed0.yaml",
        "normalizer_network": f"case{case}_ieee",
    }


def load_model_bundle(case):
    """Load compatible GENCO artifacts, migrating older YAML schemas in memory."""
    spec = model_artifact_spec(case)
    paths = {
        key: hf_hub_download(
            spec["repo_id"], f"{spec['run_dir']}/{filename}", revision=MODEL_REVISION
        )
        for key, filename in {
            "checkpoint": "best_model_state_dict.pt",
            "config": spec["config_filename"],
            "normalizer": "normalizer_stats.pt",
        }.items()
    }
    with open(paths["config"], encoding="utf-8") as handle:
        config = resolve_config(
            yaml.safe_load(handle), paths["config"], conversion_mode="auto_inline"
        )
    args = NestedNamespace(**config)
    saved_stats = torch.load(paths["normalizer"], map_location="cpu", weights_only=True)
    network = spec["normalizer_network"]
    if network not in saved_stats:
        raise ValueError(f"Normalizer does not contain the selected network: {network}")
    normalizer = load_normalizer(args)
    normalizer.fit_from_dict(saved_stats[network])
    model = load_model(args)
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    state = _normalize_loaded_state_dict_keys(state)
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()}, strict=True
    )
    model.to(DEVICE).eval()
    return model, normalizer, args, paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=int, choices=(14, 30, 57, 118), default=118,
        help="IEEE bus system (default: 118)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experiment and dataset seed (default: 42)",
    )
    options = parser.parse_args(argv)
    if not 0 <= options.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return options


def main(argv=None):
    options = parse_args(argv)
    CASE = options.case
    SEED = options.seed
    DATASET_ROOTS, OUTPUT_DIR, tb_root = experiment_paths(SEED, CASE)
    for name, root in DATASET_ROOTS.items():
        for table in ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet"):
            if not (root / "raw" / table).exists():
                raise FileNotFoundError(f"{name}: missing {root / 'raw' / table}")
    original_case_path = PROJECT_DIR / "grids" / f"case{CASE}_original.m"
    if not original_case_path.is_file():
        raise FileNotFoundError(original_case_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TB_RUN_NAME = datetime.now().strftime(
        "impedance-multiplier-denoising-%Y%m%d-%H%M%S-%f"
    )
    TB_DIR = tb_root / TB_RUN_NAME
    DENOISED_Y_PATH = OUTPUT_DIR / "denoised_admittance.pt"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"IEEE-{CASE}; seed: {SEED}; device: {DEVICE}")
    print(f"Datasets: {DATASET_ROOTS}")
    print(f"Results: {OUTPUT_DIR}")
    print(f"TensorBoard: {TB_DIR}")
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "seed": SEED,
                "model_artifacts": model_artifact_spec(CASE),
                "dataset_roots": {k: str(v) for k, v in DATASET_ROOTS.items()},
                "epochs": EPOCHS,
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "multiplier_bounds": [MULTIPLIER_MIN, MULTIPLIER_MAX],
                "tensorboard_dir": str(TB_DIR),
                "notebook_section": 7,
            },
            indent=2,
        )
        + "\n"
    )

    model, normalizer, args, artifact_paths = load_model_bundle(CASE)
    print(f"{artifact_paths = }")

    required_tables = ["bus_data.parquet", "gen_data.parquet", "branch_data.parquet"]
    for dataset_name, root in DATASET_ROOTS.items():
        missing = [
            name for name in required_tables if not (root / "raw" / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"{dataset_name} is missing raw tables: {missing}")

    datasets = {
        dataset_name: HeteroGridDatasetDisk(
            root=str(root),
            data_normalizer=normalizer,
            transform=None,
            stream_partitions=getattr(args.data, "stream_partitions", "auto"),
        )
        for dataset_name, root in DATASET_ROOTS.items()
    }

    for dataset_name, dataset in datasets.items():
        print(
            f"Loaded {len(dataset):,} scenarios from {DATASET_ROOTS[dataset_name] / 'raw'}"
        )

    pf_transform = get_task_transforms(args)

    def prepare_graph(dataset, index):
        return pf_transform(dataset[index])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def predict_all(
        dataset_name,
        dataset,
        indices=None,
        graph_preparer=prepare_graph,
    ):
        if indices is None:
            indices = range(len(dataset))
        indices = list(indices)
        output_path = OUTPUT_DIR / f"{dataset_name}_pf_ground_truth_predictions.parquet"
        temp_path = output_path.with_suffix(".tmp.parquet")
        writer = None
        rows_written = 0
        seen_scenarios = set()
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(indices), BATCH_SIZE),
                desc=f"{dataset_name} PF",
            ):
                batch_indices = indices[start : start + BATCH_SIZE]
                graphs = [graph_preparer(dataset, index) for index in batch_indices]
                batch = Batch.from_data_list(graphs).to(DEVICE)
                output = model(batch)
                pred = output["bus"]
                if pred.size(1) <= max(VM_OUT, VA_OUT):
                    raise ValueError(
                        "Model bus output does not contain Vm and Va"
                    )

                # Graphkit keeps Va in radians during inverse_transform.
                # Datakit PF heads emit Vm/Va only (already in physical units).
                # The IEEE-118 transfer head additionally emits powers, whose
                # inverse scaling requires all four bus-output columns.
                has_power_outputs = pred.size(1) > max(PG_OUT, QG_OUT)
                if has_power_outputs:
                    normalizer.inverse_output(output, batch)
                normalizer.inverse_transform(batch)
                bus_y = batch["bus"].y
                gen_y = batch["gen"].y
                gen_to_bus = batch["gen", "connected_to", "bus"].edge_index[1]
                pg_target = torch.zeros(
                    bus_y.size(0),
                    device=DEVICE,
                    dtype=gen_y.dtype,
                )
                pg_target.index_add_(0, gen_to_bus, gen_y[:, 0])

                ptr = batch["bus"].ptr.cpu().tolist()
                frames = []
                for j, index in enumerate(batch_indices):
                    a, b = ptr[j], ptr[j + 1]
                    scenario_id = int(graphs[j].scenario_id.item())
                    if scenario_id in seen_scenarios:
                        raise ValueError(f"Duplicate scenario_id: {scenario_id}")
                    seen_scenarios.add(scenario_id)
                    mask = batch.mask_dict["bus"][a:b]
                    frames.append(
                        pd.DataFrame(
                            {
                                "dataset_name": np.full(b - a, dataset_name),
                                "dataset_index": np.full(b - a, index, dtype=np.int32),
                                "scenario_id": np.full(
                                    b - a, scenario_id, dtype=np.int64
                                ),
                                "bus_index": np.arange(b - a, dtype=np.int32),
                                "pq_bus": batch.mask_dict["PQ"][a:b].cpu().numpy(),
                                "pv_bus": batch.mask_dict["PV"][a:b].cpu().numpy(),
                                "ref_bus": batch.mask_dict["REF"][a:b].cpu().numpy(),
                                "vm_masked": mask[:, VM_H].cpu().numpy(),
                                "va_masked": mask[:, VA_H].cpu().numpy(),
                                "vm_gt_pu": bus_y[a:b, VM_H].cpu().numpy(),
                                "vm_pred_pu": pred[a:b, VM_OUT].cpu().numpy(),
                                "va_gt_deg": torch.rad2deg(bus_y[a:b, VA_H])
                                .cpu()
                                .numpy(),
                                "va_pred_deg": torch.rad2deg(pred[a:b, VA_OUT])
                                .cpu()
                                .numpy(),
                                "pg_gt_mw": pg_target[a:b].cpu().numpy(),
                                "pg_pred_mw": (pred[a:b, PG_OUT].cpu().numpy()
                                               if has_power_outputs else np.full(b - a, np.nan)),
                                "qg_gt_mvar": bus_y[a:b, QG_H].cpu().numpy(),
                                "qg_pred_mvar": (pred[a:b, QG_OUT].cpu().numpy()
                                                 if has_power_outputs else np.full(b - a, np.nan)),
                            }
                        )
                    )
                table = pa.Table.from_pandas(
                    pd.concat(frames, ignore_index=True),
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                rows_written += table.num_rows
        if writer is None:
            raise RuntimeError("No PF rows were written")
        writer.close()
        temp_path.replace(output_path)
        print(f"Saved {rows_written:,} rows to {output_path}")
        return output_path

    key_columns = ["dataset_index", "scenario_id", "bus_index"]

    error_specs = {
        "Vm": ("vm_gt_pu", "vm_pred_pu", "p.u."),
        "Va": ("va_gt_deg", "va_pred_deg", "degrees"),
    }
    RELATIVE_ERROR_EPS = 1e-8

    def relative_error(ground_truth, prediction, epsilon=RELATIVE_ERROR_EPS):
        """Return signed relative error, excluding zero and near-zero targets."""
        ground_truth = np.asarray(ground_truth, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        denominator = np.abs(ground_truth)
        result = np.full(ground_truth.shape, np.nan, dtype=np.float64)
        valid = (
            np.isfinite(ground_truth)
            & np.isfinite(prediction)
            & (denominator > epsilon)
        )
        np.divide(prediction - ground_truth, denominator, out=result, where=valid)
        return result

    def prediction_error(ground_truth, prediction):
        """Return signed prediction error in the target's original unit."""
        ground_truth = np.asarray(ground_truth, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        return prediction - ground_truth

    def histogram_bins(errors):
        """Build shared bins after checking that each series has finite values."""
        if any(values.size == 0 for values in errors.values()):
            raise ValueError("Cannot plot an error series with no finite values")
        lower = min(np.quantile(values, 0.005) for values in errors.values())
        upper = max(np.quantile(values, 0.995) for values in errors.values())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        return np.linspace(lower, upper, 101)

    def histogram_fractions(values, bins):
        """Return the fraction of displayed observations in each bin."""
        counts, _ = np.histogram(values, bins=bins)
        displayed_count = counts.sum()
        if displayed_count == 0:
            raise ValueError("No observations fall within the histogram range")
        return counts / displayed_count

    plot_rows = (
        ("relative", "Signed relative error"),
        ("prediction", "Prediction error"),
    )

    ORIGINAL_NAME = f"case{CASE}_original"
    AGED_NAME = f"case{CASE}_aged"
    HIDDEN_AGING_NAME = f"case{CASE}_aged_original_impedance"
    original_dataset = datasets[ORIGINAL_NAME]
    aged_dataset = datasets[AGED_NAME]
    edge_type = ("bus", "connects", "bus")
    admittance_columns = slice(YFF_TT_R, YFT_TF_I + 1)

    def group_indices_by_load(dataset):
        groups = {}
        for index, load_index in enumerate(dataset.load_scenarios.tolist()):
            groups.setdefault(int(load_index), []).append(index)
        return groups

    def scenario_signature(graph):
        branch_on = (graph[edge_type].edge_attr[:, B_ON] > 0.5).to(torch.uint8)
        generator_on = (graph["gen"].x[:, G_ON] > 0.5).to(torch.uint8)
        generator_cost = graph["gen"].x[:, C0_H : C2_H + 1]
        return (
            branch_on.cpu().numpy().tobytes(),
            generator_on.cpu().numpy().tobytes(),
            generator_cost.cpu().numpy().tobytes(),
        )

    original_groups = group_indices_by_load(original_dataset)
    aged_groups = group_indices_by_load(aged_dataset)
    hidden_aging_pairs = {}

    def match_in_order(aged_indices, original_indices):
        """Return the longest order-preserving match of scenario signatures."""
        aged_signatures = [scenario_signature(aged_dataset[i]) for i in aged_indices]
        original_signatures = [
            scenario_signature(original_dataset[i]) for i in original_indices
        ]
        n_aged, n_original = len(aged_indices), len(original_indices)
        lengths = np.zeros((n_aged + 1, n_original + 1), dtype=np.int8)
        for i in range(n_aged):
            for j in range(n_original):
                if aged_signatures[i] == original_signatures[j]:
                    lengths[i + 1, j + 1] = lengths[i, j] + 1
                else:
                    lengths[i + 1, j + 1] = max(lengths[i, j + 1], lengths[i + 1, j])
        pairs = {}
        i, j = n_aged, n_original
        while i and j:
            if aged_signatures[i - 1] == original_signatures[j - 1]:
                pairs[aged_indices[i - 1]] = original_indices[j - 1]
                i -= 1
                j -= 1
            elif lengths[i - 1, j] >= lengths[i, j - 1]:
                i -= 1
            else:
                j -= 1
        return pairs

    for load_index, aged_indices in aged_groups.items():
        original_indices = original_groups.get(load_index, [])
        hidden_aging_pairs.update(match_in_order(aged_indices, original_indices))

    hidden_aging_indices = sorted(hidden_aging_pairs)
    excluded_count = len(aged_dataset) - len(hidden_aging_indices)
    print(f"Matched {len(hidden_aging_indices):,} aged scenarios")
    print(f"Excluded {excluded_count:,} aged scenarios without an original counterpart")

    def prepare_hidden_aging_graph(dataset, aged_index):
        aged_graph = dataset[aged_index]
        original_index = hidden_aging_pairs[aged_index]
        original_graph = original_dataset[original_index]

        if not torch.equal(
            aged_graph[edge_type].edge_index, original_graph[edge_type].edge_index
        ):
            raise ValueError(f"Edge ordering differs for aged scenario {aged_index}")
        if scenario_signature(aged_graph) != scenario_signature(original_graph):
            raise ValueError(f"Scenario pairing differs for aged scenario {aged_index}")
        if not np.isclose(float(aged_graph.baseMVA), float(original_graph.baseMVA)):
            raise ValueError(f"baseMVA differs for aged scenario {aged_index}")

        # Both graphs are already normalized with the same saved normalizer.
        # Replace only the impedance-derived model inputs; retain all aged targets.
        aged_graph[edge_type].edge_attr[:, admittance_columns] = original_graph[
            edge_type
        ].edge_attr[:, admittance_columns]
        return pf_transform(aged_graph)

    def available_dataset_indices(dataset):
        """Return actual saved scenario IDs, including datasets with solver gaps."""
        prefix = "data_index_"
        return sorted(
            int(path.stem.removeprefix(prefix))
            for path in Path(dataset.processed_dir).glob(f"{prefix}*.pt")
        )

    training_indices = available_dataset_indices(aged_dataset)
    original_training_indices = available_dataset_indices(original_dataset)
    if not training_indices:
        raise RuntimeError("The aged dataset contains no scenarios")
    if not original_training_indices:
        raise RuntimeError("The original dataset contains no scenarios")

    # Every untransformed scenario has the same complete directed-edge ordering. The
    # first half stores from->to edges and the second half their to->from counterparts.
    reference_graph = aged_dataset.get(training_indices[0])
    reference_edge_index = reference_graph[edge_type].edge_index.clone()
    n_directed_edges = reference_edge_index.size(1)
    n_lines = n_directed_edges // 2
    if not torch.equal(
        reference_edge_index[:, :n_lines],
        reference_edge_index.flip(0)[:, n_lines:],
    ):
        raise ValueError("Forward and reverse branch ordering is not aligned")

    def mean_admittance(dataset, indices, description):
        """Average Y only while each directed branch is in service."""
        total = torch.zeros((n_directed_edges, 4), dtype=torch.float64)
        total_squared = torch.zeros_like(total)
        active_count = torch.zeros(n_directed_edges, dtype=torch.float64)
        for index in tqdm(indices, desc=description):
            graph = dataset.get(index)
            if not torch.equal(graph[edge_type].edge_index, reference_edge_index):
                raise ValueError(
                    f"Full edge ordering differs for dataset index {index}"
                )
            edge_attr = graph[edge_type].edge_attr
            values = edge_attr[:, admittance_columns].double()
            active = edge_attr[:, B_ON] > 0.5
            total[active] += values[active]
            total_squared[active] += values[active].square()
            active_count[active] += 1
        if (active_count == 0).any():
            missing = (active_count == 0).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Directed branches never observed in service: {missing}"
            )
        mean = total / active_count[:, None]
        variance = (total_squared / active_count[:, None] - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float()

    # Average active observations: [2 * n_lines, 4] (Yff_r, Yff_i, Yft_r, Yft_i).
    initial_admittance, original_admittance_std = mean_admittance(
        original_dataset,
        original_training_indices,
        "Averaging original Y",
    )
    actual_aged_admittance, aged_admittance_std = mean_admittance(
        aged_dataset,
        training_indices,
        "Averaging aged Y (diagnostic only)",
    )

    # Datakit normalizes Y by baseMVA_orig/baseMVA. For an ordinary line,
    # Ytf = Yft = -1/(R+jX), so R+jX = scale/(-Yft_normalized).
    admittance_scale = float(normalizer.baseMVA_orig / normalizer.baseMVA)
    # initial_[self|cross]_[forward|reverse] are complex vectors of length n_lines.
    initial_self_forward = torch.complex(
        initial_admittance[:n_lines, 0],
        initial_admittance[:n_lines, 1],
    )
    initial_cross_forward = torch.complex(
        initial_admittance[:n_lines, 2],
        initial_admittance[:n_lines, 3],
    )
    initial_self_reverse = torch.complex(
        initial_admittance[n_lines:, 0],
        initial_admittance[n_lines:, 1],
    )
    initial_cross_reverse = torch.complex(
        initial_admittance[n_lines:, 2],
        initial_admittance[n_lines:, 3],
    )
    initial_series_y = -initial_cross_forward
    initial_impedance = admittance_scale / initial_series_y
    forward_shunt = initial_self_forward - initial_series_y
    reverse_shunt = initial_self_reverse - initial_series_y

    # Read the original MATPOWER branch metadata to identify learnable lines:
    # Estimate R/X for in-service branches without non-unity taps or phase shifts.
    original_case_path = PROJECT_DIR / "grids" / f"case{CASE}_original.m"
    original_case_text = original_case_path.read_text()
    branch_match = re.search(
        r"mpc\.branch\s*=\s*\[(.*?)\];",
        original_case_text,
        re.DOTALL,
    )

    # Mark the branch as learnable only if it is in service and is not represented as a transformer.
    if branch_match is None:
        raise ValueError(f"Missing mpc.branch table in {original_case_path}")
    branch_fields = []
    for line in branch_match.group(1).splitlines():
        fields = line.partition("%")[0].strip().rstrip(";").split()
        if fields:
            branch_fields.append(fields)
    if len(branch_fields) != n_lines:
        raise ValueError("MATPOWER branch count differs from the graph")
    learnable_line_mask = torch.tensor(
        [
            float(fields[10]) > 0
            and not (
                (float(fields[8]) != 0.0 and float(fields[8]) != 1.0)
                or float(fields[9]) != 0.0
            )
            for fields in branch_fields
        ],
        dtype=torch.bool,
    )
    # duplicate for from bus and to bus
    learnable_directed_mask = torch.cat([learnable_line_mask, learnable_line_mask])

    if not learnable_line_mask.any():
        raise RuntimeError("No eligible transmission lines are available")
    initial_resistance = initial_impedance.real.float()
    initial_reactance = initial_impedance.imag.float()
    resistance_multiplier = torch.nn.Parameter(torch.ones_like(initial_resistance))
    reactance_multiplier = torch.nn.Parameter(torch.ones_like(initial_reactance))
    multiplier_parameters = [resistance_multiplier, reactance_multiplier]
    optimizer = torch.optim.Adam(multiplier_parameters, lr=LR)

    def estimated_impedance_components():
        """Map dimensionless ageing multipliers to per-unit R/X."""
        return (
            initial_resistance * resistance_multiplier,
            initial_reactance * reactance_multiplier,
        )

    def impedance_from_admittance(admittance):
        """Recover ordinary-line R/X from normalized forward transfer admittance."""
        cross = torch.complex(admittance[:n_lines, 2], admittance[:n_lines, 3])
        return admittance_scale / (-cross)

    actual_aged_impedance = impedance_from_admittance(actual_aged_admittance)
    actual_aged_resistance = actual_aged_impedance.real
    actual_aged_reactance = actual_aged_impedance.imag

    def admittance_from_estimated_impedance():
        """Differentiably convert bounded R/X multipliers to normalized Y inputs."""
        estimated_resistance, estimated_reactance = estimated_impedance_components()
        impedance = torch.complex(estimated_resistance, estimated_reactance)
        series_y = admittance_scale / impedance
        estimated = initial_admittance.clone()

        forward_values = torch.stack(
            [
                (series_y + forward_shunt).real,
                (series_y + forward_shunt).imag,
                -series_y.real,
                -series_y.imag,
            ],
            dim=1,
        )
        reverse_values = torch.stack(
            [
                (series_y + reverse_shunt).real,
                (series_y + reverse_shunt).imag,
                -series_y.real,
                -series_y.imag,
            ],
            dim=1,
        )
        estimated[:n_lines][learnable_line_mask] = forward_values[learnable_line_mask]
        estimated[n_lines:][learnable_line_mask] = reverse_values[learnable_line_mask]
        return estimated

    # Freeze GENCO. Gradients pass through it to R/X, but model weights never update.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def prepare_denoising_graph(aged_index, estimated_admittance):
        """Insert the shared estimated line Y values into graph, then apply the standard PF mask."""
        graph = aged_dataset[aged_index].clone()
        if not torch.equal(graph[edge_type].edge_index, reference_edge_index):
            raise ValueError(f"Full edge ordering differs for aged index {aged_index}")
        edge_attr = graph[edge_type].edge_attr.clone()
        current_y = edge_attr[:, admittance_columns].clone()
        current_y[learnable_directed_mask] = estimated_admittance[
            learnable_directed_mask
        ]
        edge_attr[:, admittance_columns] = current_y
        graph[edge_type].edge_attr = edge_attr
        return pf_transform(graph)

    def masked_pf_model_loss(batch, output):
        """The checkpoint's MaskedBusMSE: masked Vm/Va in normalized space."""
        target = batch["bus"].y[:, [VM_H, VA_H]]
        prediction = output["bus"][:, [VM_OUT, VA_OUT]]
        supervised_mask = batch.mask_dict["bus"][:, [VM_H, VA_H]]
        target_count = int(supervised_mask.sum().item())
        return (
            F.mse_loss(prediction[supervised_mask], target[supervised_mask]),
            target_count,
        )

    def evaluate_model_mse(indices, description):
        """Evaluate one fixed R/X state over all requested scenarios."""
        loss_sum = 0.0
        target_count = 0
        with torch.no_grad():
            for start in tqdm(
                range(0, len(indices), BATCH_SIZE),
                desc=description,
                leave=False,
            ):
                batch_indices = indices[start : start + BATCH_SIZE]
                current_admittance = admittance_from_estimated_impedance()
                graphs = [
                    prepare_denoising_graph(index, current_admittance)
                    for index in batch_indices
                ]
                batch = Batch.from_data_list(graphs).to(DEVICE)
                model_loss, batch_target_count = masked_pf_model_loss(
                    batch, model(batch)
                )
                loss_sum += model_loss.item() * batch_target_count
                target_count += batch_target_count
        return loss_sum / target_count

    def impedance_metrics():
        """Return diagnostic errors without exposing aged Y to optimization."""
        estimated_resistance, estimated_reactance = estimated_impedance_components()
        resistance_mismatch = actual_aged_resistance - estimated_resistance.detach()
        resistance_mismatch = resistance_mismatch[learnable_line_mask]
        reactance_mismatch = actual_aged_reactance - estimated_reactance.detach()
        reactance_mismatch = reactance_mismatch[learnable_line_mask]
        combined = torch.cat([resistance_mismatch, reactance_mismatch])
        return resistance_mismatch, reactance_mismatch, combined

    def log_epoch_metrics(writer, step, model_mse, gradient_norms=None):
        resistance_mismatch, reactance_mismatch, combined = impedance_metrics()
        resistance_values = resistance_multiplier.detach()[learnable_line_mask]
        reactance_values = reactance_multiplier.detach()[learnable_line_mask]
        mismatch_mae = combined.abs().mean().item()
        mismatch_rmse = combined.square().mean().sqrt().item()
        model_rmse = model_mse**0.5
        writer.add_scalar("epoch/model_rmse", model_rmse, step)
        writer.add_scalar("epoch/impedance_mismatch_mae", mismatch_mae, step)
        writer.add_scalar("epoch/impedance_mismatch_rmse", mismatch_rmse, step)
        writer.add_scalar(
            "epoch/resistance_mismatch_rmse",
            resistance_mismatch.square().mean().sqrt().item(),
            step,
        )
        writer.add_scalar(
            "epoch/reactance_mismatch_rmse",
            reactance_mismatch.square().mean().sqrt().item(),
            step,
        )
        writer.add_histogram("epoch/resistance_mismatch", resistance_mismatch, step)
        writer.add_histogram("epoch/reactance_mismatch", reactance_mismatch, step)
        writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]["lr"], step)
        if gradient_norms is not None:
            writer.add_scalar(
                "epoch/gradient_norm",
                float(np.linalg.norm(gradient_norms)),
                step,
            )
        writer.flush()
        return (
            model_rmse,
            mismatch_mae,
            mismatch_rmse,
            resistance_values.median().item(),
            reactance_values.median().item(),
            resistance_values.max().item(),
            reactance_values.max().item(),
        )

    with SummaryWriter(log_dir=str(TB_DIR)) as writer:
        last_model_mse = None
        for epoch in range(EPOCHS):
            epoch_generator = torch.Generator().manual_seed(SEED + epoch)
            order = torch.randperm(
                len(training_indices),
                generator=epoch_generator,
            )
            order = order.tolist()
            epoch_loss_sum = 0.0
            epoch_target_count = 0
            optimizer.zero_grad()

            progress = tqdm(
                range(0, len(order), BATCH_SIZE),
                desc=f"Multiplier R/X epoch {epoch + 1}/{EPOCHS}",
                leave=False,
            )
            for start in progress:
                positions = order[start : start + BATCH_SIZE]
                batch_indices = [training_indices[position] for position in positions]
                current_admittance = admittance_from_estimated_impedance()
                graphs = [
                    prepare_denoising_graph(index, current_admittance)
                    for index in batch_indices
                ]
                # torchgemoetric.data.Batch:
                # A data object describing a batch of graphs as one big (disconnected) graph.
                batch = Batch.from_data_list(graphs).to(DEVICE)

                output = model(batch)
                model_loss, target_count = masked_pf_model_loss(batch, output)
                # Accumulate the exact full-dataset mean gradient. Stepping here would
                # apply LR once per mini-batch and make the result batch-size dependent.
                (model_loss * target_count).backward()

                epoch_loss_sum += model_loss.item() * target_count
                epoch_target_count += target_count
                progress.set_postfix(model_rmse=f"{model_loss.item() ** 0.5:.3e}")

            epoch_model_mse = epoch_loss_sum / epoch_target_count
            gradient_norms = []
            for parameter in multiplier_parameters:
                parameter.grad.div_(epoch_target_count)
                parameter.grad[~learnable_line_mask] = 0.0
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite R/X multiplier gradient")
                gradient_norms.append(parameter.grad[learnable_line_mask].norm().item())
            if epoch == 0:
                log_epoch_metrics(writer, 0, epoch_model_mse)

            optimizer.step()
            with torch.no_grad():
                for parameter in multiplier_parameters:
                    parameter.clamp_(MULTIPLIER_MIN, MULTIPLIER_MAX)
                    parameter[~learnable_line_mask] = 1.0

            candidate_mse = evaluate_model_mse(
                training_indices,
                f"Checking multiplier epoch {epoch + 1}",
            )
            last_model_mse = candidate_mse
            (
                model_rmse,
                mismatch_mae,
                mismatch_rmse,
                resistance_median,
                reactance_median,
                resistance_max,
                reactance_max,
            ) = log_epoch_metrics(writer, epoch + 1, candidate_mse, gradient_norms)
            print(
                f"Epoch {epoch + 1:02d}: model RMSE={model_rmse:.6e}, "
                f"R/X mismatch MAE={mismatch_mae:.6e}, "
                f"R/X mismatch RMSE={mismatch_rmse:.6e}, "
                f"median multipliers=({resistance_median:.4f}, {reactance_median:.4f}), "
                f"max multipliers=({resistance_max:.4f}, {reactance_max:.4f})"
            )

    estimated_resistance, estimated_reactance = estimated_impedance_components()
    final_admittance = admittance_from_estimated_impedance().detach().cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case": CASE,
            "seed": SEED,
            "model_artifacts": model_artifact_spec(CASE),
            "dataset_roots": {key: str(path) for key, path in DATASET_ROOTS.items()},
            "edge_index": reference_edge_index,
            "learnable_line_mask": learnable_line_mask,
            "estimated_resistance_pu": estimated_resistance.detach().cpu(),
            "estimated_reactance_pu": estimated_reactance.detach().cpu(),
            "estimated_resistance_multiplier": resistance_multiplier.detach().cpu(),
            "estimated_reactance_multiplier": reactance_multiplier.detach().cpu(),
            "initial_resistance_pu": initial_resistance,
            "initial_reactance_pu": initial_reactance,
            "actual_aged_resistance_pu": actual_aged_resistance,
            "actual_aged_reactance_pu": actual_aged_reactance,
            "estimated_admittance_normalized": final_admittance,
            "initial_admittance_normalized": initial_admittance,
            "actual_aged_admittance_normalized": actual_aged_admittance,
            "original_scenario_std": original_admittance_std,
            "aged_scenario_std": aged_admittance_std,
            "training_indices": torch.as_tensor(training_indices),
            "epochs": EPOCHS,
            "learning_rate": LR,
            "optimizer": "Adam",
            "multiplier_bounds": (MULTIPLIER_MIN, MULTIPLIER_MAX),
            "constraint": "R and X are initial values times bounded ageing multipliers",
        },
        DENOISED_Y_PATH,
    )
    print(f"Optimized {int(learnable_line_mask.sum()):,} non-transformer lines")
    print(f"Saved bounded multiplier-space R/X and recovered Y to {DENOISED_Y_PATH}")
    print(f"TensorBoard logs: {TB_DIR}")
    print(f"Launch with: tensorboard --logdir {TB_DIR}")

    # Compare the original, true aged, and GENCO-estimated line impedances.
    impedance_result = torch.load(
        DENOISED_Y_PATH,
        map_location="cpu",
        weights_only=True,
    )
    plot_mask = impedance_result["learnable_line_mask"].bool()
    branch_rows = torch.arange(1, plot_mask.numel() + 1)[plot_mask].numpy()

    def plot_impedance_component(component, ylabel, output_name):
        original = impedance_result[f"initial_{component}_pu"][plot_mask].numpy()
        aged = impedance_result[f"actual_aged_{component}_pu"][plot_mask].numpy()
        estimated = impedance_result[f"estimated_{component}_pu"][plot_mask].numpy()

        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        ax.plot(
            branch_rows, original, label="Original", color="tab:blue", linewidth=1.5
        )
        ax.plot(
            branch_rows,
            aged,
            label="Aged (ground truth)",
            color="tab:orange",
            linewidth=1.5,
        )
        ax.plot(
            branch_rows,
            estimated,
            label="Estimated",
            color="tab:green",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_xlabel("MATPOWER branch row")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

        output_path = OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)

    plot_impedance_component(
        "resistance",
        "Resistance R (p.u.)",
        "resistance_comparison.pdf",
    )
    plot_impedance_component(
        "reactance",
        "Reactance X (p.u.)",
        "reactance_comparison.pdf",
    )

    # Parity plots compare estimated impedance against the aged ground truth.
    def plot_impedance_parity(component, symbol, output_name):
        aged = impedance_result[f"actual_aged_{component}_pu"][plot_mask].numpy()
        estimated = impedance_result[f"estimated_{component}_pu"][plot_mask].numpy()
        valid = np.isfinite(aged) & np.isfinite(estimated)
        if not valid.any():
            raise ValueError(f"No valid {symbol} values are available to plot")

        aged = aged[valid]
        estimated = estimated[valid]
        lower = min(aged.min(), estimated.min())
        upper = max(aged.max(), estimated.max())
        if lower == upper:
            padding = max(abs(lower) * 0.05, 1e-12)
            lower -= padding
            upper += padding

        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.scatter(
            estimated,
            aged,
            s=22,
            color="tab:blue",
            alpha=0.8,
            label="Learnable lines",
        )
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Perfect estimate (y = x)",
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Estimated {symbol} (p.u.)")
        ax.set_ylabel(f"Aged ground-truth {symbol} (p.u.)")
        ax.grid(alpha=0.25)
        ax.legend()

        output_path = OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved parity plot to {output_path}")
        plt.close(fig)

    plot_impedance_parity(
        "resistance",
        "R",
        "resistance_estimated_vs_aged.pdf",
    )
    plot_impedance_parity(
        "reactance",
        "X",
        "reactance_estimated_vs_aged.pdf",
    )

    # Plot normalized impedance residuals against the aged ground truth.
    IMPEDANCE_DENOMINATOR_EPS = 1e-12

    def plot_impedance_relative_residual(component, symbol, output_name):
        aged = impedance_result[f"actual_aged_{component}_pu"][plot_mask].numpy()
        estimated = impedance_result[f"estimated_{component}_pu"][plot_mask].numpy()
        valid = (
            np.isfinite(aged)
            & np.isfinite(estimated)
            & (np.abs(aged) > IMPEDANCE_DENOMINATOR_EPS)
        )
        if not valid.any():
            raise ValueError(f"No valid nonzero {symbol} values are available to plot")

        aged = aged[valid]
        estimated = estimated[valid]
        relative_residual = (aged - estimated) / aged

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(
            aged,
            relative_residual,
            s=22,
            color="tab:blue",
            alpha=0.8,
            label="Learnable lines",
        )
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Perfect estimate (relative residual = 0)",
        )
        ax.set_xlabel(f"Aged ground-truth {symbol} (p.u.)")
        ax.set_ylabel(rf"$({symbol} - \hat{{{symbol}}}) / {symbol}$")
        ax.grid(alpha=0.25)
        ax.legend()

        output_path = OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved normalized-residual plot to {output_path}")
        plt.close(fig)

    plot_impedance_relative_residual(
        "resistance",
        "R",
        "resistance_relative_residual_vs_ground_truth.pdf",
    )
    plot_impedance_relative_residual(
        "reactance",
        "X",
        "reactance_relative_residual_vs_ground_truth.pdf",
    )

    # Plot the normalized-error distributions for the GENCO impedance estimates.
    impedance_error_components = (
        ("resistance", "R", "tab:blue"),
        ("reactance", "X", "tab:orange"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, (component, symbol, color) in zip(axes, impedance_error_components):
        aged = impedance_result[f"actual_aged_{component}_pu"][plot_mask].numpy()
        estimated = impedance_result[f"estimated_{component}_pu"][plot_mask].numpy()
        valid = (
            np.isfinite(aged)
            & np.isfinite(estimated)
            & (np.abs(aged) > IMPEDANCE_DENOMINATOR_EPS)
        )
        if not valid.any():
            raise ValueError(f"No valid nonzero {symbol} values are available")

        relative_residual = (aged[valid] - estimated[valid]) / aged[valid]
        bins = histogram_bins({component: relative_residual})
        fractions = histogram_fractions(relative_residual, bins)
        ax.stairs(
            fractions,
            bins,
            color=color,
            linewidth=1.5,
            fill=True,
            alpha=0.35,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel(rf"$({symbol} - \hat{{{symbol}}}) / {symbol}$")
        ax.set_ylabel("Fraction of learnable lines")
        ax.grid(alpha=0.25)

    error_distribution_path = OUTPUT_DIR / "impedance_relative_error_distributions.pdf"
    fig.savefig(error_distribution_path, dpi=300, bbox_inches="tight")
    print(f"Saved error-distribution plot to {error_distribution_path}")
    plt.close(fig)

    # Same 2x2 layout as In [13]: Vm/Va relative errors and prediction errors.
    # Compare GENCO errors using aged observations and three impedance inputs.
    # Load the saved estimate: rerunning optimization is unnecessary.
    estimated_pf_result = torch.load(
        OUTPUT_DIR / "denoised_admittance.pt", map_location="cpu", weights_only=True
    )
    estimated_pf_y = estimated_pf_result["estimated_admittance_normalized"]
    estimated_pf_edges = estimated_pf_result["edge_index"]
    estimated_pf_mask = estimated_pf_result["learnable_line_mask"].bool()
    estimated_pf_directed_mask = torch.cat([estimated_pf_mask, estimated_pf_mask])
    if estimated_pf_y.shape != (estimated_pf_edges.size(1), 4):
        raise ValueError("Saved admittance shape does not match saved edges")
    if estimated_pf_directed_mask.numel() != estimated_pf_edges.size(1):
        raise ValueError("Saved learnability mask does not match saved edges")
    if not torch.isfinite(estimated_pf_y).all():
        raise ValueError("Saved estimated admittance contains non-finite values")

    def prepare_estimated_pf_graph(dataset, index):
        graph = dataset[index].clone()
        if not torch.equal(graph[edge_type].edge_index.cpu(), estimated_pf_edges):
            raise ValueError(
                f"Saved estimate edge ordering differs for scenario {index}"
            )
        # Use exactly the substitution used during estimation; fixed branches
        # and all aged observations, targets, and PF masks retain their values.
        edge_attr = graph[edge_type].edge_attr.clone()
        current_y = edge_attr[:, admittance_columns].clone()
        current_y[estimated_pf_directed_mask] = estimated_pf_y.to(current_y)[
            estimated_pf_directed_mask
        ]
        edge_attr[:, admittance_columns] = current_y
        graph[edge_type].edge_attr = edge_attr
        return pf_transform(graph)

    # Evaluate identical matched scenarios for every curve. Recompute predictions
    # so an older cached run cannot hide changes to the data or saved estimate.
    estimated_pf_indices = sorted(hidden_aging_indices)
    if not estimated_pf_indices:
        raise ValueError("No matched scenarios are available for comparison")
    estimated_pf_runs = {
        "Aged pf - Aged impedance": ("aged_comparison_true", prepare_graph),
        "Aged pf - Original impedance": (
            "aged_comparison_original",
            prepare_hidden_aging_graph,
        ),
        "Aged pf - Estimated impedance": (
            "aged_comparison_estimated",
            prepare_estimated_pf_graph,
        ),
    }
    estimated_pf_tables = {}
    model.eval()
    for label, (run_name, graph_preparer) in estimated_pf_runs.items():
        path = predict_all(
            run_name,
            aged_dataset,
            indices=estimated_pf_indices,
            graph_preparer=graph_preparer,
        )
        estimated_pf_tables[label] = (
            pd.read_parquet(path).sort_values(key_columns).reset_index(drop=True)
        )

    estimated_pf_reference = estimated_pf_tables["Aged pf - Aged impedance"]
    for label, table in estimated_pf_tables.items():
        if table.duplicated(key_columns).any():
            raise ValueError(f"Duplicate prediction row keys for {label}")
        if not table[key_columns].equals(estimated_pf_reference[key_columns]):
            raise ValueError(f"Comparison row keys differ for {label}")
        for variable, (gt_column, _, _) in error_specs.items():
            if not np.allclose(table[gt_column], estimated_pf_reference[gt_column]):
                raise ValueError(f"Ground truths differ for {label}: {variable}")
            mask_column = {"Vm": "vm_masked", "Va": "va_masked"}[variable]
            if not table[mask_column].equals(estimated_pf_reference[mask_column]):
                raise ValueError(f"PF masks differ for {label}: {variable}")

    estimated_pf_colors = {
        "Aged pf - Aged impedance": "tab:orange",
        "Aged pf - Original impedance": "tab:red",
        "Aged pf - Estimated impedance": "tab:green",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    estimated_pf_metrics = []
    for col, (variable, (gt_column, pred_column, unit)) in enumerate(
        error_specs.items()
    ):
        for row, (kind, error_label) in enumerate(plot_rows):
            error_function = relative_error if kind == "relative" else prediction_error
            errors = {}
            for label, table in estimated_pf_tables.items():
                values = error_function(table[gt_column], table[pred_column])
                errors[label] = values[np.isfinite(values)]
            bins = histogram_bins(errors)
            ax = axes[row, col]
            for label, values in errors.items():
                ax.stairs(
                    histogram_fractions(values, bins),
                    bins,
                    label=label,
                    color=estimated_pf_colors[label],
                    linewidth=1.5,
                )
            ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
            x_unit = "dimensionless" if kind == "relative" else unit
            ax.set_xlabel(f"{error_label} ({x_unit})")
            ax.set_ylabel("Fraction of observations")
            ax.grid(alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

        # Metrics use all finite errors, including histogram tails. Report both
        # all buses (as in In [13]) and only quantities hidden by the PF task.
        for label, table in estimated_pf_tables.items():
            values = prediction_error(table[gt_column], table[pred_column])
            masked = table[{"Vm": "vm_masked", "Va": "va_masked"}[variable]].to_numpy(
                bool
            )
            for scope, selection in (
                ("all", np.ones(len(values), dtype=bool)),
                ("masked", masked),
            ):
                selected = values[selection & np.isfinite(values)]
                estimated_pf_metrics.append(
                    {
                        "input": label,
                        "variable": variable,
                        "scope": scope,
                        "unit": unit,
                        "count": selected.size,
                        "MAE": np.abs(selected).mean() if selected.size else np.nan,
                        "RMSE": (
                            np.sqrt(np.mean(selected**2)) if selected.size else np.nan
                        ),
                    }
                )

    estimated_pf_histogram_path = (
        OUTPUT_DIR / "estimated_impedance_pf_error_histograms.pdf"
    )
    fig.savefig(estimated_pf_histogram_path, dpi=300, bbox_inches="tight")
    print(f"Saved histogram to {estimated_pf_histogram_path}")
    pd.DataFrame(estimated_pf_metrics).to_csv(
        OUTPUT_DIR / "estimated_impedance_pf_error_metrics.csv", index=False
    )
    plt.close(fig)

    # One error histogram per physical branch, with 1,000 scenarios in this dataset.
    branch_pf_predictions = pd.read_parquet(
        OUTPUT_DIR / "aged_comparison_estimated_pf_ground_truth_predictions.parquet"
    )
    branch_pf_result = torch.load(
        OUTPUT_DIR / "denoised_admittance.pt", map_location="cpu", weights_only=True
    )
    branch_pf_edges = branch_pf_result["edge_index"].numpy()
    branch_pf_count = branch_pf_edges.shape[1] // 2
    if not np.array_equal(
        branch_pf_edges[:, :branch_pf_count], branch_pf_edges[::-1, branch_pf_count:]
    ):
        raise ValueError("Saved branch directions are not aligned")
    if branch_pf_predictions.duplicated(
        ["dataset_index", "scenario_id", "bus_index"]
    ).any():
        raise ValueError("Duplicate scenario/bus prediction rows")
    branch_pf_sample_count = branch_pf_predictions["dataset_index"].nunique()
    branch_pf_groups = dict(tuple(branch_pf_predictions.groupby("bus_index")))
    branch_pf_scenarios = set(branch_pf_predictions["dataset_index"])
    for bus_index in np.unique(branch_pf_edges):
        if (
            bus_index not in branch_pf_groups
            or set(branch_pf_groups[bus_index]["dataset_index"]) != branch_pf_scenarios
        ):
            raise ValueError(f"Incomplete scenario coverage at bus index {bus_index}")

    branch_pf_cmap = plt.get_cmap("turbo")
    branch_pf_norm = plt.Normalize(1, max(branch_pf_count, 2))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    branch_pf_summary = []
    for col, (variable, (gt_column, pred_column, unit)) in enumerate(
        error_specs.items()
    ):
        # Use identical bins for the from/to panels of each voltage quantity.
        endpoint_errors = {}
        for endpoint, endpoint_name in enumerate(("From bus", "To bus")):
            for branch in range(branch_pf_count):
                bus_index = int(branch_pf_edges[endpoint, branch])
                table = branch_pf_groups[bus_index]
                errors = prediction_error(table[gt_column], table[pred_column])
                errors = errors[np.isfinite(errors)]
                endpoint_errors[(endpoint, branch)] = errors
                branch_pf_summary.append(
                    {
                        "branch_row": branch + 1,
                        "endpoint": endpoint_name,
                        "bus_index": bus_index,
                        "variable": variable,
                        "unit": unit,
                        "count": errors.size,
                        "mean_error": errors.mean() if errors.size else np.nan,
                        "std_error": errors.std() if errors.size else np.nan,
                        "RMSE": np.sqrt(np.mean(errors**2)) if errors.size else np.nan,
                    }
                )
        bins = histogram_bins(endpoint_errors)
        for endpoint, endpoint_name in enumerate(("From bus", "To bus")):
            ax = axes[endpoint, col]
            for branch in range(branch_pf_count):
                fractions = histogram_fractions(
                    endpoint_errors[(endpoint, branch)], bins
                )
                ax.stairs(
                    fractions,
                    bins,
                    color=branch_pf_cmap(branch_pf_norm(branch + 1)),
                    alpha=0.4,
                    linewidth=0.8,
                )
            # Highlight branch 1 without a legend entry for every branch.
            ax.stairs(
                histogram_fractions(endpoint_errors[(endpoint, 0)], bins),
                bins,
                color="black",
                linewidth=1.8,
                label="Branch 1",
                zorder=5,
            )
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel(f"Prediction − ground truth ({unit})")
            ax.set_ylabel("Fraction of displayed scenarios")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.colorbar(
        plt.cm.ScalarMappable(norm=branch_pf_norm, cmap=branch_pf_cmap),
        ax=axes,
        label="Physical branch row (1-based)",
        shrink=0.8,
    )
    branch_pf_path = OUTPUT_DIR / "estimated_impedance_pf_error_by_branch.pdf"
    fig.savefig(branch_pf_path, dpi=300, bbox_inches="tight")
    pd.DataFrame(branch_pf_summary).to_csv(
        OUTPUT_DIR / "estimated_impedance_pf_error_by_branch_summary.csv", index=False
    )
    print(f"Saved branch endpoint histograms to {branch_pf_path}")
    print("Black curve: branch 1. Shared buses produce overlapping curves.")
    plt.close(fig)
    print(f"Completed IEEE-{CASE}, seed {SEED}. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

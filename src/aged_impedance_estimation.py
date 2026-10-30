"""Standalone Section 7 of 1_error_dist.ipynb (projected Adam only).

Run from any directory:
    /path/to/venv/bin/python /path/to/src/aged_impedance_estimation.py \
        --case 57 --seed 0 \
        --ageing case57_aged_seed0_k2p5_s40_d10_a60_g2_normal

Inputs: src/datakit_out/caseN/seedXX/AGEING_CASE_NAME/raw/
        src/grids/caseN_original.m
Outputs: src/results/caseN/seedXX/
TensorBoard: src/tb/caseN/seedXX/impedance-multiplier-denoising-TIMESTAMP/

IEEE-118 retains the transfer checkpoint used in the notebook. IEEE-14/30/57
use their matching Datakit Base checkpoint (model-training seed 0). The CLI
seed selects experimental data, independently of the pretrained model's seed.

GENCO is frozen. Independent R/X multipliers use the notebook's [1, 1.5]
bounds and one full-dataset Adam update per epoch. Aged admittances and
impedances are used only for diagnostics and the true-impedance baseline.
Impedance plots are title-free PDFs at 300 DPI. Signed-error analysis and
Section 8 are excluded.
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
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.globals import (
    B_ON,
    VA_H,
    VM_H,
    VA_OUT,
    VM_OUT,
    YFF_TT_R,
    YFT_TF_I,
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


def experiment_paths(seed, case, ageing):
    """Resolve the current Datakit directory layout for one experiment."""
    root = PROJECT_DIR / "datakit_out" / f"case{case}" / f"seed{seed}"
    datasets = {
        f"case{case}_aged": root / ageing,
    }
    return (
        datasets,
        PROJECT_DIR / "results" / f"case{case}" / f"seed{seed}",
        PROJECT_DIR / "tb" / f"case{case}" / f"seed{seed}",
    )


def read_matpower_branch_rows(case_path):
    """Read numeric rows from a MATPOWER ``mpc.branch`` table."""
    case_text = case_path.read_text()
    branch_match = re.search(
        r"mpc\.branch\s*=\s*\[(.*?)\];",
        case_text,
        re.DOTALL,
    )
    if branch_match is None:
        raise ValueError(f"Missing mpc.branch table in {case_path}")

    branch_rows = []
    for line in branch_match.group(1).splitlines():
        fields = line.partition("%")[0].strip().rstrip(";").split()
        if fields:
            if len(fields) < 13:
                raise ValueError(f"Invalid mpc.branch row in {case_path}: {line}")
            branch_rows.append([float(field) for field in fields])
    return branch_rows


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
        help="Experiment and dataset seed (default: 0)",
    )
    parser.add_argument(
        "--ageing",
        required=True,
        help="Exact aged Datakit case directory name",
    )
    options = parser.parse_args(argv)
    return options


def main(argv=None):
    options = parse_args(argv)
    CASE = options.case
    SEED = options.seed
    AGEING = options.ageing
    DATASET_ROOTS, OUTPUT_DIR, tb_root = experiment_paths(SEED, CASE, AGEING)
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
    print(f"Ageing case: {AGEING}")
    print(f"Datasets: {DATASET_ROOTS}")
    print(f"Results: {OUTPUT_DIR}")
    print(f"TensorBoard: {TB_DIR}")
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "seed": SEED,
                "ageing": AGEING,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    AGED_NAME = f"case{CASE}_aged"
    aged_dataset = datasets[AGED_NAME]
    edge_type = ("bus", "connects", "bus")
    admittance_columns = slice(YFF_TT_R, YFT_TF_I + 1)

    def available_dataset_indices(dataset):
        """Return actual saved scenario IDs, including datasets with solver gaps."""
        prefix = "data_index_"
        return sorted(
            int(path.stem.removeprefix(prefix))
            for path in Path(dataset.processed_dir).glob(f"{prefix}*.pt")
        )

    training_indices = available_dataset_indices(aged_dataset)
    if not training_indices:
        raise RuntimeError("The aged dataset contains no scenarios")

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

    # Average active aged observations for ground-truth diagnostics only:
    # [2 * n_lines, 4] (Yff_r, Yff_i, Yft_r, Yft_i).
    actual_aged_admittance, aged_admittance_std = mean_admittance(
        aged_dataset,
        training_indices,
        "Averaging aged Y (diagnostic only)",
    )

    # Read nominal branch parameters directly from the corresponding MATPOWER case.
    branch_rows = read_matpower_branch_rows(original_case_path)
    if len(branch_rows) != n_lines:
        raise ValueError("MATPOWER branch count differs from the graph")

    initial_resistance = torch.tensor(
        [row[2] for row in branch_rows], dtype=torch.float32
    )
    initial_reactance = torch.tensor(
        [row[3] for row in branch_rows], dtype=torch.float32
    )
    initial_impedance = torch.complex(initial_resistance, initial_reactance)
    if (initial_impedance.abs() == 0).any():
        raise ValueError(f"Zero branch impedance in {original_case_path}")

    # Datakit normalizes Y by baseMVA_orig/baseMVA. Recreate its directed-edge
    # admittance features from MATPOWER's R, X, B, tap, shift, and status fields.
    admittance_scale = float(normalizer.baseMVA_orig / normalizer.baseMVA)
    branch_charging = torch.tensor([row[4] for row in branch_rows], dtype=torch.float32)
    tap_magnitude = torch.tensor(
        [row[8] if row[8] != 0.0 else 1.0 for row in branch_rows],
        dtype=torch.float32,
    )
    tap_shift = torch.deg2rad(
        torch.tensor([row[9] for row in branch_rows], dtype=torch.float32)
    )
    branch_status = torch.tensor(
        [row[10] > 0.0 for row in branch_rows], dtype=torch.float32
    )
    tap = torch.polar(tap_magnitude, tap_shift)
    series_y = branch_status * admittance_scale / initial_impedance
    half_charging_y = torch.complex(
        torch.zeros_like(branch_charging),
        branch_status * admittance_scale * branch_charging / 2.0,
    )
    self_y = series_y + half_charging_y
    self_forward = self_y / tap.abs().square()
    cross_forward = -series_y / tap.conj()
    self_reverse = self_y
    cross_reverse = -series_y / tap
    initial_admittance = torch.cat(
        [
            torch.stack(
                [
                    self_forward.real,
                    self_forward.imag,
                    cross_forward.real,
                    cross_forward.imag,
                ],
                dim=1,
            ),
            torch.stack(
                [
                    self_reverse.real,
                    self_reverse.imag,
                    cross_reverse.real,
                    cross_reverse.imag,
                ],
                dim=1,
            ),
        ]
    )

    forward_shunt = self_forward - series_y
    reverse_shunt = self_reverse - series_y

    # Estimate R/X only for in-service branches without non-unity taps or shifts.
    learnable_line_mask = torch.tensor(
        [
            row[10] > 0 and not ((row[8] != 0.0 and row[8] != 1.0) or row[9] != 0.0)
            for row in branch_rows
        ],
        dtype=torch.bool,
    )
    # duplicate for from bus and to bus
    learnable_directed_mask = torch.cat([learnable_line_mask, learnable_line_mask])

    if not learnable_line_mask.any():
        raise RuntimeError("No eligible transmission lines are available")
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
            "ageing": AGEING,
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

    print(f"Completed IEEE-{CASE}, seed {SEED}. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

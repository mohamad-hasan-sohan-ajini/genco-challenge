"""Impedance estimation with optional aging-distribution KL regularization.

Run from any directory:
    /path/to/venv/bin/python /path/to/src/aged_impedance_estimation_2.py \
        --case 57 --seed 0 \
        --ageing case57_aged_seed0_k2p5_s40_d50_a60_logr --epochs 250

Inputs: src/datakit_out/caseN/seedXX/AGEING_CASE_NAME/raw/
        src/grids/caseN_original.m
Outputs: src/results/caseN/seedXX/AGEING_CASE_NAME/{pf_only,kl_physics_informed}/RUN_ID/
TensorBoard: src/tb/caseN/seedXX/AGEING_TYPE/{pf_only,kl_physics_informed}/

IEEE-14/30/57/118 use their matching Datakit Base checkpoint (model-training
seed 0). The CLI seed selects experimental data, independently of the
pretrained model's seed.

GENCO is frozen. Independent R/X multipliers use the notebook's [1, 1.5]
bounds and one projected Adam update per mini-batch. The default loss is
power-flow MSE only. Pass ``--kl-weight 0.01`` (or another positive value) to
add a differentiable KL divergence from the expected aging distribution.
Aged admittances and impedances are used only for diagnostics and the
true-impedance baseline.
"""

import argparse
import json
import math
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

MODEL_REPO_ID = "gridfm/genco-pf-datakit"
MODEL_REVISION = "main"
PROJECT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 16
DEFAULT_EPOCHS = 20
LR = 3e-4
MULTIPLIER_MIN = 1.0
MULTIPLIER_MAX = 1.5
DEFAULT_KL_WEIGHT = 0.0
KL_NUM_BINS = 32
KL_BANDWIDTH_IN_BINS = 1.0
KL_EPSILON = 1e-8
KL_REFERENCE_SAMPLES = 4096
AGING_WEIBULL_SHAPE = 2.5
AGING_WEIBULL_SCALE_YEARS = 40.0
AGING_MAX_AGE_YEARS = 60.0
AGING_MAX_RESISTANCE_DEGRADATION = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_paths(seed, case, ageing, *, data_root=None, results_dir=None, tb_dir=None):
    """Resolve the current Datakit directory layout for one experiment."""
    root = (Path(data_root) if data_root is not None else PROJECT_DIR / "datakit_out") / f"case{case}" / f"seed{seed}"
    datasets = {
        f"case{case}_aged": root / ageing,
    }
    return (
        datasets,
        (Path(results_dir) if results_dir is not None else PROJECT_DIR / "results") / f"case{case}" / f"seed{seed}" / ageing,
        (Path(tb_dir) if tb_dir is not None else PROJECT_DIR / "tb")
        / f"case{case}"
        / f"seed{seed}"
        / ageing.removeprefix(f"case{case}_aged_seed{seed}_"),
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
    if case not in (14, 30, 57, 118):
        raise ValueError(f"Unsupported IEEE case: {case}")
    return {
        "repo_id": MODEL_REPO_ID,
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


def parse_args(argv=None, *, resistance_only=False):
    description = (
        "Estimate aged line resistance with reactance fixed at nominal values."
        if resistance_only
        else __doc__
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "datakit_out")
    parser.add_argument("--grid-dir", type=Path, default=PROJECT_DIR / "grids")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--tb-dir", type=Path, default=PROJECT_DIR / "tb")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of optimization epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=DEFAULT_KL_WEIGHT,
        help=(
            "Weight of the aging-distribution KL term; 0 disables KL "
            f"regularization (default: {DEFAULT_KL_WEIGHT:g})"
        ),
    )
    options = parser.parse_args(argv)
    if Path(options.ageing).name != options.ageing or options.ageing in {".", ".."}:
        parser.error("--ageing must be a case directory name, not a path")
    if options.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if not math.isfinite(options.kl_weight) or options.kl_weight < 0:
        parser.error("--kl-weight must be a finite non-negative number")
    return options


def initialize_impedance_multipliers(
    initial_resistance,
    initial_reactance,
    *,
    resistance_only=False,
):
    """Create learnable multipliers, freezing X at one when estimating R only."""
    resistance_multiplier = torch.nn.Parameter(torch.ones_like(initial_resistance))
    if resistance_only:
        reactance_multiplier = torch.ones_like(initial_reactance)
        parameters = [resistance_multiplier]
    else:
        reactance_multiplier = torch.nn.Parameter(torch.ones_like(initial_reactance))
        parameters = [resistance_multiplier, reactance_multiplier]
    return resistance_multiplier, reactance_multiplier, parameters


def create_multiplier_optimizer(parameters):
    """Create the Adam optimizer used for per-mini-batch updates."""
    return torch.optim.Adam(parameters, lr=LR)


def soft_histogram(values, centers, bandwidth):
    """Return a differentiable probability histogram over fixed bin centers."""
    distances = (values[:, None] - centers[None, :]) / bandwidth
    assignments = torch.exp(-0.5 * distances.square())
    assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(
        KL_EPSILON
    )
    probabilities = assignments.mean(dim=0)
    return probabilities / probabilities.sum().clamp_min(KL_EPSILON)


def expected_aging_multiplier_samples(*, dtype, device):
    """Deterministic quantiles of the truncated-Weibull/logarithmic R prior."""
    quantiles = (
        torch.arange(KL_REFERENCE_SAMPLES, dtype=dtype, device=device) + 0.5
    ) / KL_REFERENCE_SAMPLES
    truncation_cdf = -math.expm1(
        -((AGING_MAX_AGE_YEARS / AGING_WEIBULL_SCALE_YEARS) ** AGING_WEIBULL_SHAPE)
    )
    probabilities = quantiles * truncation_cdf
    ages = AGING_WEIBULL_SCALE_YEARS * (-torch.log1p(-probabilities)) ** (
        1.0 / AGING_WEIBULL_SHAPE
    )
    degradation = (
        AGING_MAX_RESISTANCE_DEGRADATION
        * torch.log1p(ages / AGING_MAX_AGE_YEARS)
        / math.log(2.0)
    )
    return 1.0 + degradation


def build_aging_kl_reference(*, dtype, device):
    """Build soft target distributions for aged R and unchanged X."""
    bin_width = (MULTIPLIER_MAX - MULTIPLIER_MIN) / KL_NUM_BINS
    centers = torch.linspace(
        MULTIPLIER_MIN + bin_width / 2,
        MULTIPLIER_MAX - bin_width / 2,
        KL_NUM_BINS,
        dtype=dtype,
        device=device,
    )
    bandwidth = bin_width * KL_BANDWIDTH_IN_BINS
    expected_resistance = expected_aging_multiplier_samples(
        dtype=dtype,
        device=device,
    )
    expected_reactance = torch.ones_like(expected_resistance)
    return {
        "centers": centers,
        "bandwidth": bandwidth,
        "resistance": soft_histogram(expected_resistance, centers, bandwidth).detach(),
        "reactance": soft_histogram(expected_reactance, centers, bandwidth).detach(),
    }


def discrete_kl_divergence(estimated, expected):
    """Compute D_KL(estimated || expected) with numerical smoothing."""
    estimated = estimated.clamp_min(KL_EPSILON)
    expected = expected.clamp_min(KL_EPSILON)
    estimated = estimated / estimated.sum()
    expected = expected / expected.sum()
    return torch.sum(estimated * (estimated.log() - expected.log()))


def aging_multiplier_kl_loss(
    resistance_multiplier,
    reactance_multiplier,
    learnable_line_mask,
    reference,
    *,
    resistance_only=False,
):
    """KL prior for Weibull/log-aged R and nominal (unit-multiplier) X."""
    centers = reference["centers"]
    bandwidth = reference["bandwidth"]
    estimated_resistance = soft_histogram(
        resistance_multiplier[learnable_line_mask], centers, bandwidth
    )
    estimated_reactance = soft_histogram(
        reactance_multiplier[learnable_line_mask], centers, bandwidth
    )
    resistance_kl = discrete_kl_divergence(
        estimated_resistance, reference["resistance"]
    )
    reactance_kl = discrete_kl_divergence(estimated_reactance, reference["reactance"])
    total_kl = (
        resistance_kl if resistance_only else (resistance_kl + reactance_kl) / 2.0
    )
    return total_kl, resistance_kl, reactance_kl


def create_unique_run_directory(parent):
    """Create a timestamped directory without ever reusing an existing path."""
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for suffix in range(10_000):
        run_id = f"run-{timestamp}" if suffix == 0 else f"run-{timestamp}-{suffix}"
        output_dir = parent / run_id
        try:
            output_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return output_dir, run_id
    raise RuntimeError(f"Could not allocate a unique result directory under {parent}")


def log_line_multipliers(
    writer,
    step,
    learnable_line_mask,
    resistance_multiplier,
    reactance_multiplier,
):
    """Log exact per-branch R/X multipliers and their distributions."""
    branch_rows = (
        learnable_line_mask.nonzero(as_tuple=False).flatten().add(1).cpu().tolist()
    )
    resistance_values = resistance_multiplier.detach()[learnable_line_mask].cpu()
    reactance_values = reactance_multiplier.detach()[learnable_line_mask].cpu()

    writer.add_histogram(
        "multipliers/resistance_distribution",
        resistance_values,
        step,
    )
    writer.add_histogram(
        "multipliers/reactance_distribution",
        reactance_values,
        step,
    )
    for branch_row, resistance, reactance in zip(
        branch_rows,
        resistance_values.tolist(),
        reactance_values.tolist(),
        strict=True,
    ):
        writer.add_scalar(
            f"multipliers/resistance/branch_{branch_row:04d}",
            resistance,
            step,
        )
        writer.add_scalar(
            f"multipliers/reactance/branch_{branch_row:04d}",
            reactance,
            step,
        )


def main(argv=None, *, resistance_only=False):
    options = parse_args(argv, resistance_only=resistance_only)
    CASE = options.case
    SEED = options.seed
    AGEING = options.ageing
    epochs = options.epochs
    kl_weight = options.kl_weight
    kl_enabled = kl_weight > 0
    DATASET_ROOTS, OUTPUT_DIR, tb_root = experiment_paths(
        SEED, CASE, AGEING, data_root=options.data_root,
        results_dir=options.results_dir, tb_dir=options.tb_dir,
    )
    estimation_target = "resistance" if resistance_only else "impedance"
    objective_name = "kl_physics_informed" if kl_enabled else "pf_only"
    output_parent = OUTPUT_DIR / objective_name
    tb_root = tb_root / objective_name
    if resistance_only:
        output_parent = output_parent / "resistance_only"
        tb_root = tb_root / "resistance_only"
    for name, root in DATASET_ROOTS.items():
        for table in ("bus_data.parquet", "gen_data.parquet", "branch_data.parquet"):
            if not (root / "raw" / table).exists():
                raise FileNotFoundError(f"{name}: missing {root / 'raw' / table}")
    original_case_path = options.grid_dir / f"case{CASE}_original.m"
    if not original_case_path.is_file():
        raise FileNotFoundError(original_case_path)
    OUTPUT_DIR, RUN_ID = create_unique_run_directory(output_parent)
    TB_RUN_NAME = f"{estimation_target}-multiplier-denoising-{RUN_ID}"
    TB_DIR = tb_root / TB_RUN_NAME
    DENOISED_Y_PATH = OUTPUT_DIR / "denoised_admittance.pt"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"IEEE-{CASE}; seed: {SEED}; device: {DEVICE}")
    print(f"Ageing case: {AGEING}")
    print(
        f"KL regularization: {'enabled' if kl_enabled else 'disabled'} "
        f"(weight={kl_weight:g})"
    )
    print(f"Datasets: {DATASET_ROOTS}")
    print(f"Results: {OUTPUT_DIR}")
    print(f"TensorBoard: {TB_DIR}")
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "epochs": epochs,
                "seed": SEED,
                "ageing": AGEING,
                "run_id": RUN_ID,
                "estimation_target": estimation_target,
                "model_artifacts": model_artifact_spec(CASE),
                "dataset_roots": {k: str(v) for k, v in DATASET_ROOTS.items()},
                "optimizer": "Adam",
                "learning_rate": LR,
                "batch_size": BATCH_SIZE,
                "update_rule": "one projected optimizer step per mini-batch",
                "multiplier_bounds": [MULTIPLIER_MIN, MULTIPLIER_MAX],
                "loss": (
                    "power_flow_mse + kl_weight * aging_multiplier_kl"
                    if kl_enabled
                    else "power_flow_mse"
                ),
                "kl_enabled": kl_enabled,
                "kl_weight": kl_weight,
                "kl_num_bins": KL_NUM_BINS,
                "kl_bandwidth_in_bins": KL_BANDWIDTH_IN_BINS,
                "kl_direction": "estimated || expected",
                "aging_prior": {
                    "weibull_shape": AGING_WEIBULL_SHAPE,
                    "weibull_scale_years": AGING_WEIBULL_SCALE_YEARS,
                    "max_age_years": AGING_MAX_AGE_YEARS,
                    "max_resistance_degradation": (AGING_MAX_RESISTANCE_DEGRADATION),
                    "reactance_multiplier": 1.0,
                },
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

    # Estimate ordinary in-service lines only; transformers remain fixed.
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
    (
        resistance_multiplier,
        reactance_multiplier,
        multiplier_parameters,
    ) = initialize_impedance_multipliers(
        initial_resistance,
        initial_reactance,
        resistance_only=resistance_only,
    )
    optimizer = create_multiplier_optimizer(multiplier_parameters)
    kl_reference = (
        build_aging_kl_reference(
            dtype=resistance_multiplier.dtype,
            device=resistance_multiplier.device,
        )
        if kl_enabled
        else None
    )

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
        combined = (
            resistance_mismatch
            if resistance_only
            else torch.cat([resistance_mismatch, reactance_mismatch])
        )
        return resistance_mismatch, reactance_mismatch, combined

    def log_epoch_metrics(writer, step, model_mse, gradient_norm=None):
        resistance_mismatch, reactance_mismatch, combined = impedance_metrics()
        resistance_values = resistance_multiplier.detach()[learnable_line_mask]
        reactance_values = reactance_multiplier.detach()[learnable_line_mask]
        if kl_enabled:
            with torch.no_grad():
                kl_loss, resistance_kl, reactance_kl = aging_multiplier_kl_loss(
                    resistance_multiplier,
                    reactance_multiplier,
                    learnable_line_mask,
                    kl_reference,
                    resistance_only=resistance_only,
                )
            kl_value = kl_loss.item()
            resistance_kl_value = resistance_kl.item()
            reactance_kl_value = reactance_kl.item()
        else:
            kl_value = resistance_kl_value = reactance_kl_value = None
        total_loss = model_mse + kl_weight * (kl_value or 0.0)
        log_line_multipliers(
            writer,
            step,
            learnable_line_mask,
            resistance_multiplier,
            reactance_multiplier,
        )
        mismatch_mae = combined.abs().mean().item()
        mismatch_rmse = combined.square().mean().sqrt().item()
        model_rmse = model_mse**0.5
        writer.add_scalar("epoch/model_rmse", model_rmse, step)
        writer.add_scalar("loss/power_flow_mse", model_mse, step)
        if kl_enabled:
            writer.add_scalar("loss/aging_kl", kl_value, step)
            writer.add_scalar("loss/resistance_kl", resistance_kl_value, step)
            writer.add_scalar("loss/reactance_kl", reactance_kl_value, step)
            writer.add_scalar("loss/weighted_aging_kl", kl_weight * kl_value, step)
        writer.add_scalar("loss/total", total_loss, step)
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
        if gradient_norm is not None:
            writer.add_scalar(
                "epoch/gradient_norm",
                gradient_norm,
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
            kl_value,
            resistance_kl_value,
            reactance_kl_value,
        )

    with SummaryWriter(log_dir=str(TB_DIR)) as writer:
        initial_model_mse = evaluate_model_mse(
            training_indices,
            "Checking initial multipliers",
        )
        log_epoch_metrics(writer, 0, initial_model_mse)

        for epoch in range(epochs):
            epoch_generator = torch.Generator().manual_seed(SEED + epoch)
            order = torch.randperm(
                len(training_indices),
                generator=epoch_generator,
            )
            order = order.tolist()
            batch_gradient_norms = []

            progress = tqdm(
                range(0, len(order), BATCH_SIZE),
                desc=(
                    f"Resistance multiplier epoch {epoch + 1}/{epochs}"
                    if resistance_only
                    else f"Multiplier R/X epoch {epoch + 1}/{epochs}"
                ),
                leave=False,
            )
            for start in progress:
                optimizer.zero_grad(set_to_none=True)
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
                model_loss, _ = masked_pf_model_loss(batch, output)
                if kl_enabled:
                    kl_loss, _, _ = aging_multiplier_kl_loss(
                        resistance_multiplier,
                        reactance_multiplier,
                        learnable_line_mask,
                        kl_reference,
                        resistance_only=resistance_only,
                    )
                    total_loss = model_loss + kl_weight * kl_loss.to(model_loss.device)
                else:
                    kl_loss = None
                    total_loss = model_loss
                total_loss.backward()

                parameter_gradient_norms = []
                for parameter in multiplier_parameters:
                    parameter.grad[~learnable_line_mask] = 0.0
                    if not torch.isfinite(parameter.grad).all():
                        target = "resistance" if resistance_only else "R/X"
                        raise RuntimeError(f"Non-finite {target} multiplier gradient")
                    parameter_gradient_norms.append(
                        parameter.grad[learnable_line_mask].norm().item()
                    )
                batch_gradient_norms.append(
                    float(np.linalg.norm(parameter_gradient_norms))
                )

                # Mini-batch Adam update: step and enforce bounds
                # after every mini-batch rather than once per full dataset.
                optimizer.step()
                with torch.no_grad():
                    for parameter in multiplier_parameters:
                        parameter.clamp_(MULTIPLIER_MIN, MULTIPLIER_MAX)
                        parameter[~learnable_line_mask] = 1.0

                progress_metrics = {
                    "model_rmse": f"{model_loss.item() ** 0.5:.3e}",
                    "total": f"{total_loss.item():.3e}",
                }
                if kl_enabled:
                    progress_metrics["kl"] = f"{kl_loss.item():.3e}"
                progress.set_postfix(**progress_metrics)

            candidate_mse = evaluate_model_mse(
                training_indices,
                f"Checking multiplier epoch {epoch + 1}",
            )
            mean_gradient_norm = float(np.mean(batch_gradient_norms))
            (
                model_rmse,
                mismatch_mae,
                mismatch_rmse,
                resistance_median,
                reactance_median,
                resistance_max,
                reactance_max,
                kl_value,
                resistance_kl_value,
                reactance_kl_value,
            ) = log_epoch_metrics(
                writer,
                epoch + 1,
                candidate_mse,
                mean_gradient_norm,
            )
            if resistance_only:
                message = (
                    f"Epoch {epoch + 1:02d}: model RMSE={model_rmse:.6e}, "
                    f"R mismatch MAE={mismatch_mae:.6e}, "
                    f"R mismatch RMSE={mismatch_rmse:.6e}, "
                )
                if kl_enabled:
                    message += f"KL={kl_value:.6e}, "
                print(
                    message
                    + f"median R multiplier={resistance_median:.4f}, "
                    + f"max R multiplier={resistance_max:.4f}"
                )
            else:
                message = (
                    f"Epoch {epoch + 1:02d}: model RMSE={model_rmse:.6e}, "
                    f"R/X mismatch MAE={mismatch_mae:.6e}, "
                    f"R/X mismatch RMSE={mismatch_rmse:.6e}, "
                )
                if kl_enabled:
                    message += (
                        f"KL={kl_value:.6e} "
                        f"(R={resistance_kl_value:.6e}, X={reactance_kl_value:.6e}), "
                    )
                print(
                    message
                    + f"median multipliers=({resistance_median:.4f}, {reactance_median:.4f}), "
                    + f"max multipliers=({resistance_max:.4f}, {reactance_max:.4f})"
                )

    estimated_resistance, estimated_reactance = estimated_impedance_components()
    final_admittance = admittance_from_estimated_impedance().detach().cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case": CASE,
            "epochs": epochs,
            "seed": SEED,
            "ageing": AGEING,
            "run_id": RUN_ID,
            "estimation_target": estimation_target,
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
            "learning_rate": LR,
            "optimizer": "Adam",
            "batch_size": BATCH_SIZE,
            "update_rule": "one projected optimizer step per mini-batch",
            "loss": (
                "power_flow_mse + kl_weight * aging_multiplier_kl"
                if kl_enabled
                else "power_flow_mse"
            ),
            "kl_enabled": kl_enabled,
            "kl_weight": kl_weight,
            "kl_num_bins": KL_NUM_BINS,
            "kl_bandwidth_in_bins": KL_BANDWIDTH_IN_BINS,
            "kl_direction": "estimated || expected",
            "aging_prior": {
                "weibull_shape": AGING_WEIBULL_SHAPE,
                "weibull_scale_years": AGING_WEIBULL_SCALE_YEARS,
                "max_age_years": AGING_MAX_AGE_YEARS,
                "max_resistance_degradation": AGING_MAX_RESISTANCE_DEGRADATION,
                "reactance_multiplier": 1.0,
            },
            **(
                {
                    "kl_bin_centers": kl_reference["centers"].detach().cpu(),
                    "expected_resistance_multiplier_distribution": (
                        kl_reference["resistance"].detach().cpu()
                    ),
                    "expected_reactance_multiplier_distribution": (
                        kl_reference["reactance"].detach().cpu()
                    ),
                }
                if kl_enabled
                else {}
            ),
            "multiplier_bounds": (MULTIPLIER_MIN, MULTIPLIER_MAX),
            "constraint": (
                "R is the initial value times a bounded ageing multiplier; X is fixed"
                if resistance_only
                else "R and X are initial values times bounded ageing multipliers"
            ),
        },
        DENOISED_Y_PATH,
    )
    print(f"Optimized {int(learnable_line_mask.sum()):,} non-transformer lines")
    if resistance_only:
        print(
            f"Saved bounded resistance multiplier and recovered Y to {DENOISED_Y_PATH}"
        )
    else:
        print(
            f"Saved bounded multiplier-space R/X and recovered Y to {DENOISED_Y_PATH}"
        )
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
    if not resistance_only:
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
    if not resistance_only:
        plot_impedance_parity(
            "reactance",
            "X",
            "reactance_estimated_vs_aged.pdf",
        )

    print(f"Completed IEEE-{CASE}, seed {SEED}. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

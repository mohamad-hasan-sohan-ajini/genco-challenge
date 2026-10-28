"""Evaluate ``(data, model)`` pairs and plot matched-condition error histograms.

Data can be either a local gridfm-datakit dataset root or a Hugging Face
dataset. Models are always loaded from Hugging Face and include their Graphkit
configuration and normalizer.

``errors`` maps each label to a tidy DataFrame containing every evaluated bus
value. Voltage-angle actual values and errors are reported in degrees.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys
from gridfm_graphkit.datasets.globals import (
    PG_H,
    PG_OUT,
    QG_H,
    QG_OUT,
    VA_H,
    VA_OUT,
    VM_H,
    VM_OUT,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.config_version import resolve_config
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_task_transforms,
    load_model,
    load_normalizer,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CACHE = PROJECT_DIR / "data" / "huggingface"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REQUIRED_RAW_TABLES = (
    "bus_data.parquet",
    "gen_data.parquet",
    "branch_data.parquet",
)

print(f"Using device {DEVICE}")


@dataclass(frozen=True)
class HuggingFaceData:
    """A gridfm-datakit dataset hosted in a Hugging Face repository.

    ``subfolder`` may point either to a dataset root or directly to its ``raw``
    directory. Set ``repo_type="model"`` for data packaged with a model, such
    as ``gridfm/genco-pf-transfer/data/case118_ieee/raw``.
    """

    repo_id: str
    subfolder: str | None = None
    revision: str = "main"
    repo_type: str = "dataset"
    scenario_partitions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class HuggingFaceModel:
    """A GENCO checkpoint bundle hosted in a Hugging Face model repository."""

    repo_id: str
    subfolder: str
    revision: str = "main"
    checkpoint_filename: str = "best_model_state_dict.pt"
    normalizer_filename: str = "normalizer_stats.pt"
    config_filename: str | None = None
    normalizer_network: str | None = None


# compound input types
DataSource = str | Path | HuggingFaceData
ModelSource = HuggingFaceModel | Mapping[str, object]
DataModelPair = tuple[DataSource, ModelSource]


def _safe_cache_component(value: str) -> str:
    return value.replace("/", "--").replace("\\", "--")


def _validate_local_dataset_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    # Datakit writes either one parquet file per table or a Hive-partitioned
    # parquet directory with the same name.
    if root.name == "raw" and all(
        (root / name).exists() for name in REQUIRED_RAW_TABLES
    ):
        root = root.parent
    missing = [
        name for name in REQUIRED_RAW_TABLES if not (root / "raw" / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{root} is not a gridfm-datakit dataset root; missing raw tables: {missing}"
        )
    return root


def load_data_source(
    source: DataSource,
    normalizer,
    *,
    stream_partitions: str = "auto",
    cache_dir: str | Path = DEFAULT_DATA_CACHE,
) -> HeteroGridDatasetDisk:
    """Load local Datakit output or download a Datakit dataset from the Hub."""

    if isinstance(source, HuggingFaceData):
        if source.repo_type not in {"dataset", "model"}:
            raise ValueError("HuggingFaceData.repo_type must be 'dataset' or 'model'")
        local_root = (
            Path(cache_dir)
            / source.repo_type
            / _safe_cache_component(source.repo_id)
            / _safe_cache_component(source.revision)
            / _safe_cache_component(source.subfolder or "root")
            / (
                "all-partitions"
                if source.scenario_partitions is None
                else "partitions-"
                + "-".join(str(value) for value in source.scenario_partitions)
            )
        )
        if source.subfolder is None:
            raw_dir = local_root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            allow_patterns = None
            if source.scenario_partitions is not None:
                allow_patterns = [
                    f"{table}/scenario_partition={partition}/**"
                    for table in REQUIRED_RAW_TABLES
                    for partition in source.scenario_partitions
                ]
            snapshot_download(
                repo_id=source.repo_id,
                repo_type=source.repo_type,
                revision=source.revision,
                local_dir=raw_dir,
                allow_patterns=allow_patterns,
            )
            candidate = local_root
        else:
            repository_dir = local_root / "repository"
            repository_dir.mkdir(parents=True, exist_ok=True)
            subfolder = source.subfolder.strip("/")
            snapshot_download(
                repo_id=source.repo_id,
                repo_type=source.repo_type,
                revision=source.revision,
                local_dir=repository_dir,
                allow_patterns=f"{subfolder}/**",
            )
            candidate = repository_dir / subfolder
        root = _validate_local_dataset_root(candidate)
    else:
        candidate = Path(source)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Local dataset does not exist: {candidate}. Wrap a Hugging Face "
                "repository ID in HuggingFaceData(...)."
            )
        root = _validate_local_dataset_root(candidate)

    return HeteroGridDatasetDisk(
        root=str(root),
        data_normalizer=normalizer,
        transform=None,
        stream_partitions=stream_partitions,
    )


def _coerce_model_spec(source: ModelSource) -> HuggingFaceModel:
    if isinstance(source, HuggingFaceModel):
        return source
    if isinstance(source, Mapping):
        return HuggingFaceModel(**source)
    raise TypeError(
        "Each model must be HuggingFaceModel(...) or an equivalent mapping; "
        "local model checkpoints are intentionally unsupported."
    )


def _discover_config_filename(spec: HuggingFaceModel) -> str:
    if spec.config_filename is not None:
        return spec.config_filename
    prefix = spec.subfolder.strip("/") + "/"
    candidates = [
        filename.removeprefix(prefix)
        for filename in list_repo_files(
            spec.repo_id,
            revision=spec.revision,
            repo_type="model",
        )
        if filename.startswith(prefix)
        and filename.endswith((".yaml", ".yml"))
        and "/" not in filename.removeprefix(prefix)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one YAML config in {spec.repo_id}/{spec.subfolder}; "
            f"found {candidates}. Set config_filename explicitly."
        )
    return candidates[0]


def load_huggingface_model(
    source: ModelSource,
):
    """Load a GENCO model, config, and fitted normalizer from Hugging Face."""

    spec = _coerce_model_spec(source)
    config_filename = _discover_config_filename(spec)

    def download(filename: str) -> str:
        return hf_hub_download(
            repo_id=spec.repo_id,
            filename=f"{spec.subfolder.strip('/')}/{filename}",
            repo_type="model",
            revision=spec.revision,
        )

    paths = {
        "checkpoint": download(spec.checkpoint_filename),
        "config": download(config_filename),
        "normalizer": download(spec.normalizer_filename),
    }
    with open(paths["config"], encoding="utf-8") as handle:
        config = resolve_config(
            yaml.safe_load(handle), paths["config"], conversion_mode="auto_inline"
        )
    args = NestedNamespace(**config)

    saved_stats = torch.load(paths["normalizer"], map_location="cpu", weights_only=True)
    network = spec.normalizer_network
    if network is None:
        if len(saved_stats) != 1:
            raise ValueError(
                f"Normalizer contains {list(saved_stats)}; set normalizer_network explicitly."
            )
        network = next(iter(saved_stats))
    if network not in saved_stats:
        raise ValueError(f"Normalizer does not contain network {network!r}")

    normalizer = load_normalizer(args)
    normalizer.fit_from_dict(saved_stats[network])
    model = load_model(args)
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    state = _normalize_loaded_state_dict_keys(state)
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()},
        strict=True,
    )
    model.to(DEVICE).eval()
    return model, normalizer, args


def _evaluate_pair(
    pair: DataModelPair,
    label: str,
    *,
    max_samples: int | None,
    data_cache_dir: str | Path,
    model_bundle=None,
) -> pd.DataFrame:
    if (
        not isinstance(pair, Sequence)
        or isinstance(pair, (str, bytes))
        or len(pair) != 2
    ):
        raise TypeError("Each argument must be one (data, model) pair")
    data_source, model_source = pair
    if model_bundle is None:
        model_bundle = load_huggingface_model(model_source)
    model, normalizer, args = model_bundle
    dataset = load_data_source(
        data_source,
        normalizer,
        stream_partitions=getattr(args.data, "stream_partitions", "auto"),
        cache_dir=data_cache_dir,
    )
    sample_count = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    if sample_count == 0:
        raise ValueError(f"{label} has no samples to evaluate")
    transform = get_task_transforms(args)
    frames = []

    with torch.inference_mode():
        for index in tqdm(range(sample_count), desc=f"Evaluating {label}"):
            graph = transform(dataset[index])
            batch = Batch.from_data_list([graph]).to(DEVICE)
            output = model(batch)
            prediction = output["bus"]
            if prediction.ndim != 2 or prediction.size(1) <= max(
                VM_OUT, VA_OUT, PG_OUT, QG_OUT
            ):
                raise ValueError("Model bus output does not contain Vm, Va, P, and Q")

            normalizer.inverse_output(output, batch)
            normalizer.inverse_transform(batch)

            target = batch["bus"].y
            generator_target = batch["gen"].y
            generator_to_bus = batch["gen", "connected_to", "bus"].edge_index[1]
            scenario_id = int(graph.scenario_id.item())
            bus_count = target.size(0)
            bus_index = np.arange(bus_count, dtype=np.int32)
            actual_vm = target[:, VM_H].detach().cpu().numpy()
            predicted_vm = prediction[:, VM_OUT].detach().cpu().numpy()
            actual_va = torch.rad2deg(target[:, VA_H]).detach().cpu().numpy()
            predicted_va = torch.rad2deg(prediction[:, VA_OUT]).detach().cpu().numpy()
            actual_p = torch.zeros(
                bus_count,
                device=DEVICE,
                dtype=generator_target.dtype,
            )
            actual_p.index_add_(0, generator_to_bus, generator_target[:, 0])
            p_evaluation_mask = torch.zeros(
                bus_count,
                device=DEVICE,
                dtype=generator_target.dtype,
            )
            p_evaluation_mask.index_add_(
                0,
                generator_to_bus,
                batch.mask_dict["gen"][:, PG_H].to(generator_target.dtype),
            )
            p_evaluation_mask = p_evaluation_mask > 0
            q_evaluation_mask = batch.mask_dict["bus"][:, QG_H]
            p_bus_index = bus_index[p_evaluation_mask.detach().cpu().numpy()]
            q_bus_index = bus_index[q_evaluation_mask.detach().cpu().numpy()]
            actual_p = actual_p[p_evaluation_mask].detach().cpu().numpy()
            predicted_p = prediction[p_evaluation_mask, PG_OUT].detach().cpu().numpy()
            actual_q = target[q_evaluation_mask, QG_H].detach().cpu().numpy()
            predicted_q = prediction[q_evaluation_mask, QG_OUT].detach().cpu().numpy()

            for variable, unit, indices, actual, predicted in (
                ("Vm", "p.u.", bus_index, actual_vm, predicted_vm),
                ("Va", "degrees", bus_index, actual_va, predicted_va),
                ("P", "MW", p_bus_index, actual_p, predicted_p),
                ("Q", "MVAr", q_bus_index, actual_q, predicted_q),
            ):
                frames.append(
                    pd.DataFrame(
                        {
                            "pair": label,
                            "dataset_index": index,
                            "scenario_id": scenario_id,
                            "bus_index": indices,
                            "variable": variable,
                            "unit": unit,
                            "actual": actual,
                            "prediction": predicted,
                            "error": predicted - actual,
                        }
                    )
                )

    return pd.concat(frames, ignore_index=True)


def _plot_error_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    output_path: str | Path | None,
):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {
        "Nominal impedance + nominal PF": "tab:orange",
        "Aged impedance + aged PF": "green",
        "Hugging Face nominal impedance + nominal PF": "tab:blue",
    }
    statistics = []
    variables = (("Vm", "p.u."), ("Va", "degrees"), ("P", "MW"), ("Q", "MVAr"))
    for axis, (variable, unit) in zip(axes.flat, variables):
        errors = {}
        for label, table in tables.items():
            values = table.loc[table["variable"] == variable]
            finite = np.isfinite(values["actual"]) & np.isfinite(values["error"])
            errors[label] = values.loc[finite, "error"].to_numpy(dtype=float)
            if errors[label].size == 0:
                raise ValueError(f"No finite {variable} values for {label}")
            statistics.append(
                {
                    "Variable": variable,
                    "Condition": label,
                    "Unit": unit,
                    "Count": errors[label].size,
                    "Mean": errors[label].mean(),
                    "Std": errors[label].std(ddof=0),
                }
            )

        lower = min(np.quantile(values, 0.005) for values in errors.values())
        upper = max(np.quantile(values, 0.995) for values in errors.values())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 101)

        for label, values in errors.items():
            counts, _ = np.histogram(values, bins=bins)
            if counts.sum() == 0:
                raise ValueError(f"No displayed {variable} errors for {label}")
            axis.stairs(
                counts / counts.sum(),
                bins,
                color=colors.get(label),
                label=label,
                linewidth=1.5,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xlabel(f"{variable} prediction error ({unit})")
        axis.set_ylabel("Fraction of observations")
        axis.grid(alpha=0.2)
        axis.legend()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    print(
        "\nMatched-condition signed prediction-error statistics "
        "(all finite errors; population std, ddof=0):"
    )
    print(
        pd.DataFrame(statistics).to_string(
            index=False, float_format=lambda value: f"{value:.8g}"
        )
    )
    return figure


def evaluate_and_plot_errors(
    data_and_models: Sequence[DataModelPair],
    *,
    labels: Sequence[str],
    output_path: str | Path | None = "error_evaluation.png",
    max_samples: int | None = None,
    data_cache_dir: str | Path = DEFAULT_DATA_CACHE,
):
    """Evaluate pairs and plot matched-condition signed-error histograms.

    Every sample and bus is evaluated unless ``max_samples`` is set. Each
    returned DataFrame contains the predictions and signed errors used by the
    plot. Models must be Hugging Face model bundles; data can be local Datakit
    output or :class:`HuggingFaceData`.
    """

    tables = {}
    loaded_models = {}
    for label, pair in zip(labels, data_and_models):
        model_spec = _coerce_model_spec(pair[1])
        if model_spec not in loaded_models:
            loaded_models[model_spec] = load_huggingface_model(model_spec)
        tables[label] = _evaluate_pair(
            pair,
            label,
            max_samples=max_samples,
            data_cache_dir=data_cache_dir,
            model_bundle=loaded_models[model_spec],
        )
    figure = _plot_error_tables(tables, output_path=output_path)
    return figure, tables


if __name__ == "__main__":
    # model_seeds = (0, 1, 42)
    model_seeds = (0,)
    models = {
        case: [
            (
                seed,
                HuggingFaceModel(
                    "gridfm/genco-pf-datakit",
                    f"case{case}_ieee/base/seed{seed}",
                ),
            )
            for seed in model_seeds
        ]
        for case in (57, 118)
    }
    # Two matched 1k evaluation datasets per grid case. They use the same
    # seed and generation settings, while varying only the grid condition.
    datasets = {
        57: [
            (
                "Nominal impedance + nominal PF",
                PROJECT_DIR / "datakit_out" / "case57" / "seed0" / "case57_original",
                1_000,
            ),
            (
                "Aged impedance + aged PF",
                PROJECT_DIR
                / "datakit_out"
                / "case57"
                / "seed0"
                / "case57_aged_seed0_k2p5_scale40",
                1_000,
            ),
            (
                "Hugging Face nominal impedance + nominal PF",
                HuggingFaceData(
                    "gridfm/pf_small_case57_ieee",
                    scenario_partitions=tuple(range(5)),
                ),
                1_000,
            ),
        ],
        118: [
            (
                "Nominal impedance + nominal PF",
                PROJECT_DIR
                / "datakit_out"
                / "case118"
                / "datakit_out_seed0"
                / "case118_original",
                1_000,
            ),
            (
                "Aged impedance + aged PF",
                PROJECT_DIR
                / "datakit_out"
                / "case118"
                / "datakit_out_seed0"
                / "case118_aged_seed0_k2p5_scale40",
                1_000,
            ),
            (
                "Hugging Face nominal impedance + nominal PF",
                HuggingFaceData(
                    "gridfm/pf_small_case118_ieee",
                    scenario_partitions=tuple(range(5)),
                ),
                1_000,
            ),
        ],
    }

    for case in (57, 118):
        for model_seed, model in models[case]:
            data_and_models = [
                (dataset_path, model) for _, dataset_path, _ in datasets[case]
            ]
            labels = [label for label, _, _ in datasets[case]]
            sample_limit = min(count for _, _, count in datasets[case])
            figure, errors = evaluate_and_plot_errors(
                data_and_models,
                labels=labels,
                max_samples=sample_limit,
                output_path=PROJECT_DIR
                / "results"
                / "error_evaluation"
                / f"case{case}_model_seed{model_seed}_matched_condition_"
                "pf_error_histograms.pdf",
            )
            plt.close(figure)

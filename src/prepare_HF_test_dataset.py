"""Export the complete official GENCO case57 test split as a standalone dataset.

From src: python prepare_HF_test_dataset.py
Reload offline: from prepare_HF_test_dataset import load_test_dataset
                dataset = load_test_dataset("data/case57_test")
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
import yaml

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.io.config_version import resolve_config
from gridfm_graphkit.io.param_handler import NestedNamespace, load_normalizer


PROJECT_DIR = Path(__file__).resolve().parent
NETWORK = "case57_ieee"
DATASET_REPO = "gridfm/pf_small_case57_ieee"
MODEL_REPO = "gridfm/genco-pf-datakit"
MODEL_FOLDER = "case57_ieee/base/seed0"
CONFIG_FILENAME = "case57_ieee_base_seed0.yaml"
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "hf_full"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "case57_test"
TABLE_ROWS = {"bus_data.parquet": 57, "gen_data.parquet": 7, "branch_data.parquet": 80}


def complete_source_count(root):
    """Reject partial HF partition downloads using the official total and row counts."""
    raw = Path(root) / "raw"
    count_file = raw / "n_scenarios.txt"
    if not count_file.is_file():
        return None
    count = int(count_file.read_text().strip())
    if count <= 0:
        return None
    for table, rows_per_scenario in TABLE_ROWS.items():
        path = raw / table
        if not path.exists():
            return None
        rows = ds.dataset(path, format="parquet", partitioning="hive").count_rows()
        if rows != count * rows_per_scenario:
            return None
    return count


def download_dataset(data_dir, *, local_files_only=False):
    root = Path(data_dir) / NETWORK
    count = complete_source_count(root)
    if count is None:
        print(f"Downloading the full {DATASET_REPO} dataset to {root / 'raw'}")
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision="main",
            local_dir=root / "raw",
            allow_patterns=[
                pattern
                for table in TABLE_ROWS
                for pattern in (table, f"{table}/**")
            ] + ["n_scenarios.txt", "args.log", "README.md"],
            local_files_only=local_files_only,
            max_workers=4,
        )
        count = complete_source_count(root)
        if count is None:
            raise ValueError("The full HF dataset is missing or incomplete; refusing to split a subset")
    print(f"Full HF dataset: {root} ({count:,} scenarios)")
    return root, count


def validate_config(config):
    """Require the published seed-0 configuration requested for this export."""
    data = config["data"]
    if not (
        config["seed"] == 0
        and data["test_ratio"] == 0.1
        and data["val_ratio"] == 0.1
        and data["split_by_load_scenario_idx"] is True
        and data["networks"] == [NETWORK]
    ):
        raise ValueError("Official config must use case57_ieee, seed=0, 0.1/0.1 ratios, and load-group splitting")


def download_config(*, local_files_only=False):
    paths = {
        "config": hf_hub_download(
            MODEL_REPO, f"{MODEL_FOLDER}/{CONFIG_FILENAME}",
            revision="main", local_files_only=local_files_only,
        ),
        "normalizer": hf_hub_download(
            MODEL_REPO, f"{MODEL_FOLDER}/normalizer_stats.pt",
            revision="main", local_files_only=local_files_only,
        ),
    }
    with open(paths["config"], encoding="utf-8") as stream:
        config = resolve_config(
            yaml.safe_load(stream), paths["config"], conversion_mode="auto_inline"
        )
    validate_config(config)
    return config, paths


def verify_splits(splits, *, expected_count=None, description="Scenario"):
    sets = {}
    for name in ("train", "val", "test"):
        values = torch.as_tensor(splits[name]).flatten().tolist()
        sets[name] = set(values)
        if not values or len(values) != len(sets[name]):
            raise ValueError(f"{description} {name} split is empty or contains duplicates")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[first] & sets[second]
        if overlap:
            raise ValueError(f"{description} leakage between {first} and {second}: {len(overlap)} IDs")
    if expected_count is not None and len(set.union(*sets.values())) != expected_count:
        raise ValueError("Splits do not cover the entire source dataset; additional subsampling is not allowed")
    return sets


def split_with_graphkit(data_dir, config, normalizer_path):
    """Run GraphKit's original shuffle, grouping, RNG seeding, and split mapping."""
    validate_config(config)
    module = LitGridHeteroDataModule(
        NestedNamespace(**config), data_dir=str(data_dir),
        normalizer_stats_path=str(normalizer_path),
    )
    module.setup("fit")
    source = module.datasets[0]
    splits = {
        name: torch.tensor(getattr(module, f"{name}_scenario_ids")[0], dtype=torch.long)
        for name in ("train", "val", "test")
    }
    verify_splits(splits, expected_count=len(source))
    load_splits = {
        name: torch.unique(source.load_scenarios[ids]) for name, ids in splits.items()
    }
    verify_splits(load_splits, description="Load-scenario")
    return source, splits, load_splits


def load_test_dataset(root=DEFAULT_OUTPUT_DIR, *, data_normalizer=None, transform=None):
    """Load only the exported files through the official GraphKit dataset class.

    Stored graphs are raw, unmasked copies. Normalization happens on access;
    the bundled pretrained normalizer is used unless a normalizer is supplied.
    Local indices 0..len(dataset)-1 map to original graph.scenario_id values.
    No HF access or original source data is required.
    """
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"No completed test export at {root}")
    if data_normalizer is None:
        config = yaml.safe_load((root / "config.yaml").read_text())
        data_normalizer = load_normalizer(NestedNamespace(**config))
        stats = torch.load(root / "normalizer_stats.pt", map_location="cpu", weights_only=True)
        data_normalizer.fit_from_dict(stats[NETWORK])
    return HeteroGridDatasetDisk(
        root=str(root), data_normalizer=data_normalizer, transform=transform,
    )


def verify_export(root):
    """Verify every exported ID, raw table, and load-group split without source access."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = torch.load(root / "scenario_splits.pt", map_location="cpu", weights_only=True)
    load_splits = torch.load(root / "load_scenario_splits.pt", map_location="cpu", weights_only=True)
    sets = verify_splits(splits, expected_count=manifest["source_sample_count"])
    verify_splits(load_splits, description="Load-scenario")
    dataset = load_test_dataset(root)
    expected_ids = splits["test"].tolist()
    if len(dataset) != len(expected_ids) or len(dataset) != manifest["test_sample_count"]:
        raise ValueError("Exported graph count differs from the entire official test split")
    if not torch.equal(torch.unique(dataset.load_scenarios), load_splits["test"]):
        raise ValueError("Exported load-scenario IDs differ from the official test split")
    for table in TABLE_ROWS:
        raw = ds.dataset(root / "raw" / table, format="parquet")
        ids = set(pc.unique(raw.to_table(columns=["scenario"])["scenario"]).to_pylist())
        if ids != sets["test"]:
            raise ValueError(f"{table} contains missing or non-test scenario IDs")
    for index in tqdm(range(len(dataset)), desc="Verifying standalone test graphs"):
        graph = dataset[index]
        if int(graph.scenario_id.item()) != expected_ids[index]:
            raise ValueError(f"Original scenario ID changed at exported index {index}")
        graph.validate(raise_on_error=True)
    print(f"Test samples: {len(dataset):,}")
    print("Scenario-ID overlaps: train/test=0, val/test=0, train/val=0")
    print("Load-scenario-ID overlaps: train/test=0, val/test=0, train/val=0")
    return dataset


def export_test_dataset(source, splits, load_splits, config, normalizer_path, output_dir):
    """Copy every selected raw graph and raw table row into an atomic export."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    test_ids = splits["test"].tolist()
    verify_splits(splits, expected_count=len(source))
    verify_splits(load_splits, description="Load-scenario")
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temporary:
        staging = Path(temporary)
        (staging / "raw").mkdir()
        (staging / "processed").mkdir()
        # Copy raw serialized dictionaries byte for byte, before normalization
        # or PF masking. Only storage filenames get local sequential indices.
        for index, scenario_id in enumerate(tqdm(test_ids, desc="Copying test graphs")):
            shutil.copyfile(
                Path(source.processed_dir) / f"data_index_{scenario_id}.pt",
                staging / "processed" / f"data_index_{index}.pt",
            )
        torch.save(source.load_scenarios[splits["test"]], staging / "processed/load_scenarios.pt")
        (staging / "processed/processed_raw_files.done").write_text("done")
        for table in TABLE_ROWS:
            raw = ds.dataset(Path(source.raw_dir) / table, format="parquet", partitioning="hive")
            scanner = raw.scanner(filter=ds.field("scenario").isin(test_ids))
            with pq.ParquetWriter(staging / "raw" / table, scanner.projected_schema) as writer:
                for batch in scanner.to_batches():
                    writer.write_batch(batch)
        torch.save(splits, staging / "scenario_splits.pt")
        torch.save(load_splits, staging / "load_scenario_splits.pt")
        (staging / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copyfile(normalizer_path, staging / "normalizer_stats.pt")
        manifest = {
            "format_version": 1,
            "dataset_repo": DATASET_REPO,
            "model_config": f"{MODEL_REPO}/{MODEL_FOLDER}/{CONFIG_FILENAME}",
            "graphkit_version": version("gridfm-graphkit"),
            "source_sample_count": len(source),
            "test_sample_count": len(test_ids),
            "seed": 0, "test_ratio": 0.1, "val_ratio": 0.1,
            "split_by_load_scenario_idx": True,
            "split_implementation": "LitGridHeteroDataModule.setup",
            "graph_storage": "raw dictionaries; filenames use local indices; graph scenario_id values are unchanged",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_export(staging)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during export: {output_dir}")
        staging.rename(output_dir)
    print(f"Standalone test dataset saved to {output_dir}")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Full source cache parent; dataset stored in its case57_ieee subfolder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true",
                        help="Use only already-cached HF files")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    output_dir = options.output_dir.resolve()
    if output_dir.exists():
        if not (output_dir / "manifest.json").is_file():
            raise FileExistsError(f"Refusing to overwrite unrelated or incomplete output: {output_dir}")
        verify_export(output_dir)
        print(f"Reusing standalone test dataset: {output_dir}")
        return output_dir
    config, paths = download_config(local_files_only=options.local_files_only)
    source_root, count = download_dataset(options.data_dir, local_files_only=options.local_files_only)
    if config["data"]["scenarios"][0] < count:
        raise ValueError("Official config would subsample the source dataset; refusing an incomplete test export")
    source, splits, load_splits = split_with_graphkit(
        source_root.parent, config, paths["normalizer"]
    )
    if len(source) != count:
        raise ValueError("Processed source size differs from the complete official dataset")
    print("Official split sizes: " + ", ".join(f"{name}={len(ids):,}" for name, ids in splits.items()))
    return export_test_dataset(source, splits, load_splits, config, paths["normalizer"], output_dir)


if __name__ == "__main__":
    main()

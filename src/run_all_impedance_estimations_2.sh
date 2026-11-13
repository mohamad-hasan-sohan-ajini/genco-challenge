#!/usr/bin/env bash

# Run impedance estimation for all supported IEEE cases and data seeds.
# KL regularization is disabled by default; set KL_WEIGHT to a positive value.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
ESTIMATION_SCRIPT="${SCRIPT_DIR}/aged_impedance_estimation_2.py"
DATA_ROOT="${SCRIPT_DIR}/datakit_out"
GRID_DIR="${SCRIPT_DIR}/grids"

CASE_NUMBERS=(14 30 57 118)
RANDOM_SEEDS=(0 1 2 3 4)
EPOCHS=60
KL_WEIGHT=0.0
AGED_NETWORK_TAG="k2p5_s40_d50_a60_logr"
DATASET_POLL_INTERVAL_SECONDS=30
DATASET_WAIT_TIMEOUT_SECONDS=21600

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Create the virtual environment first." >&2
  exit 1
fi

if [[ ! -f "${ESTIMATION_SCRIPT}" ]]; then
  echo "Missing impedance-estimation script: ${ESTIMATION_SCRIPT}" >&2
  exit 1
fi

for case_number in "${CASE_NUMBERS[@]}"; do
  original_network="${GRID_DIR}/case${case_number}_original.m"
  if [[ ! -f "${original_network}" ]]; then
    echo "Missing nominal network: ${original_network}" >&2
    exit 1
  fi

  for random_seed in "${RANDOM_SEEDS[@]}"; do
    ageing_case="case${case_number}_aged_seed${random_seed}_${AGED_NETWORK_TAG}"
    dataset_dir="${DATA_ROOT}/case${case_number}/seed${random_seed}/${ageing_case}/raw"
    wait_started_at="${SECONDS}"
    while [[ ! -e "${dataset_dir}/bus_data.parquet" \
          || ! -e "${dataset_dir}/gen_data.parquet" \
          || ! -e "${dataset_dir}/branch_data.parquet" ]]; do
      elapsed_seconds=$((SECONDS - wait_started_at))
      if (( elapsed_seconds >= DATASET_WAIT_TIMEOUT_SECONDS )); then
        echo "Timed out waiting for Datakit to finish: ${dataset_dir}" >&2
        exit 1
      fi
      echo "Waiting for Datakit to finish IEEE-${case_number}, seed ${random_seed} (${elapsed_seconds}s elapsed)..."
      sleep "${DATASET_POLL_INTERVAL_SECONDS}"
    done
  done
done

for case_number in "${CASE_NUMBERS[@]}"; do
  for random_seed in "${RANDOM_SEEDS[@]}"; do
    ageing_case="case${case_number}_aged_seed${random_seed}_${AGED_NETWORK_TAG}"
    echo "Running IEEE-${case_number} impedance estimation, seed ${random_seed}, KL weight ${KL_WEIGHT}..."
    "${PYTHON_BIN}" "${ESTIMATION_SCRIPT}" \
      --case "${case_number}" \
      --seed "${random_seed}" \
      --ageing "${ageing_case}" \
      --epochs "${EPOCHS}" \
      --kl-weight "${KL_WEIGHT}"
  done
done

echo "Finished impedance estimation for all IEEE networks and seeds."

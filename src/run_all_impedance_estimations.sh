#!/usr/bin/env bash

# Run aged-impedance estimation for every IEEE network with a matching GENCO
# checkpoint. This script takes no command-line input.

set -euo pipefail

# All run settings are defined here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
ESTIMATION_SCRIPT="${SCRIPT_DIR}/aged_impedance_estimation.py"
DATA_ROOT="${SCRIPT_DIR}/datakit_out"
GRID_DIR="${SCRIPT_DIR}/grids"

CASE_NUMBERS=(14 30 57 118)
RANDOM_SEED=0
EPOCHS=150
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

# Wait until Datakit has finished every dataset before starting the first
# long-running estimation. A Datakit output directory appears before its final
# bus/gen/branch parquet tables are written.
for case_number in "${CASE_NUMBERS[@]}"; do
  ageing_case="case${case_number}_aged_seed${RANDOM_SEED}_${AGED_NETWORK_TAG}"
  dataset_dir="${DATA_ROOT}/case${case_number}/seed${RANDOM_SEED}/${ageing_case}/raw"
  original_network="${GRID_DIR}/case${case_number}_original.m"

  if [[ ! -f "${original_network}" ]]; then
    echo "Missing nominal network: ${original_network}" >&2
    exit 1
  fi

  wait_started_at="${SECONDS}"
  # Datakit may store each *.parquet table as a directory of parquet parts.
  while [[ ! -e "${dataset_dir}/bus_data.parquet" \
        || ! -e "${dataset_dir}/gen_data.parquet" \
        || ! -e "${dataset_dir}/branch_data.parquet" ]]; do
    elapsed_seconds=$((SECONDS - wait_started_at))
    if (( elapsed_seconds >= DATASET_WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for Datakit to finish: ${dataset_dir}" >&2
      exit 1
    fi
    echo "Waiting for Datakit to finish IEEE-${case_number} (${elapsed_seconds}s elapsed)..."
    sleep "${DATASET_POLL_INTERVAL_SECONDS}"
  done
done

for case_number in "${CASE_NUMBERS[@]}"; do
  ageing_case="case${case_number}_aged_seed${RANDOM_SEED}_${AGED_NETWORK_TAG}"

  echo "Running IEEE-${case_number} impedance estimation for ${ageing_case}..."
  "${PYTHON_BIN}" "${ESTIMATION_SCRIPT}" \
    --case "${case_number}" \
    --seed "${RANDOM_SEED}" \
    --ageing "${ageing_case}" \
    --epochs "${EPOCHS}"
done

echo "Finished impedance estimation for all IEEE networks."

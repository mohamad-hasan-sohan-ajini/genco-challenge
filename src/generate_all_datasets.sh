#!/usr/bin/env bash

# Generate nominal and 50%-maximum-resistance-aging PF datasets for every IEEE
# network with a pretrained GENCO checkpoint: 14, 30, 57, and 118 buses.
# Existing outputs are overwritten because the YAML configurations set
# settings.overwrite=true.

set -euo pipefail

# All run settings are defined here. This script takes no command-line input.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
DATAKIT_BIN="${REPO_ROOT}/venv/bin/gridfm_datakit"
AGED_NETWORK_SCRIPT="${SCRIPT_DIR}/create_aged_network.py"
GRID_DIR="${SCRIPT_DIR}/grids"
CONFIG_DIR="${SCRIPT_DIR}/configs"

CASE_NUMBERS=(14 30 57 118)
WEIBULL_SHAPE=2.5
WEIBULL_SCALE_YEARS=40
MAX_AGE_YEARS=60
MAX_RESISTANCE_DEGRADATION=0.5
DEGRADATION_TAG=d50
RANDOM_SEED=0

if [[ ! -x "${PYTHON_BIN}" || ! -x "${DATAKIT_BIN}" ]]; then
  echo "Missing ${REPO_ROOT}/venv. Create the virtual environment and install GridFM Datakit first." >&2
  exit 1
fi

mkdir -p "${GRID_DIR}"
cd "${SCRIPT_DIR}"

# Phase 1: create every aged network before starting any dataset generation.
echo "Phase 1/2: generating all 50% maximum-resistance aged networks..."
for case_number in "${CASE_NUMBERS[@]}"; do
  echo "Creating IEEE-${case_number} aged network..."
  "${PYTHON_BIN}" "${AGED_NETWORK_SCRIPT}" \
    --case "${case_number}" \
    --weibull-shape "${WEIBULL_SHAPE}" \
    --weibull-scale "${WEIBULL_SCALE_YEARS}" \
    --max-age "${MAX_AGE_YEARS}" \
    --max-degradation "${MAX_RESISTANCE_DEGRADATION}" \
    --seed "${RANDOM_SEED}" \
    --output "${GRID_DIR}/case${case_number}_aged.m"
done

# Phase 2: generate nominal and aged datasets after all networks exist.
echo "Phase 2/2: generating all nominal and aged datasets..."
for case_number in "${CASE_NUMBERS[@]}"; do
  echo "Generating IEEE-${case_number} nominal dataset..."
  "${DATAKIT_BIN}" generate "${CONFIG_DIR}/case${case_number}.yaml"

  echo "Generating IEEE-${case_number} 50% maximum-resistance-aging dataset..."
  "${DATAKIT_BIN}" generate \
    "${CONFIG_DIR}/case${case_number}_aged_${DEGRADATION_TAG}.yaml"
done

echo "Finished generating all nominal and aged datasets."

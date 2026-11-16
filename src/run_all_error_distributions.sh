#!/usr/bin/env bash

# Run all 20 case/seed combinations. Launch from any working directory:
#   bash /path/to/genco-challenge/src/run_all_error_distributions.sh
# Results: src/results/new_error_distribution/caseN/seedS/

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
EVALUATION_SCRIPT="${SCRIPT_DIR}/new_error_distribution.py"

CASE_NUMBERS=(14 30 57 118)
RANDOM_SEEDS=(0 1 2 3 4)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Create the virtual environment first." >&2
  exit 1
fi

if [[ ! -f "${EVALUATION_SCRIPT}" ]]; then
  echo "Missing evaluation script: ${EVALUATION_SCRIPT}" >&2
  exit 1
fi

for case_number in "${CASE_NUMBERS[@]}"; do
  for random_seed in "${RANDOM_SEEDS[@]}"; do
    echo "Running error distributions for IEEE-${case_number}, seed ${random_seed}..."
    "${PYTHON_BIN}" "${EVALUATION_SCRIPT}" \
      --case "${case_number}" \
      --seed "${random_seed}"
  done
done

echo "Finished error distributions for all 20 case/seed combinations."

#!/usr/bin/env bash
# Default: all four cases, seed 0; evaluates every eligible scenario.
# Add --all-seeds for seeds 0-4. No sample limit is supported.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/../venv/bin/python"
"${PYTHON_BIN}" "${SCRIPT_DIR}/new_rx_error_distribution.py" "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/new_rx_impedance_error_reduction.py" "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/new_rx_statistical_tests.py" "$@"

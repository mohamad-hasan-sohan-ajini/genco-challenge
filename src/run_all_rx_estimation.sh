#!/usr/bin/env bash
# Default: all four cases, seed 0, 60 epochs. Supports --all-seeds and --epochs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../venv/bin/python" "${SCRIPT_DIR}/run_rx_estimation.py" "$@"

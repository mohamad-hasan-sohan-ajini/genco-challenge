#!/usr/bin/env bash
# Default: all four cases, seed 0. Add --all-seeds for seeds 0-4.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../venv/bin/python" "${SCRIPT_DIR}/prepare_rx_datasets.py" "$@"

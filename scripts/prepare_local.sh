#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

python -m alan_watts_local.cli --config "$ROOT_DIR/config/local_config.yaml" prepare "$@"

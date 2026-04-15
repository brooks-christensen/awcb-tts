#!/usr/bin/env bash
source ~/xtts-venv/bin/activate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

QUERY_ARGS=()
PRESET_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --preset requires a value"
        exit 1
      fi
      PRESET_ARGS+=(--preset "$2")
      shift 2
      ;;
    *)
      QUERY_ARGS+=("$1")
      shift
      ;;
  esac
done

CONFIG_PATH="${AWCB_CONFIG:-config/local_config.xtts.yaml}"

python -m alan_watts_local.cli \
  --config "$CONFIG_PATH" \
  "${PRESET_ARGS[@]}" \
  speak-text \
  "${QUERY_ARGS[@]}"
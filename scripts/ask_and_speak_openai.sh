#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/ask_and_speak.sh \"Your question\" [--preset PRESET]"
  exit 1
fi

QUERY=""
PRESET_ARGS=()
OTHER_ARGS=()

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
      if [[ -z "$QUERY" ]]; then
        QUERY="$1"
      else
        OTHER_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$QUERY" ]]; then
  echo "ERROR: missing query"
  exit 1
fi

CONFIG_PATH="config/local_config.openai.yaml"

python -m alan_watts_local.cli \
  --config "$CONFIG_PATH" \
  "${PRESET_ARGS[@]}" \
  ask-and-speak \
  --query "$QUERY" \
  "${OTHER_ARGS[@]}"
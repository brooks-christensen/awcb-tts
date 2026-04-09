#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/ask_local.sh \"your question here\" [--preset PRESET] [--json]" >&2
  exit 1
fi

QUERY="$1"
shift || true

python -m alan_watts_local.cli --config "$ROOT_DIR/config/local_config.yaml" ask --query "$QUERY" "$@"

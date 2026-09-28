#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/amd/common.sh"
PYTHON_BIN="$(resolve_physground_python)"
print_python_identity "$PYTHON_BIN"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-artifacts/env/preflight-$STAMP.json}"
mkdir -p "$(dirname "$OUT")"
PYTHONPATH=src "$PYTHON_BIN" -m physground.preflight --output "$OUT" --require-rocm
echo "Preflight written to $OUT"

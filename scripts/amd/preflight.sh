#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-artifacts/env/preflight-$STAMP.json}"
mkdir -p "$(dirname "$OUT")"
PYTHONPATH=src python3.12 -m physground.preflight --output "$OUT" --require-rocm
echo "Preflight written to $OUT"

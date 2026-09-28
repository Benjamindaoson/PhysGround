#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="${RUN_ROOT:-artifacts/stage2}"
REDUCED_RECORDS="${REDUCED_RECORDS:-500000}"
JAX_RECORDS="${JAX_RECORDS:-500000}"
MUJOCO_RECORDS="${MUJOCO_RECORDS:-5000}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
MUJOCO_SHARD_SIZE="${MUJOCO_SHARD_SIZE:-500}"
mkdir -p "$RUN_ROOT"

bash scripts/amd/preflight.sh "$RUN_ROOT/preflight-system.json"

if [[ ! -x .venv-core/bin/python ]]; then bash scripts/amd/bootstrap_core.sh; fi
source .venv-core/bin/activate
physground preflight --output "$RUN_ROOT/preflight-core.json" --require-rocm --require-mujoco
physground generate-data --backend reduced --records "$REDUCED_RECORDS" --shard-size "$SHARD_SIZE" --output-dir "$RUN_ROOT/data/reduced-500k" --resume
physground benchmark --require-pass --output "$RUN_ROOT/reference-benchmark.json"
deactivate

if [[ ! -x .venv-jax/bin/python ]]; then bash scripts/amd/bootstrap_jax_rocm.sh; fi
source .venv-jax/bin/activate
physground preflight --output "$RUN_ROOT/preflight-jax.json" --require-rocm --require-jax-gpu --require-mujoco
physground jax-smoke --batch-size 65536 --output "$RUN_ROOT/jax-smoke.json"
physground generate-data --backend jax-reduced --records "$JAX_RECORDS" --shard-size "$SHARD_SIZE" --output-dir "$RUN_ROOT/data/jax-reduced-500k" --resume
deactivate

source .venv-core/bin/activate
physground generate-data --backend mujoco --records "$MUJOCO_RECORDS" --shard-size "$MUJOCO_SHARD_SIZE" --output-dir "$RUN_ROOT/data/mujoco-pilot" --resume
echo "Stage 2 pilot complete: $RUN_ROOT"

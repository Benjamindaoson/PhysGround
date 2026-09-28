#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -d /root/autodl-tmp ]]; then
  DEFAULT_RUN_ROOT="/root/autodl-tmp/PhysGround/stage2-nvidia"
else
  DEFAULT_RUN_ROOT="artifacts/stage2-nvidia"
fi
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
REDUCED_RECORDS="${REDUCED_RECORDS:-500000}"
JAX_RECORDS="${JAX_RECORDS:-500000}"
MUJOCO_RECORDS="${MUJOCO_RECORDS:-5000}"
SHARD_SIZE="${SHARD_SIZE:-50000}"
MUJOCO_SHARD_SIZE="${MUJOCO_SHARD_SIZE:-500}"

mkdir -p "$RUN_ROOT"

bash scripts/cloud/preflight.sh "$RUN_ROOT/preflight-system.json"

if [[ ! -x .venv-core/bin/python ]]; then
  bash scripts/amd/bootstrap_core.sh
fi
source .venv-core/bin/activate
physground preflight --output "$RUN_ROOT/preflight-core.json" --require-cuda --require-mujoco
physground generate-data   --backend reduced   --records "$REDUCED_RECORDS"   --shard-size "$SHARD_SIZE"   --output-dir "$RUN_ROOT/data/reduced-500k"   --resume
physground benchmark --require-pass --output "$RUN_ROOT/reference-benchmark.json"
deactivate

if [[ ! -x .venv-jax-cuda/bin/python ]]; then
  bash scripts/cloud/bootstrap_nvidia_jax.sh
fi
source .venv-jax-cuda/bin/activate
physground preflight   --output "$RUN_ROOT/preflight-jax.json"   --require-cuda   --require-jax-gpu   --require-mujoco
XLA_PYTHON_CLIENT_PREALLOCATE=false   physground jax-smoke --batch-size 65536 --output "$RUN_ROOT/jax-smoke.json" --require-accelerator
XLA_PYTHON_CLIENT_PREALLOCATE=false   physground generate-data     --backend jax-reduced     --records "$JAX_RECORDS"     --shard-size "$SHARD_SIZE"     --output-dir "$RUN_ROOT/data/jax-reduced-500k"     --resume
deactivate

source .venv-core/bin/activate
physground generate-data   --backend mujoco   --records "$MUJOCO_RECORDS"   --shard-size "$MUJOCO_SHARD_SIZE"   --output-dir "$RUN_ROOT/data/mujoco-pilot"   --resume

echo "NVIDIA Stage-2 pilot complete: $RUN_ROOT"

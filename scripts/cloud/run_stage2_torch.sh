#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -d /mnt/data ]]; then
  DEFAULT_RUN_ROOT="/mnt/data/PhysGround/stage2-torch"
elif [[ -d /root/autodl-tmp ]]; then
  DEFAULT_RUN_ROOT="/root/autodl-tmp/PhysGround/stage2-torch"
else
  DEFAULT_RUN_ROOT="artifacts/stage2-torch"
fi

RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
TORCH_RECORDS="${TORCH_RECORDS:-2000000}"
TORCH_SHARD_SIZE="${TORCH_SHARD_SIZE:-250000}"
MUJOCO_RECORDS="${MUJOCO_RECORDS:-200}"
MUJOCO_SHARD_SIZE="${MUJOCO_SHARD_SIZE:-50}"

mkdir -p "$RUN_ROOT"

bash scripts/cloud/preflight.sh "$RUN_ROOT/preflight-system.json"

if [[ ! -x .venv-torch/bin/python ]]; then
  bash scripts/cloud/bootstrap_torch_accel.sh
fi
source .venv-torch/bin/activate

physground preflight   --output "$RUN_ROOT/preflight-torch.json"   --require-mujoco

physground torch-smoke   --batch-size 1048576   --output "$RUN_ROOT/torch-smoke.json"   --require-accelerator

physground generate-data   --backend torch-reduced   --records "$TORCH_RECORDS"   --shard-size "$TORCH_SHARD_SIZE"   --output-dir "$RUN_ROOT/data/torch-reduced-2m"   --resume

physground benchmark --require-pass --output "$RUN_ROOT/reference-benchmark.json"

physground generate-data   --backend mujoco   --records "$MUJOCO_RECORDS"   --shard-size "$MUJOCO_SHARD_SIZE"   --output-dir "$RUN_ROOT/data/mujoco-pilot"   --resume

echo "Torch-accelerated Stage-2 pilot complete: $RUN_ROOT"

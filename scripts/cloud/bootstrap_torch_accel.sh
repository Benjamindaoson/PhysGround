#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/amd/common.sh"

PYTHON_BIN="$(resolve_physground_python)"
print_python_identity "$PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.version.cuda)
print("accelerator", torch.cuda.is_available())
print("count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("The preinstalled PyTorch cannot see an accelerator")
props = torch.cuda.get_device_properties(0)
print("name", repr(torch.cuda.get_device_name(0)))
print("memory_gb", props.total_memory / 1024**3)
PY

VENV="${PHYSGROUND_TORCH_VENV:-.venv-torch}"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

if [[ -d /mnt/data ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data/PhysGround/pip-cache}"
  mkdir -p "$PIP_CACHE_DIR"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,opt,mujoco]' --no-deps
python -m pip install pytest pytest-cov ruff scipy cma mujoco

python - <<'PY'
import torch
import mujoco
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.version.cuda)
print("accelerator", torch.cuda.is_available())
print("mujoco", mujoco.__version__)
if not torch.cuda.is_available():
    raise SystemExit("Accelerator disappeared inside system-site-packages venv")
PY

echo "Torch accelerator environment ready: $ROOT/$VENV"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/amd/common.sh"

PYTHON_BIN="$(resolve_physground_python)"
print_python_identity "$PYTHON_BIN"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; this bootstrap is for NVIDIA instances." >&2
  exit 2
fi

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
if [[ -z "$DRIVER_MAJOR" || ! "$DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then
  echo "Could not parse NVIDIA driver version: $DRIVER_VERSION" >&2
  exit 2
fi

if (( DRIVER_MAJOR >= 580 )); then
  JAX_EXTRA="cuda13"
else
  JAX_EXTRA="cuda12"
fi

VENV="${PHYSGROUND_NVIDIA_JAX_VENV:-.venv-jax-cuda}"
if [[ -d /root/autodl-tmp ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
  mkdir -p "$PIP_CACHE_DIR"
fi
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -U "jax[${JAX_EXTRA}]"
python -m pip install -U mujoco mujoco-mjx
python -m pip install -e '.[dev,opt,mujoco]' --no-deps
python -m pip install numpy pytest pytest-cov ruff scipy cma

XLA_PYTHON_CLIENT_PREALLOCATE=false python - <<'PY'
import jax
import mujoco
from mujoco import mjx

print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("mujoco", mujoco.__version__)
print("mjx", mjx.__name__)
if jax.default_backend() != "gpu":
    raise SystemExit("JAX installed but GPU backend is not active")
PY

echo "NVIDIA JAX environment ready: $ROOT/$VENV"
echo "JAX extra selected: $JAX_EXTRA (driver $DRIVER_VERSION)"

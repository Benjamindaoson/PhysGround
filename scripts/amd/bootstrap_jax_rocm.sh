#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV="${PHYSGROUND_JAX_VENV:-.venv-jax}"
CONSTRAINTS="$ROOT/configs/amd/constraints-rocm723.txt"
ROCM_VERSION=""
for file in /opt/rocm/.info/version /opt/rocm/.info/version-dev; do
  if [[ -f "$file" ]]; then ROCM_VERSION="$(cat "$file")"; break; fi
done
if [[ -z "$ROCM_VERSION" ]] && command -v hipconfig >/dev/null 2>&1; then
  ROCM_VERSION="$(hipconfig --version 2>/dev/null || true)"
fi
if [[ "$ROCM_VERSION" != 7.2* ]]; then
  echo "Expected ROCm 7.2.x, found: ${ROCM_VERSION:-unknown}" >&2
  exit 2
fi
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c "$CONSTRAINTS" 'jax[rocm7-local]==0.8.2'
python -m pip install -c "$CONSTRAINTS" 'mujoco-mjx==3.13.1'
python -m pip install -e '.[dev,opt,mujoco]' --no-deps
python -m pip install numpy pytest pytest-cov ruff scipy cma
python - <<'PY'
import jax, mujoco
from mujoco import mjx
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
print("mujoco", mujoco.__version__)
print("mjx", mjx.__name__)
if jax.default_backend() == "cpu":
    raise SystemExit("JAX installed but no accelerator backend is active")
PY

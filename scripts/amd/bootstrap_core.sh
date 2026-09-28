#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV="${PHYSGROUND_CORE_VENV:-.venv-core}"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,opt,mujoco]'
python - <<'PY'
import mujoco, numpy
print("core environment ready")
print("numpy", numpy.__version__)
print("mujoco", mujoco.__version__)
PY

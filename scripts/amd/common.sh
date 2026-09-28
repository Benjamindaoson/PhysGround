#!/usr/bin/env bash

resolve_physground_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      printf '%s\n' "$PYTHON_BIN"
      return 0
    fi
    echo "PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2
    return 2
  fi

  local candidate
  for candidate in python python3 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
      then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  echo "PhysGround requires Python >= 3.11, but no supported interpreter was found." >&2
  echo "Checked: python, python3, python3.12" >&2
  return 2
}

print_python_identity() {
  local python_bin="$1"
  echo "Using Python interpreter: $(command -v "$python_bin")"
  "$python_bin" - <<'PY'
import platform
import sys
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
PY
}

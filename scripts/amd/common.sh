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

  local candidate resolved
  for candidate in     python     python3     python3.12     /root/miniconda3/bin/python     /root/miniconda3/bin/python3     /usr/local/bin/python     /usr/bin/python3; do
    if [[ "$candidate" = /* ]]; then
      [[ -x "$candidate" ]] || continue
      resolved="$candidate"
    else
      command -v "$candidate" >/dev/null 2>&1 || continue
      resolved="$(command -v "$candidate")"
    fi
    if "$resolved" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      printf '%s\n' "$resolved"
      return 0
    fi
  done

  echo "PhysGround requires Python >= 3.11, but no supported interpreter was found." >&2
  echo "Checked PATH plus /root/miniconda3/bin/python and common system paths." >&2
  return 2
}

print_python_identity() {
  local python_bin="$1"
  echo "Using Python interpreter: $python_bin"
  "$python_bin" - <<'PY'
import platform
import sys
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
PY
}

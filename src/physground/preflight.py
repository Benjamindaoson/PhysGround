from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence


def _command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[-12000:],
            "stderr": result.stderr.strip()[-4000:],
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def _module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    payload: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "hip": torch.version.hip,
        "cuda_api_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["devices"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            payload["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": props.total_memory / 1024**3,
                }
            )
    return payload


def _jax_status() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        devices = [
            {
                "id": int(getattr(device, "id", index)),
                "platform": str(device.platform),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
            }
            for index, device in enumerate(jax.devices())
        ]
        backend = str(jax.default_backend())
    except Exception as exc:
        return {
            "available": True,
            "version": getattr(jax, "__version__", "unknown"),
            "device_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": getattr(jax, "__version__", "unknown"),
        "backend": backend,
        "devices": devices,
    }


def _rocm_version() -> str | None:
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        if path.is_file():
            value = path.read_text(encoding="utf-8", errors="replace").strip()
            if value:
                return value
    result = _command(["hipconfig", "--version"])
    if result.get("available") and result.get("returncode") == 0:
        return str(result.get("stdout") or "")
    return None


def collect_fingerprint() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "rocm": {
            "version": _rocm_version(),
            "rocminfo": _command(["rocminfo"]),
            "rocm_smi": _command(["rocm-smi", "--showproductname", "--showmeminfo", "vram"]),
        },
        "python_packages": {
            "numpy": _module_version("numpy"),
            "mujoco": _module_version("mujoco"),
            "mujoco_mjx": _module_version("mujoco.mjx"),
        },
        "torch": _torch_status(),
        "jax": _jax_status(),
    }


def evaluate_readiness(
    fingerprint: dict[str, Any],
    *,
    require_rocm: bool = False,
    require_jax_gpu: bool = False,
    require_mujoco: bool = False,
) -> dict[str, Any]:
    rocm_version = fingerprint["rocm"].get("version")
    jax = fingerprint["jax"]
    mujoco = fingerprint["python_packages"]["mujoco"]
    checks = {
        "python_supported": sys.version_info >= (3, 11),
        "python_3_12_target": sys.version_info >= (3, 12),
        "rocm_present": bool(rocm_version),
        "rocm_7_2_series": bool(rocm_version and str(rocm_version).startswith("7.2")),
        "mujoco_importable": bool(mujoco.get("available")),
        "jax_importable": bool(jax.get("available")),
        "jax_accelerator_backend": bool(
            jax.get("available") and jax.get("backend") not in (None, "cpu")
        ),
    }
    required = ["python_supported"]
    if require_rocm:
        required.extend(["rocm_present", "rocm_7_2_series"])
    if require_mujoco:
        required.append("mujoco_importable")
    if require_jax_gpu:
        required.extend(["jax_importable", "jax_accelerator_backend"])
    return {
        "passed": all(checks[name] for name in required),
        "required_checks": required,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect PhysGround runtime readiness.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-rocm", action="store_true")
    parser.add_argument("--require-jax-gpu", action="store_true")
    parser.add_argument("--require-mujoco", action="store_true")
    args = parser.parse_args(argv)

    fingerprint = collect_fingerprint()
    readiness = evaluate_readiness(
        fingerprint,
        require_rocm=args.require_rocm,
        require_jax_gpu=args.require_jax_gpu,
        require_mujoco=args.require_mujoco,
    )
    payload = {"fingerprint": fingerprint, "readiness": readiness}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if readiness["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

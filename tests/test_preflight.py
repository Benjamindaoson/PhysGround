from physground.preflight import collect_fingerprint, evaluate_readiness


def test_preflight_collects_host_and_handles_optional_packages() -> None:
    payload = collect_fingerprint()
    assert payload["schema_version"] == "1.0"
    assert "python" in payload["host"]
    assert "rocm" in payload
    assert "jax" in payload
    assert "torch" in payload


def test_readiness_can_be_evaluated_without_accelerator_requirements() -> None:
    fingerprint = collect_fingerprint()
    result = evaluate_readiness(fingerprint)
    assert "checks" in result
    assert "python_supported" in result["checks"]
    assert "python_3_12_target" in result["checks"]


def test_readiness_can_require_cuda_without_rocm() -> None:
    fingerprint = {
        "rocm": {"version": None},
        "nvidia": {
            "nvidia_smi": {
                "available": True,
                "returncode": 0,
                "stdout": "NVIDIA GeForce RTX 4090 D, 30720, 570.0",
            }
        },
        "python_packages": {
            "mujoco": {"available": False},
        },
        "torch": {
            "cuda_api_available": True,
        },
        "jax": {
            "available": False,
        },
    }
    result = evaluate_readiness(fingerprint, require_cuda=True)
    assert result["checks"]["nvidia_smi_available"]
    assert result["checks"]["torch_accelerator_available"]
    assert result["passed"]

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
    assert "python_3_12_or_newer" in result["checks"]

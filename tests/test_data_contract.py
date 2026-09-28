import json
from pathlib import Path

from physground.benchmark import default_reference_path, load_cases


def test_reference_dataset_is_versioned_and_loadable() -> None:
    path = default_reference_path()
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["benchmark"] == "PhysGround-PushBench-v1"
    assert len(payload["cases"]) == 19
    assert set(payload["priors"]) == {"friction", "cof", "joint", "boundary"}


def test_reference_dataset_reconstructs_expected_regimes() -> None:
    cases = load_cases(Path("data/reference/pushbench_v1.json"))
    assert len(cases) == 19
    assert {case.regime for case in cases} == {
        "P1_FRICTION",
        "P2_COF",
        "P3_JOINT",
        "P4_DECISION_BOUNDARY",
        "P5_MODEL_MISMATCH",
    }
    mismatch = [case for case in cases if case.regime == "P5_MODEL_MISMATCH"]
    assert all(case.world.stiction_distance == 0.025 for case in mismatch)

from physground.benchmark import default_cases, run_benchmark, run_episode
from physground.evaluation import decision_sufficiency_report


def test_default_pushbench_covers_all_target_regimes() -> None:
    regimes = {case.regime for case in default_cases()}
    assert regimes == {
        "P1_FRICTION",
        "P2_COF",
        "P3_JOINT",
        "P4_DECISION_BOUNDARY",
        "P5_MODEL_MISMATCH",
    }


def test_oracle_episode_returns_structured_metrics() -> None:
    result = run_episode(default_cases()[0], "oracle")
    assert result.method == "oracle"
    assert result.probes == 0
    assert result.parameter_error == 0.0


def test_benchmark_smoke_runs_multiple_methods() -> None:
    payload = run_benchmark(default_cases()[:2], methods=["oracle", "nominal", "physground"])
    assert payload["benchmark"] == "PhysGround-PushBench-v1"
    assert set(payload["by_method"]) == {"oracle", "nominal", "physground"}
    assert len(payload["episodes"]) == 6


def test_sufficiency_report_exposes_counterexamples() -> None:
    episodes = [
        {"case_id": "a", "regime": "P1", "method": "m", "parameter_error": 0.9, "decision_regret": 0.0},
        {"case_id": "b", "regime": "P1", "method": "m", "parameter_error": 0.8, "decision_regret": 0.01},
        {"case_id": "c", "regime": "P1", "method": "m", "parameter_error": 0.1, "decision_regret": 0.8},
        {"case_id": "d", "regime": "P1", "method": "m", "parameter_error": 0.2, "decision_regret": 0.7},
    ]
    report = decision_sufficiency_report(episodes, exclude_regimes=set())
    method = report["methods"]["m"]
    assert "a" in method["large_parameter_error_low_regret_cases"]
    assert "c" in method["small_parameter_error_high_regret_cases"]

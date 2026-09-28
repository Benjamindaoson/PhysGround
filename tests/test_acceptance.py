import math

from physground.benchmark import default_cases, evaluate_acceptance, run_benchmark
from physground.contracts import Pose2D, PushAction
from physground.hardware import SO101PushAdapter


def test_reference_benchmark_acceptance_gate_passes() -> None:
    payload = run_benchmark(default_cases())
    result = evaluate_acceptance(payload)
    assert result["passed"], result


def test_so101_adapter_generates_safe_waypoint_sequence() -> None:
    captured = []
    adapter = SO101PushAdapter(captured.append)
    action = PushAction(math.pi, 0.0, 0.0, 0.03, 0.03)
    pose = Pose2D(0.2, 0.1, 0.0)
    waypoints = adapter.waypoints(action, pose)
    assert len(waypoints) == 4
    assert waypoints[0].z > waypoints[1].z
    assert waypoints[-1].z > waypoints[-2].z
    adapter.execute(action, pose)
    assert len(captured) == 4

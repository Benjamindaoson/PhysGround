import math

from physground.contracts import PhysicsParams, Pose2D, PushAction, Transition
from physground.dynamics import PlanarPushModel
from physground.sysid import FiniteDifferenceSysID


def test_finite_difference_sysid_reduces_trajectory_loss() -> None:
    model = PlanarPushModel()
    truth = PhysicsParams(0.8, 0.012, -0.008, 1.25)
    initial = PhysicsParams(0.35, 0.0, 0.0, 0.7)
    pose = Pose2D(0.0, 0.0, 0.0)
    actions = [
        PushAction(math.pi, -0.6, 0.0, 0.025, 0.03),
        PushAction(math.pi, 0.6, 0.0, 0.025, 0.03),
        PushAction(-math.pi / 2, 0.5, math.pi / 2, 0.025, 0.03),
    ]
    transitions = []
    current = pose
    for action in actions:
        after = model.step(current, action, truth)
        transitions.append(Transition(current, action, after))
        current = after
    before = model.trajectory_loss(transitions, initial)
    result = FiniteDifferenceSysID(model, learning_rate=0.05).fit(
        transitions, initial, iterations=60
    )
    assert result.loss < before

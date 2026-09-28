import math

from physground.belief import ParticleBelief, PhysicsParticle
from physground.contracts import Goal2D, PhysicsParams, Pose2D, PushAction
from physground.dynamics import PlanarPushModel
from physground.planning import (
    PushPlanner,
    certify_action_stability,
    compute_decision_disagreement,
    generate_goal_directed_actions,
)


def test_friction_changes_translation_distance() -> None:
    model = PlanarPushModel()
    pose = Pose2D(0.0, 0.0, 0.0)
    action = PushAction(math.pi, 0.0, 0.0, 0.05)
    low = model.step(pose, action, PhysicsParams(0.2, 0.0, 0.0, 1.0))
    high = model.step(pose, action, PhysicsParams(1.0, 0.0, 0.0, 1.0))
    assert low.x > high.x
    assert abs(low.y) < 1e-9


def test_center_of_friction_changes_rotation() -> None:
    model = PlanarPushModel()
    pose = Pose2D(0.0, 0.0, 0.0)
    action = PushAction(math.pi, 0.6, 0.0, 0.04)
    left = model.step(pose, action, PhysicsParams(0.5, 0.0, -0.015, 1.0))
    right = model.step(pose, action, PhysicsParams(0.5, 0.0, 0.015, 1.0))
    assert abs(left.yaw - right.yaw) > 0.01


def test_observation_increases_weight_on_matching_physics() -> None:
    model = PlanarPushModel()
    truth = PhysicsParams(0.9, 0.01, 0.0, 1.2)
    other = PhysicsParams(0.2, -0.01, 0.0, 0.7)
    belief = ParticleBelief([PhysicsParticle(truth, 0.5), PhysicsParticle(other, 0.5)])
    pose = Pose2D(0.0, 0.0, 0.0)
    action = PushAction(math.pi, 0.5, 0.0, 0.04)
    observed = model.step(pose, action, truth)

    belief.update(before=pose, action=action, observed_after=observed, model=model)

    assert belief.particles[0].weight > 0.9
    assert belief.effective_sample_size() < 2.0


def test_oracle_plan_reduces_goal_cost() -> None:
    model = PlanarPushModel()
    planner = PushPlanner(model)
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.08, 0.02, 0.0))
    params = PhysicsParams(0.5, 0.0, 0.0, 1.0)
    actions = generate_goal_directed_actions(pose, goal)
    plan = planner.oracle_plan(pose, goal, params, actions)
    after = model.step(pose, plan.action, params)
    assert after.position_error(goal.pose) < pose.position_error(goal.pose)


def test_decision_disagreement_is_zero_for_duplicate_hypotheses() -> None:
    params = PhysicsParams(0.5, 0.0, 0.0, 1.0)
    belief = ParticleBelief([PhysicsParticle(params, 0.5), PhysicsParticle(params, 0.5)])
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.08, 0.02, math.radians(10)))
    planner = PushPlanner()
    result = compute_decision_disagreement(
        pose=pose,
        goal=goal,
        belief=belief,
        actions=generate_goal_directed_actions(pose, goal),
        planner=planner,
    )
    assert result.modal_mass == 1.0
    assert result.value == 0.0


def test_certificate_passes_for_decision_equivalent_physics() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.1, 0.0, 0.0))
    particles = [
        PhysicsParticle(PhysicsParams(0.3, 0.0, 0.0, 1.0), 0.5),
        PhysicsParticle(PhysicsParams(0.7, 0.0, 0.0, 1.0), 0.5),
    ]
    result = certify_action_stability(
        pose=pose,
        goal=goal,
        belief=ParticleBelief(particles),
        actions=generate_goal_directed_actions(pose, goal),
        planner=PushPlanner(),
        required_modal_mass=0.9,
        minimum_margin=-1e-12,
    )
    assert result.modal_mass == 1.0
    assert result.certified

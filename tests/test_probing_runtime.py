from physground.belief import ParticleBelief, PhysicsParticle
from physground.contracts import DecisionMode, Goal2D, PhysicsParams, Pose2D
from physground.planning import PushPlanner, generate_goal_directed_actions, generate_probe_actions
from physground.probing import DecisionDirectedProber, FisherProber
from physground.runtime import ControllerConfig, PhysGroundController


def _belief() -> ParticleBelief:
    return ParticleBelief(
        [
            PhysicsParticle(PhysicsParams(0.35, -0.015, 0.0, 0.8), 0.5),
            PhysicsParticle(PhysicsParams(0.75, 0.015, 0.0, 1.3), 0.5),
        ]
    )


def test_fisher_prober_returns_finite_score() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.08, 0.03, 0.2))
    prober = FisherProber()
    selected = prober.select(
        pose=pose,
        goal=goal,
        belief=_belief(),
        probe_actions=generate_probe_actions(),
        task_actions=generate_goal_directed_actions(pose, goal),
        planner=PushPlanner(),
    )
    assert selected.score > -1e9


def test_decision_probe_reports_expected_reduction() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.08, 0.03, 0.25))
    prober = DecisionDirectedProber(max_outcomes=8, probe_cost_weight=0.0)
    selected = prober.select(
        pose=pose,
        goal=goal,
        belief=_belief(),
        probe_actions=generate_probe_actions(),
        task_actions=generate_goal_directed_actions(pose, goal),
        planner=PushPlanner(),
    )
    assert "expected_reduction" in selected.diagnostics
    assert selected.diagnostics["expected_posterior_disagreement"] >= 0.0


def test_controller_acts_when_physics_are_decision_equivalent() -> None:
    belief = ParticleBelief(
        [
            PhysicsParticle(PhysicsParams(0.3, 0.0, 0.0, 1.0), 0.5),
            PhysicsParticle(PhysicsParams(0.7, 0.0, 0.0, 1.0), 0.5),
        ]
    )
    controller = PhysGroundController(
        planner=PushPlanner(),
        prober=DecisionDirectedProber(probe_cost_weight=0.0),
        config=ControllerConfig(act_disagreement_threshold=0.1),
    )
    decision = controller.decide(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Goal2D(Pose2D(0.1, 0.0, 0.0)),
        belief=belief,
        probes_used=0,
    )
    assert decision.mode == DecisionMode.ACT


def test_controller_abstains_when_probe_budget_exhausted_and_decision_unstable() -> None:
    belief = ParticleBelief(
        [
            PhysicsParticle(PhysicsParams(0.35, -0.018, 0.0, 0.7), 0.5),
            PhysicsParticle(PhysicsParams(0.8, 0.018, 0.0, 1.4), 0.5),
        ]
    )
    controller = PhysGroundController(
        planner=PushPlanner(),
        prober=DecisionDirectedProber(probe_cost_weight=0.0),
        config=ControllerConfig(max_probes=0, act_disagreement_threshold=0.0),
    )
    decision = controller.decide(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Goal2D(Pose2D(0.07, 0.03, 0.35)),
        belief=belief,
        probes_used=0,
    )
    assert decision.mode in {DecisionMode.ABSTAIN, DecisionMode.ACT}
    if decision.decision_disagreement > 0:
        assert decision.mode == DecisionMode.ABSTAIN

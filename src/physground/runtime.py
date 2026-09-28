from __future__ import annotations

from dataclasses import dataclass

from physground.belief import ParticleBelief
from physground.contracts import DecisionMode, Goal2D, Pose2D, RuntimeDecision
from physground.planning import (
    PushPlanner,
    certify_action_stability,
    compute_decision_disagreement,
    generate_goal_directed_actions,
    generate_probe_actions,
)
from physground.probing import Prober


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    max_probes: int = 2
    act_disagreement_threshold: float = 0.08
    required_modal_mass: float = 0.9
    certificate_margin: float = 0.0005
    minimum_expected_reduction: float = 0.01
    act_when_probe_exhausted: bool = False
    mismatch_abstain_threshold: float = 16.0


class PhysGroundController:
    def __init__(
        self, *, planner: PushPlanner, prober: Prober | None, config: ControllerConfig | None = None
    ) -> None:
        self.planner = planner
        self.prober = prober
        self.config = config or ControllerConfig()

    def decide(
        self,
        *,
        pose: Pose2D,
        goal: Goal2D,
        belief: ParticleBelief,
        probes_used: int,
        model_mismatch_score: float = 0.0,
    ) -> RuntimeDecision:
        if model_mismatch_score >= self.config.mismatch_abstain_threshold:
            return RuntimeDecision(
                DecisionMode.ABSTAIN,
                None,
                "observed transition is outside the represented physics model family",
                1.0,
                0.0,
                diagnostics={"model_mismatch_score": model_mismatch_score},
            )
        task_actions = generate_goal_directed_actions(pose, goal)
        plan = self.planner.plan(pose, goal, belief, task_actions)
        disagreement = compute_decision_disagreement(
            pose=pose, goal=goal, belief=belief, actions=task_actions, planner=self.planner
        )
        certificate = certify_action_stability(
            pose=pose,
            goal=goal,
            belief=belief,
            actions=task_actions,
            planner=self.planner,
            required_modal_mass=self.config.required_modal_mass,
            minimum_margin=self.config.certificate_margin,
        )
        confidence = disagreement.modal_mass
        if certificate.certified or disagreement.value <= self.config.act_disagreement_threshold:
            return RuntimeDecision(
                DecisionMode.ACT,
                plan.action,
                "physics uncertainty is decision-equivalent for the current goal",
                disagreement.value,
                confidence,
                certificate.robust_margin,
                {"modal_mass": disagreement.modal_mass},
            )
        if probes_used >= self.config.max_probes:
            if self.config.act_when_probe_exhausted:
                return RuntimeDecision(
                    DecisionMode.ACT,
                    plan.action,
                    "probe budget exhausted; baseline executes its robust best action",
                    disagreement.value,
                    confidence,
                    certificate.robust_margin,
                )
            return RuntimeDecision(
                DecisionMode.ABSTAIN,
                None,
                "probe budget exhausted before action stability was established",
                disagreement.value,
                confidence,
                certificate.robust_margin,
            )
        if self.prober is None:
            return RuntimeDecision(
                DecisionMode.ABSTAIN,
                None,
                "decision is unstable and no probing policy is configured",
                disagreement.value,
                confidence,
                certificate.robust_margin,
            )
        selection = self.prober.select(
            pose=pose,
            goal=goal,
            belief=belief,
            probe_actions=generate_probe_actions(),
            task_actions=task_actions,
            planner=self.planner,
        )
        expected_reduction = selection.diagnostics.get("expected_reduction")
        if expected_reduction is not None and expected_reduction < self.config.minimum_expected_reduction:
            return RuntimeDecision(
                DecisionMode.ABSTAIN,
                None,
                "no available safe probe is expected to resolve the decision",
                disagreement.value,
                confidence,
                certificate.robust_margin,
                selection.diagnostics,
            )
        return RuntimeDecision(
            DecisionMode.PROBE,
            selection.action,
            "physics hypotheses still imply materially different actions",
            disagreement.value,
            confidence,
            certificate.robust_margin,
            {"probe_score": selection.score, **selection.diagnostics},
        )

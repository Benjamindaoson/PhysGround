from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Iterable

import numpy as np

from physground.belief import ParticleBelief
from physground.contracts import Goal2D, PhysicsParams, Pose2D, PushAction
from physground.dynamics import PlanarPushModel


def generate_goal_directed_actions(
    pose: Pose2D,
    goal: Goal2D,
    *,
    distances: tuple[float, ...] = (0.02, 0.04, 0.06),
    angular_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25),
    contact_offsets: tuple[float, ...] = (-0.5, 0.0, 0.5),
    speed: float = 0.04,
) -> list[PushAction]:
    desired = math.atan2(goal.pose.y - pose.y, goal.pose.x - pose.x)
    actions: list[PushAction] = []
    for angle_delta in angular_offsets:
        direction = desired + angle_delta
        contact_angle = direction + math.pi
        for offset in contact_offsets:
            for distance in distances:
                actions.append(PushAction(contact_angle, offset, direction, distance, speed))
    return actions


def generate_probe_actions(distance: float = 0.018, speed: float = 0.025) -> list[PushAction]:
    actions: list[PushAction] = []
    for direction in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
        contact_angle = direction + math.pi
        for offset in (-0.6, 0.0, 0.6):
            actions.append(PushAction(contact_angle, offset, direction, distance, speed))
    return actions


class RiskMode(StrEnum):
    MEAN = "mean"
    WORST = "worst"
    CVAR = "cvar"


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: PushAction
    expected_cost: float
    worst_cost: float
    cvar_cost: float
    robust_cost: float
    particle_costs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PlanResult:
    action: PushAction
    evaluation: ActionEvaluation
    all_evaluations: tuple[ActionEvaluation, ...]


def goal_cost(pose: Pose2D, goal: Goal2D, yaw_weight: float = 0.05) -> float:
    return pose.position_error(goal.pose) + yaw_weight * pose.yaw_error(goal.pose)


class PushPlanner:
    def __init__(
        self,
        model: PlanarPushModel | None = None,
        *,
        risk_mode: RiskMode = RiskMode.MEAN,
        action_cost_weight: float = 0.08,
        yaw_weight: float = 0.05,
        cvar_alpha: float = 0.25,
    ) -> None:
        self.model = model or PlanarPushModel()
        self.risk_mode = risk_mode
        self.action_cost_weight = action_cost_weight
        self.yaw_weight = yaw_weight
        self.cvar_alpha = cvar_alpha

    def cost_for_params(self, pose: Pose2D, goal: Goal2D, action: PushAction, params: PhysicsParams) -> float:
        after = self.model.step(pose, action, params)
        return goal_cost(after, goal, self.yaw_weight) + self.action_cost_weight * action.distance

    def evaluate(self, pose: Pose2D, goal: Goal2D, action: PushAction, belief: ParticleBelief) -> ActionEvaluation:
        costs = np.asarray([self.cost_for_params(pose, goal, action, p.params) for p in belief.particles], dtype=float)
        weights = belief.weights
        expected = float(np.sum(weights * costs))
        worst = float(np.max(costs))
        count = max(1, int(np.ceil(len(costs) * self.cvar_alpha)))
        cvar = float(np.mean(np.sort(costs)[-count:]))
        robust = expected if self.risk_mode == RiskMode.MEAN else worst if self.risk_mode == RiskMode.WORST else cvar
        return ActionEvaluation(action, expected, worst, cvar, robust, tuple(float(v) for v in costs))

    def plan(self, pose: Pose2D, goal: Goal2D, belief: ParticleBelief, actions: Iterable[PushAction]) -> PlanResult:
        evaluations = tuple(self.evaluate(pose, goal, action, belief) for action in actions)
        if not evaluations:
            raise ValueError("planner requires at least one candidate action")
        best = min(evaluations, key=lambda item: item.robust_cost)
        return PlanResult(best.action, best, evaluations)

    def oracle_plan(self, pose: Pose2D, goal: Goal2D, params: PhysicsParams, actions: Iterable[PushAction]) -> PlanResult:
        belief = ParticleBelief.grid(
            frictions=[params.friction], cof_xs=[params.cof_x], cof_ys=[params.cof_y],
            rotational_drags=[params.rotational_drag]
        )
        return self.plan(pose, goal, belief, actions)


@dataclass(frozen=True, slots=True)
class DecisionDisagreement:
    value: float
    modal_action_index: int
    modal_mass: float
    best_action_indices: tuple[int, ...]
    weighted_entropy: float


def compute_decision_disagreement(
    *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
    actions: Iterable[PushAction], planner: PushPlanner
) -> DecisionDisagreement:
    candidates = list(actions)
    if not candidates:
        raise ValueError("decision disagreement requires candidate actions")
    best_indices: list[int] = []
    masses = np.zeros(len(candidates), dtype=float)
    for particle in belief.particles:
        costs = [planner.cost_for_params(pose, goal, action, particle.params) for action in candidates]
        index = int(np.argmin(costs))
        best_indices.append(index)
        masses[index] += particle.weight
    modal = int(np.argmax(masses))
    modal_mass = float(masses[modal])
    positive = masses[masses > 0]
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0
    max_entropy = float(np.log(max(2, len(candidates))))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    pairwise_disagreement = 1.0 - float(np.sum(np.square(masses)))
    value = 0.7 * pairwise_disagreement + 0.3 * normalized_entropy
    return DecisionDisagreement(float(np.clip(value, 0.0, 1.0)), modal, modal_mass, tuple(best_indices), normalized_entropy)


@dataclass(frozen=True, slots=True)
class ActionStabilityCertificate:
    certified: bool
    action_index: int
    modal_mass: float
    robust_margin: float
    decision_disagreement: float
    reason: str


def certify_action_stability(
    *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
    actions: Iterable[PushAction], planner: PushPlanner,
    required_modal_mass: float = 0.95, minimum_margin: float = 0.001,
) -> ActionStabilityCertificate:
    candidates = list(actions)
    if not candidates:
        raise ValueError("certificate requires candidate actions")
    disagreement = compute_decision_disagreement(
        pose=pose, goal=goal, belief=belief, actions=candidates, planner=planner
    )
    target = disagreement.modal_action_index
    margins: list[float] = []
    for particle in belief.particles:
        costs = np.asarray([planner.cost_for_params(pose, goal, action, particle.params) for action in candidates])
        alternatives = np.delete(costs, target)
        margins.append(float(np.min(alternatives) - costs[target]) if len(alternatives) else np.inf)
    robust_margin = float(min(margins)) if margins else 0.0
    certified = disagreement.modal_mass >= required_modal_mass and robust_margin >= minimum_margin
    if certified:
        reason = "all material physics hypotheses sufficiently agree on the action"
    elif disagreement.modal_mass < required_modal_mass:
        reason = "physics hypotheses still disagree on the best action"
    else:
        reason = "best action is not separated by a robust cost margin"
    return ActionStabilityCertificate(
        certified, target, disagreement.modal_mass, robust_margin, disagreement.value, reason
    )

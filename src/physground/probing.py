from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from physground.belief import ParticleBelief
from physground.contracts import Goal2D, PhysicsParams, Pose2D, PushAction
from physground.dynamics import PlanarPushModel
from physground.planning import PushPlanner, compute_decision_disagreement, generate_goal_directed_actions


@dataclass(frozen=True, slots=True)
class ProbeSelection:
    action: PushAction
    score: float
    diagnostics: dict[str, float]


class Prober(Protocol):
    def select(
        self, *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
        probe_actions: list[PushAction], task_actions: list[PushAction], planner: PushPlanner
    ) -> ProbeSelection: ...


class RandomProber:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(
        self, *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
        probe_actions: list[PushAction], task_actions: list[PushAction], planner: PushPlanner
    ) -> ProbeSelection:
        del pose, goal, belief, task_actions, planner
        if not probe_actions:
            raise ValueError("random prober requires probe actions")
        index = int(self.rng.integers(0, len(probe_actions)))
        return ProbeSelection(probe_actions[index], 0.0, {"random_index": float(index)})


class FisherProber:
    """Parameter-directed probing baseline using a finite-difference Fisher proxy."""

    def __init__(
        self,
        model: PlanarPushModel | None = None,
        *,
        observation_noise: tuple[float, float, float] = (0.006, 0.006, 0.08),
        finite_difference: tuple[float, float, float, float] = (0.02, 0.002, 0.002, 0.04),
        regularization: float = 1e-6,
    ) -> None:
        self.model = model or PlanarPushModel()
        self.noise = np.asarray(observation_noise, dtype=float)
        self.steps = np.asarray(finite_difference, dtype=float)
        self.regularization = regularization

    def _jacobian(self, pose: Pose2D, action: PushAction, params: PhysicsParams) -> np.ndarray:
        center = params.as_array()
        columns: list[np.ndarray] = []
        for index, delta in enumerate(self.steps):
            plus = center.copy()
            minus = center.copy()
            plus[index] += delta
            minus[index] -= delta
            p_plus = self.model.step(pose, action, PhysicsParams.from_array(plus)).as_array()
            p_minus = self.model.step(pose, action, PhysicsParams.from_array(minus)).as_array()
            columns.append((p_plus - p_minus) / (2.0 * delta))
        return np.stack(columns, axis=1)

    def score(self, pose: Pose2D, action: PushAction, belief: ParticleBelief) -> float:
        jacobian = self._jacobian(pose, action, belief.mean())
        whitening = np.diag(1.0 / np.square(self.noise))
        fisher = jacobian.T @ whitening @ jacobian
        covariance = belief.covariance() + self.regularization * np.eye(4)
        sign, logdet = np.linalg.slogdet(
            np.eye(4) + covariance @ fisher + self.regularization * np.eye(4)
        )
        if sign <= 0:
            return float("-inf")
        cost = max(action.distance / max(action.speed, 1e-6), 1e-6)
        return float(logdet / cost)

    def select(
        self, *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
        probe_actions: list[PushAction], task_actions: list[PushAction], planner: PushPlanner
    ) -> ProbeSelection:
        del goal, task_actions, planner
        if not probe_actions:
            raise ValueError("Fisher prober requires probe actions")
        scores = [self.score(pose, action, belief) for action in probe_actions]
        index = int(np.argmax(scores))
        return ProbeSelection(probe_actions[index], float(scores[index]), {"fisher_score": float(scores[index])})


class DecisionDirectedProber:
    """Choose probes for expected reduction in downstream action disagreement."""

    def __init__(
        self,
        model: PlanarPushModel | None = None,
        *,
        position_sigma: float = 0.006,
        yaw_sigma: float = math.radians(5.0),
        probe_cost_weight: float = 0.02,
        max_outcomes: int = 24,
    ) -> None:
        self.model = model or PlanarPushModel()
        self.position_sigma = position_sigma
        self.yaw_sigma = yaw_sigma
        self.probe_cost_weight = probe_cost_weight
        self.max_outcomes = max_outcomes

    def expected_posterior_disagreement(
        self,
        *,
        pose: Pose2D,
        goal: Goal2D,
        belief: ParticleBelief,
        probe: PushAction,
        task_actions: list[PushAction],
        planner: PushPlanner,
    ) -> float:
        del task_actions
        ranked = sorted(belief.particles, key=lambda p: p.weight, reverse=True)[: self.max_outcomes]
        total_mass = sum(p.weight for p in ranked)
        if total_mass <= 0:
            return 1.0
        expected = 0.0
        for truth in ranked:
            observation = self.model.step(pose, probe, truth.params)
            posterior = belief.copy()
            posterior.update(
                before=pose,
                action=probe,
                observed_after=observation,
                model=self.model,
                position_sigma=self.position_sigma,
                yaw_sigma=self.yaw_sigma,
            )
            posterior_actions = generate_goal_directed_actions(observation, goal)
            disagreement = compute_decision_disagreement(
                pose=observation,
                goal=goal,
                belief=posterior,
                actions=posterior_actions,
                planner=planner,
            ).value
            expected += truth.weight / total_mass * disagreement
        return float(expected)

    def select(
        self, *, pose: Pose2D, goal: Goal2D, belief: ParticleBelief,
        probe_actions: list[PushAction], task_actions: list[PushAction], planner: PushPlanner
    ) -> ProbeSelection:
        if not probe_actions:
            raise ValueError("decision-directed prober requires probe actions")
        if not task_actions:
            raise ValueError("decision-directed prober requires task actions")
        current = compute_decision_disagreement(
            pose=pose, goal=goal, belief=belief, actions=task_actions, planner=planner
        ).value
        scores: list[float] = []
        expected_values: list[float] = []
        for probe in probe_actions:
            expected = self.expected_posterior_disagreement(
                pose=pose, goal=goal, belief=belief, probe=probe,
                task_actions=task_actions, planner=planner
            )
            expected_values.append(expected)
            time_cost = probe.distance / max(probe.speed, 1e-6)
            scores.append(float(current - expected - self.probe_cost_weight * time_cost))
        index = int(np.argmax(scores))
        return ProbeSelection(
            probe_actions[index],
            float(scores[index]),
            {
                "current_disagreement": float(current),
                "expected_posterior_disagreement": float(expected_values[index]),
                "expected_reduction": float(current - expected_values[index]),
            },
        )

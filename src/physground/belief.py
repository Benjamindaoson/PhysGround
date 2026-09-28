from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from physground.contracts import PhysicsParams, Pose2D, PushAction
from physground.dynamics import PlanarPushModel


@dataclass(slots=True)
class PhysicsParticle:
    params: PhysicsParams
    weight: float = 1.0


class ParticleBelief:
    def __init__(self, particles: Iterable[PhysicsParticle]) -> None:
        self.particles = [PhysicsParticle(p.params, float(p.weight)) for p in particles]
        if not self.particles:
            raise ValueError("particle belief requires at least one particle")
        self.normalize()

    @classmethod
    def grid(
        cls,
        *,
        frictions: Iterable[float],
        cof_xs: Iterable[float],
        cof_ys: Iterable[float],
        rotational_drags: Iterable[float],
    ) -> "ParticleBelief":
        return cls([
            PhysicsParticle(PhysicsParams(mu, x, y, drag), 1.0)
            for mu in frictions for x in cof_xs for y in cof_ys for drag in rotational_drags
        ])

    @classmethod
    def random_prior(cls, n: int = 64, seed: int = 0) -> "ParticleBelief":
        rng = np.random.default_rng(seed)
        particles = [
            PhysicsParticle(
                PhysicsParams(
                    friction=float(rng.uniform(0.15, 1.0)),
                    cof_x=float(rng.uniform(-0.02, 0.02)),
                    cof_y=float(rng.uniform(-0.02, 0.02)),
                    rotational_drag=float(rng.uniform(0.5, 1.8)),
                ),
                1.0,
            )
            for _ in range(n)
        ]
        return cls(particles)

    def copy(self) -> "ParticleBelief":
        return ParticleBelief(self.particles)

    def normalize(self) -> None:
        total = sum(max(0.0, particle.weight) for particle in self.particles)
        if total <= 0:
            uniform = 1.0 / len(self.particles)
            for particle in self.particles:
                particle.weight = uniform
            return
        for particle in self.particles:
            particle.weight = max(0.0, particle.weight) / total

    @property
    def weights(self) -> np.ndarray:
        return np.asarray([p.weight for p in self.particles], dtype=float)

    def mean(self) -> PhysicsParams:
        values = np.stack([p.params.as_array() for p in self.particles])
        return PhysicsParams.from_array(np.average(values, axis=0, weights=self.weights))

    def covariance(self) -> np.ndarray:
        values = np.stack([p.params.as_array() for p in self.particles])
        mean = np.average(values, axis=0, weights=self.weights)
        centered = values - mean
        return (centered * self.weights[:, None]).T @ centered

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))

    def entropy(self) -> float:
        weights = self.weights
        return float(-np.sum(weights * np.log(np.clip(weights, 1e-12, None))))

    def predictive_residual_score(
        self,
        *,
        before: Pose2D,
        action: PushAction,
        observed_after: Pose2D,
        model: PlanarPushModel,
        position_sigma: float = 0.006,
        yaw_sigma: float = math.radians(5.0),
    ) -> float:
        scores = []
        for particle in self.particles:
            predicted = model.step(before, action, particle.params)
            dp = predicted.position_error(observed_after)
            dyaw = predicted.yaw_error(observed_after)
            scores.append((dp / position_sigma) ** 2 + (dyaw / yaw_sigma) ** 2)
        return float(min(scores)) if scores else float("inf")

    def update(
        self,
        *,
        before: Pose2D,
        action: PushAction,
        observed_after: Pose2D,
        model: PlanarPushModel,
        position_sigma: float = 0.006,
        yaw_sigma: float = math.radians(5.0),
        resample_threshold: float | None = None,
        seed: int = 0,
    ) -> None:
        log_likelihoods = []
        for particle in self.particles:
            predicted = model.step(before, action, particle.params)
            dp = predicted.position_error(observed_after)
            dyaw = predicted.yaw_error(observed_after)
            log_likelihoods.append(-0.5 * (dp / position_sigma) ** 2 - 0.5 * (dyaw / yaw_sigma) ** 2)
        max_log = max(log_likelihoods)
        for particle, log_like in zip(self.particles, log_likelihoods, strict=True):
            particle.weight *= math.exp(log_like - max_log)
        self.normalize()
        if resample_threshold is not None and self.effective_sample_size() < resample_threshold * len(self.particles):
            self.systematic_resample(seed=seed)

    def systematic_resample(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        n = len(self.particles)
        positions = (rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        indexes = np.searchsorted(cumulative, positions, side="left")
        selected = [self.particles[min(int(index), n - 1)].params for index in indexes]
        self.particles = [PhysicsParticle(params, 1.0 / n) for params in selected]

    def top(self, k: int = 5) -> list[PhysicsParticle]:
        return sorted(self.particles, key=lambda p: p.weight, reverse=True)[:k]

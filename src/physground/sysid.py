from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physground.contracts import PhysicsParams, Transition
from physground.dynamics import PlanarPushModel


@dataclass(frozen=True, slots=True)
class SysIDResult:
    params: PhysicsParams
    loss: float
    iterations: int
    history: tuple[float, ...]


class FiniteDifferenceSysID:
    def __init__(
        self,
        model: PlanarPushModel | None = None,
        *,
        steps: tuple[float, float, float, float] = (0.02, 0.002, 0.002, 0.04),
        learning_rate: float = 0.08,
        gradient_clip: float = 2.0,
    ) -> None:
        self.model = model or PlanarPushModel()
        self.steps = np.asarray(steps, dtype=float)
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip

    def _loss(self, transitions: list[Transition], values: np.ndarray) -> float:
        return self.model.trajectory_loss(transitions, PhysicsParams.from_array(values))

    def fit(
        self, transitions: list[Transition], initial: PhysicsParams, *, iterations: int = 80
    ) -> SysIDResult:
        if not transitions:
            raise ValueError("system identification requires transitions")
        values = initial.as_array().copy()
        history: list[float] = []
        best_values = values.copy()
        best_loss = self._loss(transitions, values)
        for _ in range(iterations):
            loss = self._loss(transitions, values)
            history.append(loss)
            gradient = np.zeros(4, dtype=float)
            for index, delta in enumerate(self.steps):
                plus = values.copy(); minus = values.copy()
                plus[index] += delta; minus[index] -= delta
                gradient[index] = (
                    self._loss(transitions, plus) - self._loss(transitions, minus)
                ) / (2.0 * delta)
            norm = float(np.linalg.norm(gradient))
            if norm > self.gradient_clip:
                gradient *= self.gradient_clip / norm
            values -= self.learning_rate * gradient
            values = PhysicsParams.from_array(values).as_array()
            candidate = self._loss(transitions, values)
            if candidate < best_loss:
                best_loss = candidate; best_values = values.copy()
        return SysIDResult(
            PhysicsParams.from_array(best_values), float(best_loss), iterations, tuple(history)
        )


@dataclass(slots=True)
class CMAESSysID:
    """Optional black-box baseline backed by the `cma` package."""

    model: PlanarPushModel = PlanarPushModel()
    sigma0: float = 0.25

    def fit(
        self, transitions: list[Transition], initial: PhysicsParams, *, iterations: int = 40
    ) -> SysIDResult:
        if not transitions:
            raise ValueError("system identification requires transitions")
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError("Install physground[opt] to use CMA-ES SysID") from exc
        scales = [1.0, 0.03, 0.03, 1.0]
        x0 = (initial.as_array() / scales).tolist()

        def objective(normalized: list[float]) -> float:
            values = [value * scale for value, scale in zip(normalized, scales, strict=True)]
            return self.model.trajectory_loss(transitions, PhysicsParams.from_array(values))

        options = {
            "maxiter": iterations,
            "verbose": -9,
            "verb_disp": 0,
            "bounds": [[0.05, -1.34, -1.34, 0.1], [1.5, 1.34, 1.34, 3.0]],
        }
        solution, _ = cma.fmin2(objective, x0, self.sigma0, options=options)
        values = [value * scale for value, scale in zip(solution, scales, strict=True)]
        params = PhysicsParams.from_array(values)
        loss = self.model.trajectory_loss(transitions, params)
        return SysIDResult(params, float(loss), iterations, ())

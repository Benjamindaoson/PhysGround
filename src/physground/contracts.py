from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.yaw], dtype=float)

    def position_error(self, other: "Pose2D") -> float:
        return float(math.hypot(self.x - other.x, self.y - other.y))

    def yaw_error(self, other: "Pose2D") -> float:
        return abs(wrap_angle(self.yaw - other.yaw))


@dataclass(frozen=True, slots=True)
class PhysicsParams:
    friction: float
    cof_x: float
    cof_y: float
    rotational_drag: float

    def __post_init__(self) -> None:
        if not 0.05 <= self.friction <= 1.5:
            raise ValueError("friction must be in [0.05, 1.5]")
        if not -0.04 <= self.cof_x <= 0.04 or not -0.04 <= self.cof_y <= 0.04:
            raise ValueError("center-of-friction offsets must be in [-0.04, 0.04] m")
        if not 0.1 <= self.rotational_drag <= 3.0:
            raise ValueError("rotational_drag must be in [0.1, 3.0]")

    def as_array(self) -> np.ndarray:
        return np.asarray([self.friction, self.cof_x, self.cof_y, self.rotational_drag], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float]) -> "PhysicsParams":
        values = np.asarray(values, dtype=float).reshape(4)
        return cls(
            friction=float(np.clip(values[0], 0.05, 1.5)),
            cof_x=float(np.clip(values[1], -0.04, 0.04)),
            cof_y=float(np.clip(values[2], -0.04, 0.04)),
            rotational_drag=float(np.clip(values[3], 0.1, 3.0)),
        )


@dataclass(frozen=True, slots=True)
class PushAction:
    contact_angle: float
    contact_offset: float
    direction: float
    distance: float
    speed: float = 0.04

    def __post_init__(self) -> None:
        if not -1.0 <= self.contact_offset <= 1.0:
            raise ValueError("contact_offset must be in [-1, 1]")
        if not 0.001 <= self.distance <= 0.20:
            raise ValueError("distance must be in [0.001, 0.20] m")
        if not 0.005 <= self.speed <= 0.5:
            raise ValueError("speed must be in [0.005, 0.5] m/s")

    def vector(self) -> np.ndarray:
        return self.distance * np.asarray([math.cos(self.direction), math.sin(self.direction)])

    def feature_vector(self) -> np.ndarray:
        return np.asarray([
            math.cos(self.contact_angle), math.sin(self.contact_angle), self.contact_offset,
            math.cos(self.direction), math.sin(self.direction), self.distance, self.speed,
        ], dtype=float)


@dataclass(frozen=True, slots=True)
class Goal2D:
    pose: Pose2D
    position_tolerance: float = 0.02
    yaw_tolerance: float = math.radians(12.0)

    def reached(self, pose: Pose2D) -> bool:
        return pose.position_error(self.pose) <= self.position_tolerance and pose.yaw_error(self.pose) <= self.yaw_tolerance


@dataclass(frozen=True, slots=True)
class Transition:
    before: Pose2D
    action: PushAction
    after: Pose2D


class DecisionMode(StrEnum):
    ACT = "ACT"
    PROBE = "PROBE"
    ABSTAIN = "ABSTAIN"


@dataclass(slots=True)
class RuntimeDecision:
    mode: DecisionMode
    action: PushAction | None
    reason: str
    decision_disagreement: float
    confidence: float
    certificate_margin: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

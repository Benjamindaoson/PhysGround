from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

from physground.contracts import Pose2D, PushAction


@dataclass(frozen=True, slots=True)
class CartesianWaypoint:
    x: float
    y: float
    z: float
    speed: float


class PoseProvider(Protocol):
    def observe(self) -> Pose2D: ...


class PushExecutor(Protocol):
    def execute(self, action: PushAction, object_pose: Pose2D) -> None: ...


@dataclass(frozen=True, slots=True)
class SO101Workspace:
    table_z: float = 0.02
    precontact_height: float = 0.06
    push_height: float = 0.025
    retreat_height: float = 0.08
    object_half_extent: float = 0.04
    pusher_clearance: float = 0.01


class SO101PushAdapter:
    """Hardware-neutral adapter; caller supplies the low-level Cartesian controller."""

    def __init__(
        self,
        waypoint_callback: Callable[[CartesianWaypoint], None],
        *,
        workspace: SO101Workspace | None = None,
    ) -> None:
        self.callback = waypoint_callback
        self.workspace = workspace or SO101Workspace()

    def waypoints(self, action: PushAction, object_pose: Pose2D) -> list[CartesianWaypoint]:
        ws = self.workspace
        normal_angle = object_pose.yaw + action.contact_angle
        tangent_angle = normal_angle + math.pi / 2.0
        contact_x = object_pose.x + ws.object_half_extent * (
            math.cos(normal_angle) + action.contact_offset * math.cos(tangent_angle)
        )
        contact_y = object_pose.y + ws.object_half_extent * (
            math.sin(normal_angle) + action.contact_offset * math.sin(tangent_angle)
        )
        start_x = contact_x - ws.pusher_clearance * math.cos(action.direction)
        start_y = contact_y - ws.pusher_clearance * math.sin(action.direction)
        end_x = start_x + action.distance * math.cos(action.direction)
        end_y = start_y + action.distance * math.sin(action.direction)
        return [
            CartesianWaypoint(start_x, start_y, ws.precontact_height, action.speed),
            CartesianWaypoint(start_x, start_y, ws.push_height, action.speed),
            CartesianWaypoint(end_x, end_y, ws.push_height, action.speed),
            CartesianWaypoint(end_x, end_y, ws.retreat_height, action.speed),
        ]

    def execute(self, action: PushAction, object_pose: Pose2D) -> None:
        for waypoint in self.waypoints(action, object_pose):
            self.callback(waypoint)


@dataclass(frozen=True, slots=True)
class PlanarCalibration:
    pixels_to_meters: float
    origin_u: float
    origin_v: float
    yaw_offset: float = 0.0


class MarkerPoseProvider:
    """Convert an external AprilTag/ArUco detector into the planar pose contract."""

    def __init__(
        self,
        detector: Callable[[], tuple[float, float, float]],
        calibration: PlanarCalibration,
    ) -> None:
        self.detector = detector
        self.calibration = calibration

    def observe(self) -> Pose2D:
        u, v, yaw = self.detector()
        c = self.calibration
        return Pose2D(
            x=(u - c.origin_u) * c.pixels_to_meters,
            y=(v - c.origin_v) * c.pixels_to_meters,
            yaw=yaw + c.yaw_offset,
        )

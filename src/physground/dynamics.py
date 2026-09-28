from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from physground.contracts import PhysicsParams, Pose2D, PushAction, Transition, wrap_angle


@dataclass(frozen=True, slots=True)
class PlanarPushModel:
    """Fast reduced quasi-static model for belief rollouts and system ID."""

    half_extent: float = 0.04
    translation_friction_gain: float = 0.55
    torque_gain: float = 0.32
    speed_drag_gain: float = 0.12

    def step(self, pose: Pose2D, action: PushAction, params: PhysicsParams) -> Pose2D:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        rotation = np.asarray([[c, -s], [s, c]], dtype=float)
        normal = np.asarray([math.cos(action.contact_angle), math.sin(action.contact_angle)], dtype=float)
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        contact_local = self.half_extent * (normal + action.contact_offset * tangent)
        cof_local = np.asarray([params.cof_x, params.cof_y], dtype=float)
        lever_world = rotation @ (contact_local - cof_local)
        command = action.vector()
        speed_penalty = 1.0 + self.speed_drag_gain * action.speed / 0.04
        efficiency = 1.0 / (1.0 + self.translation_friction_gain * params.friction * speed_penalty)
        displacement = efficiency * command
        cross = lever_world[0] * command[1] - lever_world[1] * command[0]
        characteristic = self.half_extent * self.half_extent
        delta_yaw = self.torque_gain * cross / (
            max(characteristic, 1e-8) * params.rotational_drag * (1.0 + 0.25 * params.friction)
        )
        cof_world = rotation @ cof_local
        drift = delta_yaw * np.asarray([-cof_world[1], cof_world[0]])
        total = displacement + drift
        return Pose2D(
            x=pose.x + float(total[0]),
            y=pose.y + float(total[1]),
            yaw=wrap_angle(pose.yaw + float(delta_yaw)),
        )

    def rollout(self, pose: Pose2D, actions: Iterable[PushAction], params: PhysicsParams) -> list[Pose2D]:
        states = [pose]
        current = pose
        for action in actions:
            current = self.step(current, action, params)
            states.append(current)
        return states

    def trajectory_loss(
        self, transitions: Iterable[Transition], params: PhysicsParams, *, yaw_weight: float = 0.08
    ) -> float:
        losses = []
        for transition in transitions:
            predicted = self.step(transition.before, transition.action, params)
            position = predicted.position_error(transition.after)
            yaw = predicted.yaw_error(transition.after)
            losses.append(position * position + yaw_weight * yaw * yaw)
        return float(np.mean(losses)) if losses else 0.0


@dataclass(slots=True)
class JaxPlanarPushModel:
    """Optional autodiff wrapper for gradient experiments."""

    model: PlanarPushModel = PlanarPushModel()

    @staticmethod
    def _jax():
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError("Install physground[jax] to use JAX differentiation") from exc
        return jax, jnp

    def next_pose_array(self, pose: Pose2D, action: PushAction, theta: np.ndarray):
        _, jnp = self._jax()
        friction, cof_x, cof_y, rotational_drag = theta
        c, s = jnp.cos(pose.yaw), jnp.sin(pose.yaw)
        rotation = jnp.asarray([[c, -s], [s, c]])
        normal = jnp.asarray([jnp.cos(action.contact_angle), jnp.sin(action.contact_angle)])
        tangent = jnp.asarray([-normal[1], normal[0]])
        contact_local = self.model.half_extent * (normal + action.contact_offset * tangent)
        cof_local = jnp.asarray([cof_x, cof_y])
        lever_world = rotation @ (contact_local - cof_local)
        command = action.distance * jnp.asarray([jnp.cos(action.direction), jnp.sin(action.direction)])
        speed_penalty = 1.0 + self.model.speed_drag_gain * action.speed / 0.04
        efficiency = 1.0 / (1.0 + self.model.translation_friction_gain * friction * speed_penalty)
        displacement = efficiency * command
        cross = lever_world[0] * command[1] - lever_world[1] * command[0]
        characteristic = self.model.half_extent * self.model.half_extent
        delta_yaw = self.model.torque_gain * cross / (
            characteristic * rotational_drag * (1.0 + 0.25 * friction)
        )
        cof_world = rotation @ cof_local
        drift = delta_yaw * jnp.asarray([-cof_world[1], cof_world[0]])
        total = displacement + drift
        return jnp.asarray([pose.x + total[0], pose.y + total[1], pose.yaw + delta_yaw])

    def jacobian_params(self, pose: Pose2D, action: PushAction, params: PhysicsParams) -> np.ndarray:
        jax, _ = self._jax()
        fn = lambda theta: self.next_pose_array(pose, action, theta)
        return np.asarray(jax.jacfwd(fn)(params.as_array()), dtype=float)


@dataclass(slots=True)
class MuJoCoPushVerifier:
    """Optional independent high-fidelity verifier with deliberate model mismatch."""

    half_extent: float = 0.04
    timestep: float = 0.002

    @staticmethod
    def _mujoco():
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install physground[mujoco] to use MuJoCo verification") from exc
        return mujoco

    def step(self, pose: Pose2D, action: PushAction, params: PhysicsParams) -> Pose2D:
        mujoco = self._mujoco()
        xml = f"""
        <mujoco model='physground_push'>
          <option timestep='{self.timestep}' gravity='0 0 -9.81'/>
          <worldbody>
            <geom name='floor' type='plane' size='1 1 .1' friction='{params.friction} 0.01 0.001'/>
            <body name='slider' pos='{pose.x} {pose.y} {self.half_extent}'>
              <freejoint/>
              <inertial pos='{params.cof_x} {params.cof_y} 0' mass='.25' diaginertia='.0008 .0008 .0012'/>
              <geom type='box' size='{self.half_extent} {self.half_extent} {self.half_extent}' friction='{params.friction} 0.01 0.001'/>
            </body>
            <body name='pusher' mocap='true' pos='0 0 {self.half_extent}'>
              <geom type='sphere' size='.008' friction='1 0.01 0.001'/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[3:7] = [math.cos(pose.yaw / 2.0), 0.0, 0.0, math.sin(pose.yaw / 2.0)]
        mujoco.mj_forward(model, data)
        normal_angle = pose.yaw + action.contact_angle
        tangent_angle = normal_angle + math.pi / 2.0
        cx = pose.x + self.half_extent * (
            math.cos(normal_angle) + action.contact_offset * math.cos(tangent_angle)
        )
        cy = pose.y + self.half_extent * (
            math.sin(normal_angle) + action.contact_offset * math.sin(tangent_angle)
        )
        start = (
            cx - 0.01 * math.cos(action.direction),
            cy - 0.01 * math.sin(action.direction),
        )
        data.mocap_pos[0] = [start[0], start[1], self.half_extent]
        mujoco.mj_forward(model, data)
        duration = action.distance / max(action.speed, 1e-6)
        steps = max(1, int(duration / self.timestep))
        for index in range(steps):
            alpha = (index + 1) / steps
            data.mocap_pos[0, 0] = start[0] + alpha * action.distance * math.cos(action.direction)
            data.mocap_pos[0, 1] = start[1] + alpha * action.distance * math.sin(action.direction)
            mujoco.mj_step(model, data)
        q = data.qpos[3:7]
        yaw = math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )
        return Pose2D(float(data.qpos[0]), float(data.qpos[1]), wrap_angle(yaw))

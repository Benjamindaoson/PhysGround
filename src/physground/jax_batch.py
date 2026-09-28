from __future__ import annotations

from typing import Any

import numpy as np


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("JAX is not installed in this environment") from exc
    return jax, jnp


def batched_planar_step(
    poses: np.ndarray,
    actions: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    jax, jnp = _jax()
    poses_j = jnp.asarray(poses)
    actions_j = jnp.asarray(actions)
    physics_j = jnp.asarray(physics)

    def step_one(pose: Any, action: Any, theta: Any) -> Any:
        x, y, yaw = pose
        contact_angle, contact_offset, direction, distance, speed = action
        friction, cof_x, cof_y, rotational_drag = theta
        half_extent = 0.04
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        rotation = jnp.asarray([[c, -s], [s, c]])
        normal = jnp.asarray([jnp.cos(contact_angle), jnp.sin(contact_angle)])
        tangent = jnp.asarray([-normal[1], normal[0]])
        contact_local = half_extent * (normal + contact_offset * tangent)
        cof_local = jnp.asarray([cof_x, cof_y])
        lever_world = rotation @ (contact_local - cof_local)
        command = distance * jnp.asarray([jnp.cos(direction), jnp.sin(direction)])
        speed_penalty = 1.0 + 0.12 * speed / 0.04
        efficiency = 1.0 / (1.0 + 0.55 * friction * speed_penalty)
        displacement = efficiency * command
        cross = lever_world[0] * command[1] - lever_world[1] * command[0]
        delta_yaw = 0.32 * cross / (
            (half_extent * half_extent)
            * rotational_drag
            * (1.0 + 0.25 * friction)
        )
        cof_world = rotation @ cof_local
        drift = delta_yaw * jnp.asarray([-cof_world[1], cof_world[0]])
        total = displacement + drift
        next_yaw = (yaw + delta_yaw + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        return jnp.asarray([x + total[0], y + total[1], next_yaw])

    compiled = jax.jit(jax.vmap(step_one, in_axes=(0, 0, 0)))
    return np.asarray(compiled(poses_j, actions_j, physics_j), dtype=np.float32)


def smoke(batch_size: int = 65536, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    poses = np.column_stack(
        [
            rng.uniform(-0.08, 0.08, batch_size),
            rng.uniform(-0.08, 0.08, batch_size),
            rng.uniform(-np.pi, np.pi, batch_size),
        ]
    ).astype(np.float32)
    actions = np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, batch_size),
            rng.uniform(-1.0, 1.0, batch_size),
            rng.uniform(-np.pi, np.pi, batch_size),
            rng.uniform(0.005, 0.08, batch_size),
            rng.uniform(0.01, 0.12, batch_size),
        ]
    ).astype(np.float32)
    physics = np.column_stack(
        [
            rng.uniform(0.1, 1.2, batch_size),
            rng.uniform(-0.02, 0.02, batch_size),
            rng.uniform(-0.02, 0.02, batch_size),
            rng.uniform(0.5, 1.8, batch_size),
        ]
    ).astype(np.float32)
    output = batched_planar_step(poses, actions, physics)
    jax, _ = _jax()
    return {
        "batch_size": batch_size,
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "backend": str(jax.default_backend()),
        "devices": [str(device) for device in jax.devices()],
    }

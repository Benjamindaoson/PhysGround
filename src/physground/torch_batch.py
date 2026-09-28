from __future__ import annotations

from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment") from exc
    return torch


def batched_planar_step(
    poses: np.ndarray,
    actions: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    """Vectorized reduced PhysGround dynamics on the active PyTorch accelerator."""

    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch accelerator is unavailable")

    device = torch.device("cuda")
    dtype = torch.float32

    pose = torch.as_tensor(poses, device=device, dtype=dtype)
    action = torch.as_tensor(actions, device=device, dtype=dtype)
    theta = torch.as_tensor(physics, device=device, dtype=dtype)

    yaw = pose[:, 2]
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    contact_angle = action[:, 0]
    contact_offset = action[:, 1]
    direction = action[:, 2]
    distance = action[:, 3]
    speed = action[:, 4]

    normal = torch.stack([torch.cos(contact_angle), torch.sin(contact_angle)], dim=1)
    tangent = torch.stack([-normal[:, 1], normal[:, 0]], dim=1)
    contact_local = 0.04 * (normal + contact_offset[:, None] * tangent)
    cof_local = theta[:, 1:3]
    lever_local = contact_local - cof_local

    lever_world = torch.stack(
        [
            c * lever_local[:, 0] - s * lever_local[:, 1],
            s * lever_local[:, 0] + c * lever_local[:, 1],
        ],
        dim=1,
    )

    command = distance[:, None] * torch.stack(
        [torch.cos(direction), torch.sin(direction)],
        dim=1,
    )
    friction = theta[:, 0]
    rotational_drag = theta[:, 3]
    speed_penalty = 1.0 + 0.12 * speed / 0.04
    efficiency = 1.0 / (1.0 + 0.55 * friction * speed_penalty)
    displacement = efficiency[:, None] * command
    cross = lever_world[:, 0] * command[:, 1] - lever_world[:, 1] * command[:, 0]
    delta_yaw = 0.32 * cross / (
        (0.04 * 0.04) * rotational_drag * (1.0 + 0.25 * friction)
    )

    cof_world = torch.stack(
        [
            c * cof_local[:, 0] - s * cof_local[:, 1],
            s * cof_local[:, 0] + c * cof_local[:, 1],
        ],
        dim=1,
    )
    drift = delta_yaw[:, None] * torch.stack(
        [-cof_world[:, 1], cof_world[:, 0]],
        dim=1,
    )
    total = displacement + drift

    after = torch.stack(
        [
            pose[:, 0] + total[:, 0],
            pose[:, 1] + total[:, 1],
            torch.remainder(pose[:, 2] + delta_yaw + torch.pi, 2.0 * torch.pi) - torch.pi,
        ],
        dim=1,
    )

    return after.detach().cpu().numpy().astype(np.float32, copy=False)


def smoke(batch_size: int = 1048576, seed: int = 0) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch accelerator is unavailable")

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
    props = torch.cuda.get_device_properties(0)

    return {
        "batch_size": batch_size,
        "output_shape": list(output.shape),
        "finite": bool(np.isfinite(output).all()),
        "torch": str(torch.__version__),
        "hip": str(torch.version.hip) if torch.version.hip else None,
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "device_name": str(torch.cuda.get_device_name(0)),
        "device_memory_gb": props.total_memory / 1024**3,
    }

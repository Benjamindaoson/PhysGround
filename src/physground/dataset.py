from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from physground.contracts import PhysicsParams, Pose2D, PushAction
from physground.dynamics import MuJoCoPushVerifier


SCHEMA_VERSION = "1.0"
COLUMNS = {
    "before": ["x", "y", "yaw"],
    "action": ["contact_angle", "contact_offset", "direction", "distance", "speed"],
    "physics": ["friction", "cof_x", "cof_y", "rotational_drag"],
    "after": ["x", "y", "yaw"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    before = np.column_stack(
        [
            rng.uniform(-0.08, 0.08, n),
            rng.uniform(-0.08, 0.08, n),
            rng.uniform(-math.pi, math.pi, n),
        ]
    ).astype(np.float32)
    action = np.column_stack(
        [
            rng.uniform(-math.pi, math.pi, n),
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(-math.pi, math.pi, n),
            rng.uniform(0.005, 0.08, n),
            rng.uniform(0.01, 0.12, n),
        ]
    ).astype(np.float32)
    physics = np.column_stack(
        [
            rng.uniform(0.1, 1.2, n),
            rng.uniform(-0.02, 0.02, n),
            rng.uniform(-0.02, 0.02, n),
            rng.uniform(0.5, 1.8, n),
        ]
    ).astype(np.float32)
    return before, action, physics


def _reduced_batch(
    before: np.ndarray,
    action: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    yaw = before[:, 2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    contact_angle = action[:, 0]
    contact_offset = action[:, 1]
    direction = action[:, 2]
    distance = action[:, 3]
    speed = action[:, 4]
    normal = np.column_stack([np.cos(contact_angle), np.sin(contact_angle)])
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    contact_local = 0.04 * (normal + contact_offset[:, None] * tangent)
    cof_local = physics[:, 1:3]
    lever_local = contact_local - cof_local
    lever_world = np.column_stack(
        [
            c * lever_local[:, 0] - s * lever_local[:, 1],
            s * lever_local[:, 0] + c * lever_local[:, 1],
        ]
    )
    command = distance[:, None] * np.column_stack(
        [np.cos(direction), np.sin(direction)]
    )
    friction = physics[:, 0]
    rotational_drag = physics[:, 3]
    speed_penalty = 1.0 + 0.12 * speed / 0.04
    efficiency = 1.0 / (1.0 + 0.55 * friction * speed_penalty)
    displacement = efficiency[:, None] * command
    cross = lever_world[:, 0] * command[:, 1] - lever_world[:, 1] * command[:, 0]
    delta_yaw = 0.32 * cross / (
        (0.04 * 0.04) * rotational_drag * (1.0 + 0.25 * friction)
    )
    cof_world = np.column_stack(
        [
            c * cof_local[:, 0] - s * cof_local[:, 1],
            s * cof_local[:, 0] + c * cof_local[:, 1],
        ]
    )
    drift = delta_yaw[:, None] * np.column_stack(
        [-cof_world[:, 1], cof_world[:, 0]]
    )
    total = displacement + drift
    return np.column_stack(
        [
            before[:, 0] + total[:, 0],
            before[:, 1] + total[:, 1],
            (before[:, 2] + delta_yaw + np.pi) % (2.0 * np.pi) - np.pi,
        ]
    ).astype(np.float32)


def _mujoco_batch(
    before: np.ndarray,
    action: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    verifier = MuJoCoPushVerifier()
    output = np.empty((len(before), 3), dtype=np.float32)
    for index in range(len(before)):
        p = before[index]
        a = action[index]
        after = verifier.step(
            Pose2D(float(p[0]), float(p[1]), float(p[2])),
            PushAction(
                contact_angle=float(a[0]),
                contact_offset=float(a[1]),
                direction=float(a[2]),
                distance=float(a[3]),
                speed=float(a[4]),
            ),
            PhysicsParams.from_array(physics[index]),
        )
        output[index] = after.as_array()
    return output


def _jax_batch(
    before: np.ndarray,
    action: np.ndarray,
    physics: np.ndarray,
) -> np.ndarray:
    from physground.jax_batch import batched_planar_step

    return batched_planar_step(before, action, physics)


BACKENDS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "reduced": _reduced_batch,
    "mujoco": _mujoco_batch,
    "jax-reduced": _jax_batch,
}


def generate_transition_dataset(
    *,
    output_dir: Path,
    records: int,
    shard_size: int = 50000,
    seed: int = 42,
    backend: str = "reduced",
    resume: bool = True,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if records < 1 or shard_size < 1:
        raise ValueError("records and shard_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    remaining = records
    shard_index = 0
    generated = 0
    while remaining > 0:
        count = min(shard_size, remaining)
        path = output_dir / f"part-{shard_index:05d}.npz"
        if resume and path.is_file():
            with np.load(path) as loaded:
                existing = int(loaded["before"].shape[0])
            if existing != count:
                raise ValueError(
                    f"existing shard {path} has {existing} rows, expected {count}"
                )
        else:
            before, action, physics = sample_inputs(count, seed + shard_index)
            after = BACKENDS[backend](before, action, physics)
            if not np.isfinite(after).all():
                raise ValueError(f"non-finite transition detected in shard {shard_index}")
            np.savez_compressed(
                path,
                before=before,
                action=action,
                physics=physics,
                after=after,
            )
        shards.append(
            {
                "path": path.name,
                "records": count,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        generated += count
        remaining -= count
        shard_index += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "PhysGround-Transitions",
        "backend": backend,
        "records": generated,
        "shard_size": shard_size,
        "seed": seed,
        "columns": COLUMNS,
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

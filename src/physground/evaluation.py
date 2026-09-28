from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from physground.contracts import PhysicsParams, Pose2D


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    case_id: str
    regime: str
    method: str
    success: bool
    abstained: bool
    final_pose: Pose2D
    goal_pose: Pose2D
    probes: int
    actions: int
    decision_regret: float
    parameter_error: float
    false_confident_failure: bool
    high_confidence_acts: int
    trace: tuple[dict, ...]


def normalized_parameter_error(estimate: PhysicsParams, truth: PhysicsParams) -> float:
    scales = np.asarray([1.0, 0.02, 0.02, 1.0], dtype=float)
    delta = (estimate.as_array() - truth.as_array()) / scales
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


def aggregate_results(results: Iterable[EpisodeResult]) -> dict[str, float | int]:
    rows = list(results)
    if not rows:
        return {
            "episodes": 0, "success_rate": 0.0, "abstention_rate": 0.0,
            "mean_position_error_m": 0.0, "mean_yaw_error_rad": 0.0,
            "mean_probes": 0.0, "mean_decision_regret": 0.0,
            "mean_parameter_error": 0.0, "false_confidence_rate": 0.0,
        }
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row.success for row in rows])),
        "abstention_rate": float(np.mean([row.abstained for row in rows])),
        "mean_position_error_m": float(np.mean([row.final_pose.position_error(row.goal_pose) for row in rows])),
        "mean_yaw_error_rad": float(np.mean([row.final_pose.yaw_error(row.goal_pose) for row in rows])),
        "mean_probes": float(np.mean([row.probes for row in rows])),
        "mean_decision_regret": float(np.mean([row.decision_regret for row in rows])),
        "mean_parameter_error": float(np.mean([row.parameter_error for row in rows])),
        "false_confidence_rate": float(np.mean([row.false_confident_failure for row in rows])),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def decision_sufficiency_report(
    episodes: Iterable[dict[str, Any]], *, exclude_regimes: set[str] | None = None
) -> dict[str, Any]:
    excluded = exclude_regimes or {"P5_MODEL_MISMATCH"}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if episode.get("regime") not in excluded:
            grouped[str(episode["method"])].append(episode)
    methods: dict[str, Any] = {}
    for method, rows in grouped.items():
        parameter = np.asarray([float(row["parameter_error"]) for row in rows])
        regret = np.asarray([float(row["decision_regret"]) for row in rows])
        if not rows:
            continue
        p75, p25 = float(np.quantile(parameter, 0.75)), float(np.quantile(parameter, 0.25))
        r75, r25 = float(np.quantile(regret, 0.75)), float(np.quantile(regret, 0.25))
        methods[method] = {
            "episodes": len(rows),
            "pearson_parameter_error_vs_regret": _pearson(parameter, regret),
            "parameter_error_mean": float(np.mean(parameter)),
            "decision_regret_mean": float(np.mean(regret)),
            "large_parameter_error_low_regret_cases": [
                row["case_id"] for row in rows
                if float(row["parameter_error"]) >= p75 and float(row["decision_regret"]) <= r25
            ],
            "small_parameter_error_high_regret_cases": [
                row["case_id"] for row in rows
                if float(row["parameter_error"]) <= p25 and float(row["decision_regret"]) >= r75
            ],
        }
    return {
        "claim_test": "parameter reconstruction accuracy vs downstream decision sufficiency",
        "excluded_regimes": sorted(excluded),
        "methods": methods,
    }

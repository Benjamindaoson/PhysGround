from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from physground.belief import ParticleBelief, PhysicsParticle
from physground.contracts import DecisionMode, Goal2D, PhysicsParams, Pose2D, PushAction, wrap_angle
from physground.dynamics import PlanarPushModel
from physground.evaluation import EpisodeResult, aggregate_results, decision_sufficiency_report, normalized_parameter_error
from physground.planning import PushPlanner, RiskMode, generate_goal_directed_actions
from physground.probing import DecisionDirectedProber, FisherProber, RandomProber
from physground.runtime import ControllerConfig, PhysGroundController


@dataclass(frozen=True, slots=True)
class PhysicsWorld:
    params: PhysicsParams
    lateral_bias: float = 0.0
    yaw_bias: float = 0.0
    stiction_distance: float = 0.0

    def step(self, model: PlanarPushModel, pose: Pose2D, action: PushAction) -> Pose2D:
        if action.distance < self.stiction_distance:
            return pose
        after = model.step(pose, action, self.params)
        if not self.lateral_bias and not self.yaw_bias:
            return after
        lateral = self.lateral_bias * action.distance
        return Pose2D(
            x=after.x - lateral * math.sin(action.direction),
            y=after.y + lateral * math.cos(action.direction),
            yaw=wrap_angle(after.yaw + self.yaw_bias * action.distance),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    regime: str
    initial_pose: Pose2D
    goal: Goal2D
    world: PhysicsWorld
    prior: ParticleBelief


def _belief(params: list[PhysicsParams]) -> ParticleBelief:
    return ParticleBelief([PhysicsParticle(item, 1.0) for item in params])


def _physics(values: list[float]) -> PhysicsParams:
    return PhysicsParams(
        friction=float(values[0]),
        cof_x=float(values[1]),
        cof_y=float(values[2]),
        rotational_drag=float(values[3]),
    )


def default_reference_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "reference" / "pushbench_v1.json"


def load_cases(path: Path | None = None) -> list[BenchmarkCase]:
    source = path or default_reference_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "PhysGround-PushBench-v1":
        raise ValueError(f"unsupported benchmark payload: {payload.get('benchmark')}")

    priors = {
        name: [_physics(values) for values in rows]
        for name, rows in payload["priors"].items()
    }
    goals = {
        name: Goal2D(
            Pose2D(float(values[0]), float(values[1]), float(values[2])),
            position_tolerance=float(values[3]),
            yaw_tolerance=float(values[4]),
        )
        for name, values in payload["goals"].items()
    }

    cases: list[BenchmarkCase] = []
    for row in payload["cases"]:
        case_id, regime, goal_id, prior_id, truth_index, *tail = row
        prior = priors[str(prior_id)]
        truth = prior[int(truth_index)]
        mismatch = tail[0] if tail else {}
        cases.append(
            BenchmarkCase(
                case_id=str(case_id),
                regime=str(regime),
                initial_pose=Pose2D(0.0, 0.0, 0.0),
                goal=goals[str(goal_id)],
                world=PhysicsWorld(
                    truth,
                    lateral_bias=float(mismatch.get("lateral_bias", 0.0)),
                    yaw_bias=float(mismatch.get("yaw_bias", 0.0)),
                    stiction_distance=float(mismatch.get("stiction_distance", 0.0)),
                ),
                prior=_belief(prior),
            )
        )
    return cases


def default_cases() -> list[BenchmarkCase]:
    """Load the versioned 19-case reference benchmark from data/reference."""
    return load_cases()


METHODS = ("oracle", "nominal", "robust", "random", "fisher", "physground")


def _controller(method: str, model: PlanarPushModel) -> PhysGroundController:
    baseline = dict(act_when_probe_exhausted=True, mismatch_abstain_threshold=float("inf"))
    if method == "nominal":
        return PhysGroundController(
            planner=PushPlanner(model, risk_mode=RiskMode.MEAN),
            prober=None,
            config=ControllerConfig(max_probes=0, **baseline),
        )
    if method == "robust":
        return PhysGroundController(
            planner=PushPlanner(model, risk_mode=RiskMode.WORST),
            prober=None,
            config=ControllerConfig(max_probes=0, **baseline),
        )
    if method == "random":
        return PhysGroundController(
            planner=PushPlanner(model),
            prober=RandomProber(seed=7),
            config=ControllerConfig(max_probes=2, **baseline),
        )
    if method == "fisher":
        return PhysGroundController(
            planner=PushPlanner(model),
            prober=FisherProber(model),
            config=ControllerConfig(max_probes=2, **baseline),
        )
    if method == "physground":
        return PhysGroundController(
            planner=PushPlanner(model),
            prober=DecisionDirectedProber(model),
            config=ControllerConfig(
                max_probes=2,
                act_when_probe_exhausted=True,
                minimum_expected_reduction=0.0,
            ),
        )
    if method == "oracle":
        return PhysGroundController(
            planner=PushPlanner(model),
            prober=None,
            config=ControllerConfig(max_probes=0, **baseline),
        )
    raise ValueError(f"unknown benchmark method: {method}")


def run_episode(
    case: BenchmarkCase,
    method: str,
    *,
    model: PlanarPushModel | None = None,
    max_actions: int = 4,
    confidence_threshold: float = 0.9,
) -> EpisodeResult:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    reduced = model or PlanarPushModel()
    belief = case.prior.copy()
    if method == "oracle":
        belief = ParticleBelief([PhysicsParticle(case.world.params, 1.0)])
    elif method == "nominal":
        belief = ParticleBelief([PhysicsParticle(belief.mean(), 1.0)])

    controller = _controller(method, reduced)
    pose = case.initial_pose
    probes = 0
    acts = 0
    regret_values: list[float] = []
    high_confidence_acts = 0
    trace: list[dict] = []
    abstained = False
    model_mismatch_score = 0.0

    for step_index in range(max_actions + controller.config.max_probes + 1):
        if case.goal.reached(pose):
            break
        decision = controller.decide(
            pose=pose,
            goal=case.goal,
            belief=belief,
            probes_used=probes,
            model_mismatch_score=model_mismatch_score,
        )
        trace.append(
            {
                "step": step_index,
                "mode": decision.mode.value,
                "pose": asdict(pose),
                "decision_disagreement": decision.decision_disagreement,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "diagnostics": decision.diagnostics,
            }
        )
        if decision.mode == DecisionMode.ABSTAIN:
            abstained = True
            break
        if decision.action is None:
            raise RuntimeError("ACT/PROBE decision must include an action")

        before = pose
        pose = case.world.step(reduced, pose, decision.action)
        model_mismatch_score = belief.predictive_residual_score(
            before=before,
            action=decision.action,
            observed_after=pose,
            model=reduced,
        )
        belief.update(
            before=before,
            action=decision.action,
            observed_after=pose,
            model=reduced,
            resample_threshold=None,
        )
        if decision.mode == DecisionMode.PROBE:
            probes += 1
            continue

        acts += 1
        if decision.confidence >= confidence_threshold:
            high_confidence_acts += 1
        candidates = generate_goal_directed_actions(before, case.goal)
        oracle = PushPlanner(reduced).oracle_plan(before, case.goal, case.world.params, candidates)
        chosen_cost = PushPlanner(reduced).cost_for_params(
            before, case.goal, decision.action, case.world.params
        )
        regret_values.append(max(0.0, chosen_cost - oracle.evaluation.robust_cost))
        if acts >= max_actions:
            break

    success = case.goal.reached(pose)
    estimate = belief.mean()
    regret = sum(regret_values) / len(regret_values) if regret_values else 0.0
    false_confident_failure = bool(high_confidence_acts > 0 and not success and not abstained)
    return EpisodeResult(
        case_id=case.case_id,
        regime=case.regime,
        method=method,
        success=success,
        abstained=abstained,
        final_pose=pose,
        goal_pose=case.goal.pose,
        probes=probes,
        actions=acts,
        decision_regret=float(regret),
        parameter_error=normalized_parameter_error(estimate, case.world.params),
        false_confident_failure=false_confident_failure,
        high_confidence_acts=high_confidence_acts,
        trace=tuple(trace),
    )


def run_benchmark(
    cases: Iterable[BenchmarkCase], *, methods: Iterable[str] = METHODS
) -> dict[str, Any]:
    case_list = list(cases)
    method_list = list(methods)
    results = [run_episode(case, method) for method in method_list for case in case_list]
    by_method = {
        method: aggregate_results([row for row in results if row.method == method])
        for method in method_list
    }
    by_regime = {
        regime: {
            method: aggregate_results(
                [row for row in results if row.method == method and row.regime == regime]
            )
            for method in method_list
        }
        for regime in sorted({case.regime for case in case_list})
    }
    episodes = [
        {
            **asdict(row),
            "final_pose": asdict(row.final_pose),
            "goal_pose": asdict(row.goal_pose),
        }
        for row in results
    ]
    return {
        "benchmark": "PhysGround-PushBench-v1",
        "cases": len(case_list),
        "methods": method_list,
        "by_method": by_method,
        "by_regime": by_regime,
        "decision_sufficiency": decision_sufficiency_report(episodes),
        "episodes": episodes,
    }


def evaluate_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    regimes = payload["by_regime"]
    represented = [name for name in regimes if name != "P5_MODEL_MISMATCH"]

    def weighted(method: str, key: str) -> float:
        total = 0.0
        episodes = 0
        for regime in represented:
            row = regimes[regime][method]
            count = int(row["episodes"])
            total += float(row[key]) * count
            episodes += count
        return total / max(episodes, 1)

    fisher_success = weighted("fisher", "success_rate")
    phys_success = weighted("physground", "success_rate")
    fisher_probes = weighted("fisher", "mean_probes")
    phys_probes = weighted("physground", "mean_probes")
    fisher_false_conf = weighted("fisher", "false_confidence_rate")
    phys_false_conf = weighted("physground", "false_confidence_rate")
    mismatch = regimes["P5_MODEL_MISMATCH"]["physground"]
    checks = {
        "oracle_solves_represented_regimes": weighted("oracle", "success_rate") >= 0.99,
        "physground_matches_fisher_success": phys_success >= fisher_success - 0.02,
        "physground_uses_fewer_probes": phys_probes <= fisher_probes,
        "physground_not_more_false_confident": phys_false_conf <= fisher_false_conf,
        "physground_abstains_on_model_mismatch": float(mismatch["abstention_rate"]) >= 0.90,
        "physground_no_false_confidence_on_mismatch": float(mismatch["false_confidence_rate"]) == 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            "represented_physground_success": phys_success,
            "represented_fisher_success": fisher_success,
            "represented_physground_mean_probes": phys_probes,
            "represented_fisher_mean_probes": fisher_probes,
            "represented_physground_false_confidence": phys_false_conf,
            "represented_fisher_false_confidence": fisher_false_conf,
            "mismatch_physground_abstention_rate": float(mismatch["abstention_rate"]),
        },
    }

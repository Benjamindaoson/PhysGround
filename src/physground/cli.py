from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from physground.benchmark import METHODS, default_cases, evaluate_acceptance, run_benchmark
from physground.belief import ParticleBelief, PhysicsParticle
from physground.contracts import Goal2D, PhysicsParams, Pose2D, PushAction, Transition
from physground.dynamics import PlanarPushModel
from physground.dataset import BACKENDS, generate_transition_dataset
from physground.jax_batch import smoke as jax_smoke
from physground.preflight import collect_fingerprint, evaluate_readiness
from physground.planning import PushPlanner, compute_decision_disagreement, generate_goal_directed_actions
from physground.sysid import FiniteDifferenceSysID


def _write(payload: dict, output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def benchmark_command(args: argparse.Namespace) -> int:
    requested = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(unknown)}")
    cases = default_cases()
    if args.regime:
        cases = [case for case in cases if case.regime == args.regime]
    if args.limit is not None:
        cases = cases[: args.limit]
    payload = run_benchmark(cases, methods=requested)
    if args.require_pass:
        if set(requested) != set(METHODS) or args.regime or args.limit is not None:
            raise SystemExit("--require-pass requires the full default benchmark and all methods")
        gate = evaluate_acceptance(payload)
        payload["acceptance"] = gate
        _write(payload, args.output)
        return 0 if gate["passed"] else 2
    _write(payload, args.output)
    return 0


def sysid_command(args: argparse.Namespace) -> int:
    model = PlanarPushModel()
    truth = PhysicsParams(0.82, 0.014, -0.009, 1.35)
    initial = PhysicsParams(0.45, 0.0, 0.0, 0.8)
    pose = Pose2D(0.0, 0.0, 0.0)
    actions = [
        PushAction(3.14159, -0.6, 0.0, 0.025, 0.03),
        PushAction(3.14159, 0.6, 0.0, 0.025, 0.03),
        PushAction(-1.5708, 0.5, 1.5708, 0.025, 0.03),
    ]
    transitions: list[Transition] = []
    current = pose
    for action in actions:
        after = model.step(current, action, truth)
        transitions.append(Transition(current, action, after))
        current = after
    result = FiniteDifferenceSysID(model, learning_rate=args.learning_rate).fit(
        transitions, initial, iterations=args.iterations
    )
    _write(
        {
            "truth": asdict(truth),
            "initial": asdict(initial),
            "estimate": asdict(result.params),
            "loss": result.loss,
            "iterations": result.iterations,
        },
        args.output,
    )
    return 0


def disagreement_command(args: argparse.Namespace) -> int:
    model = PlanarPushModel()
    planner = PushPlanner(model)
    pose = Pose2D(0.0, 0.0, 0.0)
    goal = Goal2D(Pose2D(0.08, 0.03, 0.2))
    belief = ParticleBelief([
        PhysicsParticle(PhysicsParams(0.5, -0.012, 0.0, 0.9), 0.5),
        PhysicsParticle(PhysicsParams(0.55, 0.012, 0.0, 1.1), 0.5),
    ])
    actions = generate_goal_directed_actions(pose, goal)
    result = compute_decision_disagreement(
        pose=pose, goal=goal, belief=belief, actions=actions, planner=planner
    )
    _write(asdict(result), args.output)
    return 0


def preflight_command(args: argparse.Namespace) -> int:
    fingerprint = collect_fingerprint()
    readiness = evaluate_readiness(
        fingerprint,
        require_rocm=args.require_rocm,
        require_jax_gpu=args.require_jax_gpu,
        require_mujoco=args.require_mujoco,
    )
    payload = {"fingerprint": fingerprint, "readiness": readiness}
    _write(payload, args.output)
    return 0 if readiness["passed"] else 2


def generate_data_command(args: argparse.Namespace) -> int:
    payload = generate_transition_dataset(
        output_dir=args.output_dir,
        records=args.records,
        shard_size=args.shard_size,
        seed=args.seed,
        backend=args.backend,
        resume=args.resume,
    )
    _write(payload, args.manifest_copy)
    return 0


def jax_smoke_command(args: argparse.Namespace) -> int:
    payload = jax_smoke(batch_size=args.batch_size, seed=args.seed)
    _write(payload, args.output)
    if not payload["finite"]:
        return 2
    if args.require_accelerator and payload["backend"] == "cpu":
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physground")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="Run PhysGround-PushBench-v1")
    benchmark.add_argument("--methods", default=",".join(METHODS))
    benchmark.add_argument("--regime")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--require-pass", action="store_true")
    benchmark.set_defaults(func=benchmark_command)
    sysid = subparsers.add_parser("sysid-demo", help="Run finite-difference system ID demo")
    sysid.add_argument("--iterations", type=int, default=80)
    sysid.add_argument("--learning-rate", type=float, default=0.08)
    sysid.add_argument("--output", type=Path)
    sysid.set_defaults(func=sysid_command)
    disagreement = subparsers.add_parser(
        "disagreement-demo", help="Inspect decision disagreement for two physics hypotheses"
    )
    disagreement.add_argument("--output", type=Path)
    disagreement.set_defaults(func=disagreement_command)

    preflight = subparsers.add_parser("preflight", help="Inspect ROCm/JAX/MuJoCo readiness")
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--require-rocm", action="store_true")
    preflight.add_argument("--require-jax-gpu", action="store_true")
    preflight.add_argument("--require-mujoco", action="store_true")
    preflight.set_defaults(func=preflight_command)

    generate = subparsers.add_parser("generate-data", help="Generate sharded transition data")
    generate.add_argument("--backend", choices=sorted(BACKENDS), default="reduced")
    generate.add_argument("--records", type=int, required=True)
    generate.add_argument("--shard-size", type=int, default=50000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--manifest-copy", type=Path)
    generate.add_argument("--resume", action="store_true")
    generate.set_defaults(func=generate_data_command)

    jax_cmd = subparsers.add_parser("jax-smoke", help="Run vectorized reduced dynamics on JAX")
    jax_cmd.add_argument("--batch-size", type=int, default=65536)
    jax_cmd.add_argument("--seed", type=int, default=0)
    jax_cmd.add_argument("--output", type=Path)
    jax_cmd.add_argument("--require-accelerator", action="store_true")
    jax_cmd.set_defaults(func=jax_smoke_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

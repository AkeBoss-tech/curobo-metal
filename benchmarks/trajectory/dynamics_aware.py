#!/usr/bin/env python3
"""Synchronized dynamics-aware trajectory correctness and latency evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from curobo_metal.backend import resolve_device, synchronize
from curobo_metal.ops.trajectory import (
    DynamicsAwareProblem,
    bspline_matrices,
    evaluate_dynamics_aware,
    optimize_dynamics_aware,
)
from curobo_metal.ops.whole_body import WholeBodyModel
from curobo_metal.reference import TreeRobot

ROOT = Path(__file__).parents[2]


def make_problem(device: torch.device, batch: int) -> DynamicsAwareProblem:
    case = json.loads(
        (ROOT / "tests/fixtures/whole_body/panda_inertial.json").read_text()
    )
    model = WholeBodyModel(
        TreeRobot.from_dict(case["robot"]), device=device, dtype=torch.float32
    )
    start = torch.tensor(case["inputs"]["q"][0], device=device, dtype=torch.float32)
    goal = torch.tensor(case["inputs"]["q"][1], device=device, dtype=torch.float32)
    if batch > 1:
        start, goal = start.expand(batch, -1).clone(), goal.expand(batch, -1).clone()
    lower, upper = torch.full((7,), -3.0, device=device), torch.full((7,), 3.0, device=device)
    return DynamicsAwareProblem(
        model, start, goal, lower, upper, control_points=10, samples=32,
        duration=3.0, velocity_limits=torch.full((7,), 3.0, device=device),
        acceleration_limits=torch.full((7,), 10.0, device=device),
        jerk_limits=torch.full((7,), 50.0, device=device),
        max_iterations=25,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch-size and repeats must be positive")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("MPS evidence requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    device = resolve_device(args.device)
    problem = make_problem(device, args.batch_size)
    matrices = bspline_matrices(10, 32, device=device)
    partition_error = float(
        (matrices.position.sum(-1) - 1).abs().max().detach().cpu().item()
    )
    samples: list[float] = []
    final = optimize_dynamics_aware(problem)
    for _ in range(args.repeats):
        synchronize(device)
        started = time.perf_counter_ns()
        final = optimize_dynamics_aware(problem)
        synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    evaluated = evaluate_dynamics_aware(problem, final.control_points, final.duration)
    payload = {
        "format": "curobo-metal-dynamics-aware-trajectory",
        "version": 1,
        "device": device.type,
        "dtype": "float32",
        "fallback_disabled": (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
            if device.type == "mps" else None
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "batch_size": args.batch_size,
        "correctness": {
            "basis_partition_max_error": partition_error,
            "finite_objective": bool(torch.isfinite(evaluated.total).all().item()),
            "success": final.success.detach().cpu().tolist(),
            "status": final.status,
            "maximum_violation": final.maximum_violation.detach().cpu().tolist(),
        },
        "latency_ms": {
            "samples": samples,
            "median": statistics.median(samples),
            "minimum": min(samples),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

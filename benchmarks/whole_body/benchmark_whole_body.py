#!/usr/bin/env python3
"""Synchronized correctness and latency evidence for whole-body operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import time
from typing import Callable

import numpy as np
import torch

from curobo_metal.backend import resolve_device, synchronize
from curobo_metal.ops.whole_body import (
    WholeBodyModel,
    bias_torque,
    gravity_torque,
    inverse_dynamics,
    mass_matrix,
    tree_forward_kinematics,
)
from curobo_metal.reference import TreeRobot

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "whole_body"
REFERENCE = ROOT / "artifacts" / "correctness" / "whole_body_reference.json"


def error_record(
    actual: torch.Tensor, expected: object, atol: float, rtol: float
) -> dict[str, float | bool]:
    observed = actual.detach().cpu().double().numpy()
    wanted = np.asarray(expected, dtype=np.float64)
    error = np.abs(observed - wanted)
    threshold = atol + rtol * np.abs(wanted)
    return {
        "max_absolute_error": float(error.max(initial=0)),
        "max_normalized_error": float(
            np.divide(
                error, threshold, out=np.zeros_like(error), where=threshold > 0
            ).max(initial=0)
        ),
        "passed": bool(np.all(error <= threshold)),
    }


def correctness(device: torch.device) -> dict[str, object]:
    reference = json.loads(REFERENCE.read_text())
    records: dict[str, object] = {}
    tolerances = {
        "transforms": (2e-4, 2e-4),
        "geometric_jacobian": (5e-4, 5e-4),
        "inverse_dynamics": (5e-4, 5e-4),
        "mass_matrix": (8e-4, 8e-4),
        "gravity": (5e-4, 5e-4),
        "bias": (5e-4, 5e-4),
        "d_tau_d_q": (8e-4, 8e-4),
        "d_tau_d_qd": (8e-4, 8e-4),
        "d_tau_d_qdd": (8e-4, 8e-4),
    }
    for name, expected in reference["cases"].items():
        path = FIXTURES / name
        case = json.loads(path.read_text())
        robot = TreeRobot.from_dict(case["robot"])
        model = WholeBodyModel(robot, device=device, dtype=torch.float32)
        tensor = lambda key: torch.tensor(
            case["inputs"][key], device=device, dtype=torch.float32
        )
        q, qd, qdd = tensor("q"), tensor("qd"), tensor("qdd")
        fk = tree_forward_kinematics(model, q)
        derivative_rows: list[list[torch.Tensor]] = [[], [], []]
        for batch_index in range(q.shape[0]):
            arguments = (q[batch_index], qd[batch_index], qdd[batch_index])
            for argument in range(3):
                def evaluate(value: torch.Tensor, argument: int = argument) -> torch.Tensor:
                    values = list(arguments)
                    values[argument] = value
                    return inverse_dynamics(model, *values).torque[0]

                derivative_rows[argument].append(
                    torch.autograd.functional.jacobian(evaluate, arguments[argument])
                )
        observed = {
            "transforms": fk.transforms,
            "geometric_jacobian": fk.geometric_jacobian,
            "inverse_dynamics": inverse_dynamics(model, q, qd, qdd).torque,
            "mass_matrix": mass_matrix(model, q),
            "gravity": gravity_torque(model, q),
            "bias": bias_torque(model, q, qd),
            "d_tau_d_q": torch.stack(derivative_rows[0]),
            "d_tau_d_qd": torch.stack(derivative_rows[1]),
            "d_tau_d_qdd": torch.stack(derivative_rows[2]),
        }
        checks = {
            key: error_record(value, expected[key], *tolerances[key])
            for key, value in observed.items()
        }
        records[name] = {
            "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checks": checks,
            "passed": all(bool(check["passed"]) for check in checks.values()),
        }
    return {
        "cases": records,
        "passed": all(bool(record["passed"]) for record in records.values()),
    }


def timed(
    function: Callable[[], object],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter_ns()
        function()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(samples)
    p95 = ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)]
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "p95_ms": p95,
    }


def benchmark(
    device: torch.device, batch_size: int, warmup: int, iterations: int
) -> dict[str, object]:
    case = json.loads((FIXTURES / "panda_inertial.json").read_text())
    robot = TreeRobot.from_dict(case["robot"])
    model = WholeBodyModel(robot, device=device, dtype=torch.float32)
    base = lambda key: torch.tensor(
        case["inputs"][key], device=device, dtype=torch.float32
    )

    def expand(key: str) -> torch.Tensor:
        value = base(key)
        repeats = (batch_size + value.shape[0] - 1) // value.shape[0]
        return value.repeat(repeats, 1)[:batch_size]

    q, qd, qdd = expand("q"), expand("qd"), expand("qdd")

    def dynamics_backward() -> None:
        values = q.detach().clone().requires_grad_(True)
        torque = inverse_dynamics(model, values, qd, qdd).torque
        torch.autograd.grad(torque.square().sum(), values)

    return {
        "batch_size": batch_size,
        "tree_fk_forward": timed(
            lambda: tree_forward_kinematics(model, q),
            device, warmup, iterations,
        ),
        "inverse_dynamics_forward": timed(
            lambda: inverse_dynamics(model, q, qd, qdd),
            device, warmup, iterations,
        ),
        "inverse_dynamics_q_backward": timed(
            dynamics_backward, device, warmup, iterations
        ),
        "mass_matrix_forward": timed(
            lambda: mass_matrix(model, q), device, warmup, iterations
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(x < 1 for x in args.batch_sizes):
        parser.error("iterations and batch sizes must be positive; warmup nonnegative")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("MPS evidence requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    device = resolve_device(args.device)
    checks = correctness(device)
    if not checks["passed"]:
        raise SystemExit("correctness validation failed")
    result = {
        "format": "curobo-metal-whole-body-ops",
        "version": 1,
        "implementation": "composed-pytorch",
        "device": device.type,
        "dtype": "float32",
        "fallback_disabled": (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
            if device.type == "mps" else None
        ),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "correctness": checks,
        "benchmarks": [
            benchmark(device, batch, args.warmup, args.iterations)
            for batch in args.batch_sizes
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(args.output)


if __name__ == "__main__":
    main()

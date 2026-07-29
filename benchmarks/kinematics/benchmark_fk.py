#!/usr/bin/env python3
"""Synchronization-correct FK benchmark and correctness evidence runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from curobo_metal.backend import resolve_device, synchronize
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.reference import SerialRobot, load_case

ROOT = Path(__file__).parents[2]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "panda_serial.json"


def timed_call(chain: KinematicChain, q: torch.Tensor) -> tuple[Any, float]:
    synchronize(q.device)
    started = time.perf_counter_ns()
    result = forward_kinematics(chain, q)
    synchronize(q.device)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def correctness(case: dict[str, Any], device: torch.device) -> dict[str, Any]:
    robot = SerialRobot.from_dict(case["robot"])
    q = torch.tensor(case["inputs"]["q"], dtype=torch.float32, device=device)
    actual, _ = timed_call(
        KinematicChain(robot, device=device, dtype=torch.float32), q
    )
    values: dict[str, Any] = {}
    tolerances = {
        "transforms": (2e-5, 2e-5),
        "transform_jacobian": (8e-5, 8e-5),
        "geometric_jacobian": (8e-5, 8e-5),
    }
    for name, (atol, rtol) in tolerances.items():
        observed = getattr(actual, name).detach().cpu().numpy()
        expected = np.asarray(case["expected"][name])
        error = np.abs(observed - expected)
        threshold = atol + rtol * np.abs(expected)
        values[name] = {
            "max_absolute_error": float(error.max(initial=0.0)),
            "max_normalized_error": float(
                np.divide(error, threshold, out=np.zeros_like(error), where=threshold > 0).max(
                    initial=0.0
                )
            ),
            "passed": bool(np.all(error <= threshold)),
        }
    values["passed"] = all(value["passed"] for value in values.values())
    return values


def benchmark(
    robot: SerialRobot,
    source_q: list[list[float]],
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    base = torch.tensor(source_q, dtype=torch.float32, device=device)
    repeats = (batch_size + base.shape[0] - 1) // base.shape[0]
    q = base.repeat((repeats, 1))[:batch_size]
    chain = KinematicChain(robot, device=device, dtype=torch.float32)

    with torch.inference_mode():
        _, first_call_ms = timed_call(chain, q)
        warmup_samples = [timed_call(chain, q)[1] for _ in range(warmup)]
        samples = [timed_call(chain, q)[1] for _ in range(iterations)]

    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "batch_size": batch_size,
        "first_call_ms": first_call_ms,
        "warmup": {
            "iterations": warmup,
            "total_ms": sum(warmup_samples),
        },
        "steady_state": {
            "iterations": iterations,
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "min_ms": min(samples),
            "p95_ms": ordered[p95_index],
            "configurations_per_second_median": batch_size
            / (statistics.median(samples) / 1000.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    if any(batch < 1 for batch in args.batch_sizes):
        parser.error("batch sizes must be positive")

    device = resolve_device(args.device)
    case = load_case(args.fixture)
    robot = SerialRobot.from_dict(case["robot"])
    result = {
        "format": "curobo-metal-fk-benchmark",
        "version": 1,
        "device": str(device),
        "dtype": "float32",
        "implementation": "composed-pytorch",
        "fallback": {
            "environment_value": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "fixture": str(args.fixture),
        "correctness": correctness(case, device),
        "benchmarks": [
            benchmark(
                robot,
                case["inputs"]["q"],
                device,
                batch,
                args.warmup,
                args.iterations,
            )
            for batch in args.batch_sizes
        ],
    }
    if not result["correctness"]["passed"]:
        raise SystemExit("correctness validation failed")
    if device.type == "mps" and not result["fallback"]["disabled"]:
        raise SystemExit("MPS evidence requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

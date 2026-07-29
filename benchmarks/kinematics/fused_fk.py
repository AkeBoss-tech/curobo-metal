#!/usr/bin/env python3
"""Benchmark fused Metal FK with compilation, validation, and steady state split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from curobo_metal.backend import synchronize
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.ops.kinematics.forward import _validate_q
from curobo_metal.ops.kinematics.metal import fused_forward_kinematics
from curobo_metal.reference import SerialRobot, load_case

ROOT = Path(__file__).parents[2]


def timed(call):
    synchronize("mps")
    started = time.perf_counter_ns()
    result = call()
    synchronize("mps")
    return result, (time.perf_counter_ns() - started) / 1e6


def summary(samples):
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999) - 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "tests/fixtures/panda_serial.json"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(x < 1 for x in args.batch_sizes):
        parser.error("positive batches/iterations and nonnegative warmup required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    case = load_case(args.fixture)
    robot = SerialRobot.from_dict(case["robot"])
    chain = KinematicChain(robot, device="mps", dtype=torch.float32)
    base = torch.tensor(case["inputs"]["q"], dtype=torch.float32, device="mps")
    records = []
    with torch.inference_mode():
        for batch in args.batch_sizes:
            q = base.repeat(((batch + len(base) - 1) // len(base), 1))[:batch]
            _, first_public = timed(lambda: forward_kinematics(chain, q))
            validation = [
                timed(lambda: _validate_q(chain, q))[1] for _ in range(args.iterations)
            ]
            core = lambda: fused_forward_kinematics(chain, q)
            for _ in range(args.warmup):
                timed(core)
            core_samples = [timed(core)[1] for _ in range(args.iterations)]
            public_samples = [
                timed(lambda: forward_kinematics(chain, q))[1]
                for _ in range(args.iterations)
            ]
            records.append(
                {
                    "batch_size": batch,
                    "first_public_call_ms": first_public,
                    "validation": summary(validation),
                    "fused_core": summary(core_samples),
                    "public_end_to_end": summary(public_samples),
                }
            )

    result = {
        "format": "curobo-metal-fused-fk-benchmark",
        "version": 1,
        "implementation": "runtime-compiled-metal-single-forward-dispatch",
        "fallback_disabled": True,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()

"""Synchronized trajectory latency and outcome benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import torch

from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory import TrajectoryProblem, optimize_trajectory
from curobo_metal.reference import SerialRobot


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    root = Path(__file__).parents[2]
    case = json.loads((root / "tests/fixtures/trajectory/two_link_obstacle_free.json").read_text())
    inputs = case["inputs"]
    tensor = lambda value: torch.tensor(value, dtype=dtype, device=device)
    problem = TrajectoryProblem(
        KinematicChain(SerialRobot.from_dict(case["robot"]), device=device, dtype=dtype),
        tensor(inputs["start"]), tensor(inputs["goal"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]), inputs["steps"], inputs["dt"], max_iterations=100,
    )
    compile_start = time.perf_counter()
    warm = optimize_trajectory(problem)
    synchronize(device)
    compile_ms = 1e3 * (time.perf_counter() - compile_start)
    samples = []
    final = warm
    for _ in range(args.repeats):
        synchronize(device)
        start = time.perf_counter()
        final = optimize_trajectory(problem)
        synchronize(device)
        samples.append(1e3 * (time.perf_counter() - start))
    document = {
        "format": "curobo-metal-trajectory-benchmark", "version": 1,
        "device": args.device, "dtype": str(dtype), "compile_warmup_ms": compile_ms,
        "fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "torch": torch.__version__},
        "latency_ms": {"median": torch.tensor(samples).median().item(), "samples": samples},
        "success": final.success.tolist(), "status": list(final.status),
        "endpoint_error": final.endpoint_error.tolist(),
        "minimum_clearance": [None if torch.isinf(x) else x.item() for x in final.minimum_clearance],
        "maximum_limit_violation": final.maximum_limit_violation.tolist(),
    }
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

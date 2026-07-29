#!/usr/bin/env python3
"""Synchronization-correct collision correctness and throughput runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from curobo_metal.backend import resolve_device, synchronize
from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
    transform_spheres,
)
from curobo_metal.reference.collision import load_collision_case

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "collision"


def timed(call: Callable[[], object], device: torch.device) -> float:
    synchronize(device)
    started = time.perf_counter_ns()
    call()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1_000_000


def summary(samples: list[float], work_items: int) -> dict[str, object]:
    ordered = sorted(samples)
    p95 = ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)]
    median = statistics.median(samples)
    return {
        "iterations": len(samples),
        "samples_ms": samples,
        "median_ms": median,
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "p95_ms": p95,
        "work_items_per_second_median": work_items / (median / 1000),
    }


def measure(
    name: str,
    call: Callable[[], object],
    device: torch.device,
    work_items: int,
    warmup: int,
    iterations: int,
    dimensions: dict[str, int],
) -> dict[str, object]:
    with torch.inference_mode():
        compile_or_first_call_ms = timed(call, device)
        warmup_samples = [timed(call, device) for _ in range(warmup)]
        samples = [timed(call, device) for _ in range(iterations)]
    return {
        "operation": name,
        "dimensions": dimensions,
        "compile_or_first_call_ms": compile_or_first_call_ms,
        "warmup": {"iterations": warmup, "total_ms": sum(warmup_samples)},
        "steady_state": summary(samples, work_items),
    }


def make_workload(
    device: torch.device, batch: int, spheres: int, cuboids: int
) -> tuple[Callable[[], object], Callable[[], object], Callable[[], object]]:
    generator = torch.Generator(device="cpu").manual_seed(7321)
    cpu_spheres = torch.rand((batch, spheres, 4), generator=generator)
    cpu_spheres[..., :3] = cpu_spheres[..., :3] * 4 - 2
    cpu_spheres[..., 3] = cpu_spheres[..., 3] * 0.15 + 0.02
    sphere_values = cpu_spheres.to(device)
    pairs = torch.combinations(torch.arange(spheres), r=2)[::2].to(device)
    centers = (torch.rand((cuboids, 3), generator=generator) * 4 - 2).to(device)
    rotations = torch.eye(3).expand(cuboids, 3, 3).clone().to(device)
    half = (torch.rand((cuboids, 3), generator=generator) + 0.1).to(device)
    transforms = torch.eye(4).expand(batch, spheres, 4, 4).clone().to(device)
    transforms[..., :3, 3] = sphere_values[..., :3]
    local = torch.zeros((spheres, 4), device=device)
    local[:, 3] = sphere_values[0, :, 3]
    links = torch.arange(spheres, dtype=torch.int64, device=device)
    return (
        lambda: transform_spheres(transforms, local, links),
        lambda: sphere_sphere_signed_distance(sphere_values, pairs),
        lambda: sphere_cuboid_signed_distance(
            sphere_values, centers, rotations, half
        ),
    )


def correctness(device: torch.device) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for fixture_name in ("minimal.json", "realistic.json"):
        replay = load_collision_case(FIXTURES / fixture_name)
        for case in replay["cases"]:
            inputs, expected = case["inputs"], case["expected"]
            operation = case["operation"]
            if operation == "transform_spheres":
                observed = transform_spheres(
                    torch.tensor(inputs["transforms"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["local_spheres"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["link_indices"], dtype=torch.int64, device=device),
                ).spheres
                target = expected["spheres"]
            elif operation == "sphere_sphere":
                result = sphere_sphere_signed_distance(
                    torch.tensor(inputs["spheres"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["pairs"], dtype=torch.int64, device=device),
                    sphere_active=(
                        torch.tensor(inputs["sphere_active"], dtype=torch.bool, device=device)
                        if "sphere_active" in inputs else None
                    ),
                    pair_active=(
                        torch.tensor(inputs["pair_active"], dtype=torch.bool, device=device)
                        if "pair_active" in inputs else None
                    ),
                    padding=inputs.get("padding", 0),
                )
                observed = result.reduced_distance
                target = np.min(np.asarray(expected["distance"]), axis=1)
            else:
                result = sphere_cuboid_signed_distance(
                    torch.tensor(inputs["spheres"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["centers"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["rotations"], dtype=torch.float32, device=device),
                    torch.tensor(inputs["half_extents"], dtype=torch.float32, device=device),
                    cuboid_active=(
                        torch.tensor(inputs["cuboid_active"], dtype=torch.bool, device=device)
                        if "cuboid_active" in inputs else None
                    ),
                    padding=inputs.get("padding", 0),
                )
                observed = result.reduced_distance
                target = expected["distance"]
            error = np.abs(observed.cpu().numpy() - np.asarray(target))
            threshold = 2e-6 + 2e-5 * np.abs(np.asarray(target))
            checks.append(
                {
                    "fixture": fixture_name,
                    "case": case["name"],
                    "operation": operation,
                    "max_absolute_error": float(error.max(initial=0)),
                    "passed": bool(np.all(error <= threshold)),
                }
            )
    return {"checks": checks, "passed": all(c["passed"] for c in checks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 512])
    parser.add_argument("--sphere-count", type=int, default=32)
    parser.add_argument("--cuboid-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.iterations < 1
        or args.sphere_count < 2
        or args.cuboid_count < 0
        or any(batch < 1 for batch in args.batch_sizes)
    ):
        parser.error("counts and iterations are outside their valid ranges")
    device = resolve_device(args.device)
    fallback_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if device.type == "mps" and fallback_value != "0":
        raise SystemExit("MPS evidence requires PYTORCH_ENABLE_MPS_FALLBACK=0")

    correct = correctness(device)
    if not correct["passed"]:
        raise SystemExit("correctness validation failed")
    benchmarks: list[dict[str, object]] = []
    pair_count = args.sphere_count * (args.sphere_count - 1) // 4
    for batch in args.batch_sizes:
        transform, pair, cuboid = make_workload(
            device, batch, args.sphere_count, args.cuboid_count
        )
        dimensions = {
            "batch": batch,
            "spheres": args.sphere_count,
            "cuboids": args.cuboid_count,
            "pairs": pair_count,
        }
        benchmarks.extend(
            (
                measure("transform_spheres", transform, device, batch * args.sphere_count,
                        args.warmup, args.iterations, dimensions),
                measure("sphere_sphere", pair, device, batch * pair_count,
                        args.warmup, args.iterations, dimensions),
                measure("sphere_cuboid", cuboid, device,
                        batch * args.sphere_count * args.cuboid_count,
                        args.warmup, args.iterations, dimensions),
            )
        )
    result = {
        "format": "curobo-metal-collision-benchmark",
        "version": 1,
        "device": str(device),
        "dtype": "float32",
        "implementation": "composed-pytorch",
        "custom_metal": False,
        "fallback": {
            "environment_value": fallback_value,
            "disabled": fallback_value == "0",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "correctness": correct,
        "timing_protocol": {
            "device_synchronized_before_and_after_each_sample": True,
            "compile_or_first_call_excluded_from_warmup_and_steady_state": True,
            "warmup_excluded_from_steady_state": True,
        },
        "benchmarks": benchmarks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

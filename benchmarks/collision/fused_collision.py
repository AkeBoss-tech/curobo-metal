#!/usr/bin/env python3
"""Attribute composed collision materialization costs and benchmark fused MPS."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from curobo_metal.backend import synchronize
from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
)


def _time(call, device, warmup, iterations):
    synchronize(device)
    start = time.perf_counter_ns()
    call()
    synchronize(device)
    first = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmup):
        call()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter_ns()
        call()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"first_call_ms": first, "steady_state_median_ms": statistics.median(samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 512, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(7321)
    rows = []
    for batch in args.batch_sizes:
        spheres = torch.rand((batch, 32, 4), generator=generator)
        spheres[..., :3] = spheres[..., :3] * 4 - 2
        spheres[..., 3] = spheres[..., 3] * 0.15 + 0.02
        spheres = spheres.to(device)
        pairs = torch.combinations(torch.arange(32), r=2)[::2].to(device)
        centers = (torch.rand((16, 3), generator=generator) * 4 - 2).to(device)
        rotations = torch.eye(3).expand(16, 3, 3).clone().to(device)
        half = (torch.rand((16, 3), generator=generator) + 0.1).to(device)

        first, second = pairs[:, 0], pairs[:, 1]
        pair_distance_only = lambda: (
            torch.linalg.vector_norm(
                spheres[:, first, :3] - spheres[:, second, :3], dim=-1
            )
            - spheres[:, first, 3]
            - spheres[:, second, 3]
        )
        pair_distance_and_reduction = lambda: pair_distance_only().min(dim=1)
        offset = spheres[:, :, None, :3] - centers[None, None, :, :]
        cuboid_clearance_only = lambda: (
            (
                (offset.abs() - half[None, None]).clamp_min(0)
            ).square().sum(dim=-1).sqrt()
            + (offset.abs() - half[None, None]).max(dim=-1).values.clamp_max(0)
            - spheres[:, :, None, 3]
        )
        calls = {
            "pair_composed_distance_only": pair_distance_only,
            "pair_composed_distance_plus_reduction": pair_distance_and_reduction,
            "pair_public_full_result": lambda: sphere_sphere_signed_distance(spheres, pairs),
            "cuboid_composed_clearance_only_axis_aligned": cuboid_clearance_only,
            "cuboid_public_full_result": lambda: sphere_cuboid_signed_distance(
                spheres, centers, rotations, half
            ),
        }
        for name, call in calls.items():
            rows.append(
                {
                    "operation": name,
                    "batch": batch,
                    **_time(call, device, args.warmup, args.iterations),
                }
            )
    result = {
        "format": "curobo-metal-fused-collision-profile",
        "version": 1,
        "device": args.device,
        "dtype": "float32",
        "timing": "first call, warmup, and synchronized steady state separated",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

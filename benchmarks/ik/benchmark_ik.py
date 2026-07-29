#!/usr/bin/env python3
"""Synchronized fixed-suite IK correctness and latency benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from curobo_metal.backend import synchronize
from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.ik import IKProblem, solve_ik
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.reference import SerialRobot

ROOT = Path(__file__).parents[2]
NAMES = ("reachable", "unreachable", "limit_constrained", "collision_constrained")


def load_problem(name: str, device: str) -> tuple[IKProblem, float]:
    start = time.perf_counter()
    case = json.loads((ROOT / "tests" / "fixtures" / "ik" / f"{name}.json").read_text())
    raw, options = case["inputs"], case["options"]
    dtype = torch.float32 if device == "mps" else torch.float64
    tensor = lambda value: torch.tensor(value, device=device, dtype=dtype)
    model = None
    if "collision" in raw:
        value = raw["collision"]
        model = CollisionModel(
            tensor(value["local_spheres"]),
            torch.tensor(value["link_indices"], device=device, dtype=torch.int64),
            cuboid_centers=tensor(value["cuboid_centers"]),
            cuboid_rotations=tensor(value["cuboid_rotations"]),
            cuboid_half_extents=tensor(value["cuboid_half_extents"]),
            activation_distance=value["activation_distance"],
            weight=value["weight"],
        )
    problem = IKProblem(
        KinematicChain(SerialRobot.from_dict(case["robot"]), device=device, dtype=dtype),
        tensor(raw["target_position"]), tensor(raw["target_quaternion"]),
        tensor(raw["seeds"]), tensor(raw["lower"]), tensor(raw["upper"]),
        tensor(raw["pose_weights"]),
        position_tolerance=options["position_tolerance"],
        rotation_tolerance=options["rotation_tolerance"],
        max_iterations=options["max_iterations"],
        step_tolerance=options["step_tolerance"],
        collision_model=model,
    )
    synchronize(device)
    return problem, (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 for an auditable MPS run")

    records = []
    for name in NAMES:
        problem, compile_ms = load_problem(name, args.device)
        start = time.perf_counter()
        warm = solve_ik(problem)
        synchronize(args.device)
        warmup_ms = (time.perf_counter() - start) * 1e3
        samples = []
        final = warm
        for _ in range(args.repeats):
            start = time.perf_counter()
            final = solve_ik(problem)
            synchronize(args.device)
            samples.append((time.perf_counter() - start) * 1e3)
        records.append({
            "case": name,
            "compile_metadata_ms": compile_ms,
            "warmup_solve_ms": warmup_ms,
            "steady_state_solve_ms": {
                "median": statistics.median(samples),
                "minimum": min(samples),
                "samples": samples,
            },
            "status": list(final.status),
            "success": final.success.detach().cpu().tolist(),
            "position_error": final.position_error.detach().cpu().tolist(),
            "rotation_error": final.rotation_error.detach().cpu().tolist(),
            "collision_cost": final.collision_cost.detach().cpu().tolist(),
            "collision_free": final.collision_free.detach().cpu().tolist(),
        })
    document = {
        "format": "curobo-metal-ik-ops-benchmark",
        "version": 1,
        "device": args.device,
        "dtype": "float32" if args.device == "mps" else "float64",
        "mps_fallback_disabled": (
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
            if args.device == "mps" else None
        ),
        "repeats": args.repeats,
        "success_rate": sum(bool(r["success"][0]) for r in records) / len(records),
        "expected_success_rate": 0.25,
        "cases": records,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

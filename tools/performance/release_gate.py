#!/usr/bin/env python3
"""Fail closed when a release benchmark exceeds its Apple Silicon ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CEILINGS_MS = {
    "fk_batch_1024": 12.0,
    "collision_sphere_sphere_batch_512": 25.0,
    "collision_sphere_cuboid_batch_512": 40.0,
    "ik_reachable": 1500.0,
    "trajectory": 600.0,
    "prm": 250.0,
    "perception_update": 25.0,
    "perception_query": 20.0,
}


def _load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text())


def collect(root: Path) -> dict[str, float]:
    fk = _load(root, "fk.json")
    collision = _load(root, "collision.json")
    ik = _load(root, "ik.json")
    trajectory = _load(root, "trajectory.json")
    graph = _load(root, "graph.json")
    perception = _load(root, "perception.json")
    fk_1024 = next(x for x in fk["benchmarks"] if x["batch_size"] == 1024)
    collision_by_key = {
        (x["operation"], x["dimensions"]["batch"]): x for x in collision["benchmarks"]
    }
    ik_reachable = next(x for x in ik["cases"] if x["case"] == "reachable")
    return {
        "fk_batch_1024": fk_1024["steady_state"]["median_ms"],
        "collision_sphere_sphere_batch_512": collision_by_key[("sphere_sphere", 512)]["steady_state"]["median_ms"],
        "collision_sphere_cuboid_batch_512": collision_by_key[("sphere_cuboid", 512)]["steady_state"]["median_ms"],
        "ik_reachable": ik_reachable["steady_state_solve_ms"]["median"],
        "trajectory": trajectory["latency_ms"]["median"],
        "prm": graph["latency_ms"]["median"],
        "perception_update": perception["update_ms_median"],
        "perception_query": perception["query_ms_median"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    measurements = collect(args.input)
    checks = {
        key: {"median_ms": value, "ceiling_ms": CEILINGS_MS[key], "passed": value <= CEILINGS_MS[key]}
        for key, value in measurements.items()
    }
    document = {
        "format": "curobo-metal-release-performance-gate",
        "version": 1,
        "policy": "synchronized warm median on the self-hosted Apple Silicon release runner",
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

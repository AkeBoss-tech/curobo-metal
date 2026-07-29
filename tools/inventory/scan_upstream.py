#!/usr/bin/env python3
"""Generate a deterministic dependency census from a pinned cuRoboV2 checkout."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

PINNED_SHA = "8e734f3ced1df898990bcd92de40abce475907db"

WORKLOAD_FILES = (
    "curobo/kinematics.py",
    "curobo/collision_checking.py",
    "curobo/_src/robot/kinematics/kinematics.py",
    "curobo/_src/curobolib/cuda_ops/kinematics.py",
    "curobo/_src/collision/collision_robot_scene.py",
    "curobo/_src/collision/collision_robot_scene_cfg.py",
    "curobo/_src/cost/cost_scene_collision.py",
    "curobo/_src/cost/cost_self_collision.py",
    "curobo/_src/curobolib/cuda_ops/geometry.py",
    "curobo/_src/geom/collision/collision_scene.py",
    "curobo/_src/geom/collision/checker_collision.py",
    "curobo/_src/geom/collision/wp_autograd.py",
    "curobo/_src/geom/collision/wp_collision_kernel.py",
    "curobo/_src/geom/collision/wp_collision_common.py",
    "curobo/_src/geom/data/data_scene.py",
    "curobo/_src/geom/data/data_cuboid.py",
    "curobo/_src/util/warp.py",
)

PATTERNS = {
    "cuda": re.compile(r"\b(?:torch\.cuda|cuda_core|kinematics_cu|geometry_cu|\.cu\b|\.cuh\b)"),
    "warp": re.compile(r"\b(?:import warp|warp as wp|wp\.|init_warp|get_warp_device_stream)\b"),
    "cuda_graph": re.compile(r"\b(?:CUDAGraph|cuda_graph|cuda\.graph)\b", re.IGNORECASE),
    "isaac": re.compile(r"\b(?:isaac|omni\.|SimulationApp)\b", re.IGNORECASE),
    "torch": re.compile(r"\b(?:import torch|torch\.)\b"),
}


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return sorted(set(found))


def evidence(source: Path, relative: str) -> dict:
    path = source / relative
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    matches = {
        name: [i for i, line in enumerate(lines, 1) if pattern.search(line)]
        for name, pattern in PATTERNS.items()
    }
    return {
        "path": relative,
        "sha256": subprocess.check_output(
            ["shasum", "-a", "256", str(path)], text=True
        ).split()[0],
        "imports": imports(path) if path.suffix == ".py" else [],
        "matches": {name: nums for name, nums in matches.items() if nums},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-other-revision", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    sha = git(source, "rev-parse", "HEAD")
    if sha != PINNED_SHA and not args.allow_other_revision:
        raise SystemExit(f"expected {PINNED_SHA}, found {sha}")

    missing = [item for item in WORKLOAD_FILES if not (source / item).is_file()]
    if missing:
        raise SystemExit(f"missing expected upstream files: {missing}")

    files = [evidence(source, item) for item in WORKLOAD_FILES]
    counts = Counter()
    for item in files:
        counts.update({name: len(lines) for name, lines in item["matches"].items()})

    result = {
        "schema_version": 1,
        "upstream": {
            "repository": "https://github.com/NVlabs/curobo.git",
            "revision": sha,
            "commit_date": git(source, "show", "-s", "--format=%cI", "HEAD"),
            "subject": git(source, "show", "-s", "--format=%s", "HEAD"),
            "license_spdx": "Apache-2.0",
        },
        "scope": {
            "workloads": [
                "Kinematics.compute_kinematics (poses and robot spheres)",
                "RobotSceneCollision.get_self_collision_distance",
                "RobotSceneCollision.get_collision_distance (discrete primitives)",
            ],
            "excluded": ["swept collision", "mesh", "voxel/ESDF", "motion solvers"],
        },
        "pattern_match_counts": dict(sorted(counts.items())),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

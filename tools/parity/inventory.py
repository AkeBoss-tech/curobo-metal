#!/usr/bin/env python3
"""Generate the pinned, evidence-backed cuRoboV2 capability inventory."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

PIN = "8e734f3ced1df898990bcd92de40abce475907db"
REPOSITORY = "https://github.com/NVlabs/curobo.git"
CLASSES = ("compatible", "semantically_equivalent", "partial", "unsupported", "not_applicable")

# This is deliberately curated: equivalence is a semantic decision, not something
# an AST scanner can infer.  Evidence and symbols are validated against both trees.
CAPABILITIES = [
    ("configuration", "robot_config", "curobo/_src/types/robot.py", "RobotCfg", "src/curobo_metal/compat/curobo_v2.py", "convert_kinematics_config", "partial", "Converted kinematics metadata only; no YAML/URDF/USD/XRDF parser or mutable RobotCfg."),
    ("kinematics", "forward_kinematics", "curobo/_src/robot/kinematics/kinematics.py", "compute_kinematics", "src/curobo_metal/ops/kinematics/forward.py", "forward_kinematics", "semantically_equivalent", "Portable batched poses and spheres; API containers and fused CUDA execution differ."),
    ("kinematics", "jacobian", "curobo/_src/robot/kinematics/kinematics.py", "get_robot_as_spheres", None, None, "unsupported", "No public geometric Jacobian API; gradients exist through PyTorch autograd only."),
    ("collision", "self_collision", "curobo/_src/collision/collision_robot_scene.py", "get_self_collision_distance", "src/curobo_metal/ops/collision/core.py", "sphere_sphere_signed_distance", "partial", "Pair primitives exist; upstream cache, pair filtering, max reduction, and RobotSceneCollision wrapper are absent."),
    ("collision", "cuboid_world", "curobo/_src/collision/collision_robot_scene.py", "get_collision_distance", "src/curobo_metal/ops/collision/core.py", "sphere_cuboid_signed_distance", "partial", "Differentiable discrete cuboid query exists without upstream checker/cache/environment API."),
    ("collision", "mesh_world", "curobo/_src/geom/collision/buffer_collision.py", "CollisionBuffer", "src/curobo_metal/ops/world_collision/core.py", "mesh_distance", "partial", "Triangle distance is provided; Warp BVH, signed watertight semantics, cache, and swept query parity are absent."),
    ("collision", "voxel_esdf", "curobo/_src/geom/collision/buffer_collision.py", "CollisionBuffer", "src/curobo_metal/ops/world_collision/core.py", "query_esdf", "partial", "Grid sampling exists; upstream float16 storage, cache/update API, boundary behavior, and Warp kernel are not API-compatible."),
    ("cost", "pose_cost", "curobo/_src/cost/cost_tool_pose.py", "ToolPoseCost", "src/curobo_metal/ops/costs/core.py", "pose_cost", "partial", "Core differentiable error/cost exists without upstream configuration, convergence, offset-waypoint, or run_weight semantics."),
    ("cost", "bounds_smoothness_collision", "curobo/_src/cost", None, "src/curobo_metal/ops/costs/core.py", "smoothness_cost", "partial", "Representative composable costs exist; not the full upstream cost class/config surface."),
    ("ik", "inverse_kinematics", "curobo/_src/solver/solver_ik.py", "IKSolver", "src/curobo_metal/ops/ik/solver.py", "solve_ik", "partial", "Deterministic batched solver exists; seed policy, CUDA graphs, result types, retract config, and all upstream options differ."),
    ("trajectory", "trajectory_optimization", "curobo/_src/solver/solver_trajopt.py", "TrajOptSolver", "src/curobo_metal/ops/trajectory/core.py", "optimize_trajectory", "partial", "Portable optimizer covers documented contract, not upstream particle/L-BFGS solver stack or CUDA graphs."),
    ("motion_generation", "motion_gen", "curobo/_src/motion/motion_planner.py", "MotionPlanner", "src/curobo_metal/ops/trajectory/motion_generation.py", "generate_motion", "partial", "Local IK/graph/trajectory composition exists, but upstream API, warmup, graph cache, retries, and result semantics differ."),
    ("graph", "graph_planner", "curobo/_src/graph_planner/graph_planner_prm.py", "PRMGraphPlanner", "src/curobo_metal/ops/graph_planning/core.py", "plan_graph", "partial", "Deterministic geometric planner exists; not upstream CUDA graph planner or its persistent graph representation."),
    ("dynamics", "inverse_dynamics", "curobo/_src/robot/dynamics/dynamics.py", None, "src/curobo_metal/reference/dynamics.py", "inverse_dynamics", "partial", "CPU reference RNEA is not exported as a production Metal op and upstream public parity is not established."),
    ("types", "joint_state_pose_results", "curobo/_src/state/state_joint.py", "JointState", None, None, "unsupported", "No drop-in cuRobo public data-model compatibility layer."),
    ("runtime", "cuda_graphs_and_cuda_kernels", "curobo/_src/curobolib", None, None, None, "not_applicable", "CUDA implementation mechanisms are intentionally not portable to Metal; observable semantics must be compared instead."),
    ("integration", "isaac_ros_usd", "curobo/_src/util/usd_writer.py", "UsdWriter", None, None, "unsupported", "Isaac Sim, ROS, USD, and visualization integrations are outside this repository."),
]


def symbols(path: Path) -> set[str]:
    if not path.is_file() or path.suffix != ".py":
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def line_for(path: Path, symbol: str | None) -> int | None:
    if symbol is None or not path.is_file() or path.suffix != ".py":
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return node.lineno
    return None


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/parity/capabilities.json"))
    args = parser.parse_args()
    upstream, repository = args.upstream.resolve(), args.repository.resolve()
    actual = git(upstream, "rev-parse", "HEAD")
    if actual != PIN:
        raise SystemExit(f"expected upstream {PIN}, found {actual}")

    records = []
    for area, capability, up_file, up_symbol, local_file, local_symbol, status, blocker in CAPABILITIES:
        if status not in CLASSES:
            raise AssertionError(status)
        up_path = upstream / up_file
        local_path = repository / local_file if local_file else None
        if not up_path.exists():
            raise SystemExit(f"missing upstream evidence: {up_file}")
        if up_symbol and up_symbol not in symbols(up_path):
            raise SystemExit(f"missing upstream symbol: {up_file}:{up_symbol}")
        if local_path and not local_path.exists():
            raise SystemExit(f"missing local evidence: {local_file}")
        if local_path and local_symbol and local_symbol not in symbols(local_path):
            raise SystemExit(f"missing local symbol: {local_file}:{local_symbol}")
        records.append({
            "id": f"{area}.{capability}", "area": area, "capability": capability,
            "classification": status, "blocker": blocker,
            "upstream_evidence": {"path": up_file, "symbol": up_symbol, "line": line_for(up_path, up_symbol)},
            "local_evidence": None if not local_file else {"path": local_file, "symbol": local_symbol, "line": line_for(local_path, local_symbol)},
        })
    result = {
        "schema_version": 1,
        "generated_from": {"repository": REPOSITORY, "revision": PIN},
        "classification_vocabulary": list(CLASSES),
        "summary": {c: sum(r["classification"] == c for r in records) for c in CLASSES},
        "capabilities": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

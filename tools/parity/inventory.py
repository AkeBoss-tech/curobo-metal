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
CLASSES = (
    "semantically_equivalent",
    "partial",
    "intentionally_platform_inapplicable",
    "external_integration_only",
    "evidence_blocked",
)

# The inventory is deliberately curated.  AST validation prevents stale source
# citations, but classification remains an audit decision based on the cited
# implementation and tests.  "Semantically equivalent" is always scoped to the
# named capability, never to an entire upstream class or package.
CAPABILITIES = [
    {
        "id": "configuration.robot_config_and_loaders",
        "upstream": ("curobo/_src/types/robot.py", "RobotCfg"),
        "local": ("src/curobo_metal/config/loaders.py", "load_robot_config"),
        "tests": ["tests/compat/types_config/test_config.py"],
        "classification": "partial",
        "boundary": "RobotCfg construction plus YAML, URDF, and XRDF subsets are implemented; USD and the complete mutable upstream configuration graph are not.",
        "implementable_gaps": ["complete supported YAML/XRDF field coverage", "mesh/asset and mimic-joint loader coverage", "upstream-compatible mutation and serialization behavior"],
    },
    {
        "id": "types.device_cfg",
        "upstream": ("curobo/_src/types/device_cfg.py", "DeviceCfg"),
        "local": ("src/curobo_metal/types/device.py", "DeviceCfg"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "partial",
        "boundary": "Device/dtype selection and tensor conversion are compatible for the tested subset; CUDA identifiers and the full upstream helper surface are absent.",
        "implementable_gaps": ["remaining non-CUDA DeviceCfg helpers and exact constructor/default compatibility"],
    },
    {
        "id": "types.pose",
        "upstream": ("curobo/_src/types/pose.py", "Pose"),
        "local": ("src/curobo_metal/types/math.py", "Pose"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "partial",
        "boundary": "Quaternion pose construction, conversion, indexing, and device transfer exist; the upstream algebra and broadcast/helper surface is incomplete.",
        "implementable_gaps": ["pose multiply/inverse/transform helpers", "exact upstream broadcasting and optional-field behavior"],
    },
    {
        "id": "types.joint_state",
        "upstream": ("curobo/_src/state/state_joint.py", "JointState"),
        "local": ("src/curobo_metal/types/state.py", "JointState"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "partial",
        "boundary": "Position/velocity/acceleration/jerk containers and common tensor transforms exist; upstream trajectory manipulation is incomplete.",
        "implementable_gaps": ["remaining trajectory interpolation/integration helpers", "exact upstream indexing, auxiliary-state, and name-reordering behavior"],
    },
    {
        "id": "types.solver_results",
        "upstream": ("curobo/_src/solver/solver_base_result.py", "BaseSolverResult"),
        "local": ("src/curobo_metal/types/result.py", "PlanningResult"),
        "tests": ["tests/compat/api_surface/test_motion_gen.py"],
        "classification": "partial",
        "boundary": "Portable IK, graph, trajectory, and motion result records exist, but they do not expose every upstream metric/debug/timing field or method.",
        "implementable_gaps": ["complete result field/method compatibility", "upstream status and debug-record conversions"],
    },
    {
        "id": "kinematics.forward_kinematics",
        "upstream": ("curobo/_src/robot/kinematics/kinematics.py", "Kinematics"),
        "local": ("src/curobo_metal/ops/kinematics/forward.py", "forward_kinematics"),
        "tests": ["tests/contracts/test_forward_kinematics.py", "tests/ops/kinematics/test_forward.py"],
        "classification": "evidence_blocked",
        "boundary": "Batched link poses and collision spheres are implemented and locally oracle-tested, but no paired run against the pinned NVIDIA implementation is available.",
        "implementable_gaps": [],
        "evidence_needed": "A pinned cuRobo CUDA runner replaying identical chains, batches, dtypes, gradients, and edge cases.",
    },
    {
        "id": "kinematics.geometric_jacobian",
        "upstream": ("curobo/_src/robot/kinematics/kinematics.py", "Kinematics"),
        "local": ("src/curobo_metal/ops/kinematics/jacobian.py", "geometric_jacobian"),
        "tests": ["tests/compat/jacobian_collision_checker/test_jacobian.py"],
        "classification": "partial",
        "boundary": "A batched geometric Jacobian API exists, including link selection; it is not an API-compatible implementation of upstream Kinematics buffers and modes.",
        "implementable_gaps": ["upstream-compatible result/buffer integration", "complete link-selection, fixed-joint, and whole-body mode coverage"],
    },
    {
        "id": "collision.sphere_pair_primitives",
        "upstream": ("curobo/_src/collision/collision_robot_scene.py", "_point_robot_distance"),
        "local": ("src/curobo_metal/ops/collision/core.py", "sphere_sphere_signed_distance"),
        "tests": ["tests/contracts/test_collision.py", "tests/ops/collision/test_core.py"],
        "classification": "semantically_equivalent",
        "boundary": "Signed sphere-pair distance and gradients match the repository's explicit mathematical contract; this does not imply RobotSceneCollision class parity.",
        "implementable_gaps": [],
    },
    {
        "id": "collision.robot_scene",
        "upstream": ("curobo/_src/collision/collision_robot_scene.py", "RobotSceneCollision"),
        "local": ("src/curobo_metal/collision_checker/core.py", "RobotSceneCollision"),
        "tests": ["tests/compat/jacobian_collision_checker/test_collision_checker.py"],
        "classification": "partial",
        "boundary": "Self/world discrete queries, primitive caches, and compatibility configuration exist; upstream pair filtering, all cache/update modes, swept queries, and result shapes are incomplete.",
        "implementable_gaps": ["self-collision pair filtering and aggregate semantics", "complete cache/update/environment API", "swept collision and exact upstream output layout"],
    },
    {
        "id": "collision.cuboid_world",
        "upstream": ("curobo/_src/geom/collision/checker_collision.py", "CollisionChecker"),
        "local": ("src/curobo_metal/ops/collision/core.py", "sphere_cuboid_signed_distance"),
        "tests": ["tests/contracts/test_collision.py", "tests/ops/collision/test_core.py"],
        "classification": "semantically_equivalent",
        "boundary": "The scoped discrete sphere-to-oriented-cuboid signed-distance contract and gradients are oracle-backed; upstream checker/cache/sweep behavior is inventoried separately.",
        "implementable_gaps": [],
    },
    {
        "id": "collision.mesh_world",
        "upstream": ("curobo/_src/geom/collision/buffer_collision.py", "CollisionBuffer"),
        "local": ("src/curobo_metal/ops/world_collision/core.py", "mesh_distance"),
        "tests": ["tests/contracts/test_world_collision.py", "tests/ops/world_collision/test_core.py"],
        "classification": "partial",
        "boundary": "Batched triangle distance, transforms, masks, ties, and gradients exist; signed watertight/BVH/cache/sweep semantics differ.",
        "implementable_gaps": ["signed watertight mesh semantics", "persistent mesh cache/update API", "swept mesh queries and upstream result layout"],
    },
    {
        "id": "collision.voxel_esdf_query",
        "upstream": ("curobo/_src/geom/collision/buffer_collision.py", "CollisionBuffer"),
        "local": ("src/curobo_metal/ops/world_collision/core.py", "query_esdf"),
        "tests": ["tests/contracts/test_world_collision.py", "tests/ops/world_collision/test_core.py"],
        "classification": "partial",
        "boundary": "Differentiable dense-grid ESDF queries exist; upstream storage precision, cache/update behavior, interpolation boundaries, and output layout are not fully reproduced.",
        "implementable_gaps": ["exact boundary/out-of-bounds semantics", "upstream voxel cache/update and result layout", "supported storage-precision compatibility"],
    },
    {
        "id": "cost.pose_and_composable_costs",
        "upstream": ("curobo/_src/cost/cost_tool_pose.py", "ToolPoseCost"),
        "local": ("src/curobo_metal/ops/costs/core.py", "pose_cost"),
        "tests": ["tests/contracts/test_ik.py", "tests/compat/api_surface/test_configs.py"],
        "classification": "partial",
        "boundary": "Pose, bounds, smoothness, and collision-oriented portable costs/config records exist; the full upstream cost-class and rollout contract does not.",
        "implementable_gaps": ["offset-waypoint and run-weight execution semantics", "remaining upstream cost classes and convergence interfaces", "exact batch/result layouts"],
    },
    {
        "id": "optim.particle_evolution",
        "upstream": ("curobo/_src/optim/particle/evolution_strategies.py", "EvolutionStrategies"),
        "local": ("src/curobo_metal/optim/core.py", "particle_optimize"),
        "tests": ["tests/ops/optim/test_optimizers.py"],
        "classification": "partial",
        "boundary": "Deterministic batched elite evolution, bounds, seeds, and execution cache exist; upstream MPPI/ES sampling/config/debug behavior is broader.",
        "implementable_gaps": ["MPPI and remaining sampling strategies", "upstream covariance/noise and debug-record behavior", "config/result adapter completeness"],
    },
    {
        "id": "optim.lbfgs",
        "upstream": ("curobo/_src/optim/gradient/lbfgs.py", "LBFGSOpt"),
        "local": ("src/curobo_metal/optim/core.py", "lbfgs_optimize"),
        "tests": ["tests/ops/optim/test_optimizers.py"],
        "classification": "partial",
        "boundary": "Batched limited-memory BFGS with Armijo search and cache lifecycle exists; upstream line-search strategies, CUDA graph capture, and exact state/config surfaces differ.",
        "implementable_gaps": ["remaining portable line-search strategies and termination modes", "upstream optimizer state/config/result adapters"],
    },
    {
        "id": "ik.inverse_kinematics",
        "upstream": ("curobo/_src/solver/solver_ik.py", "IKSolver"),
        "local": ("src/curobo_metal/ops/ik/solver.py", "solve_ik"),
        "tests": ["tests/contracts/test_ik.py", "tests/ops/ik/test_solver.py"],
        "classification": "partial",
        "boundary": "Deterministic batched seeds, pose residuals, limits, and optimizer selection exist; upstream solve modes, seed manager, result/config surface, and collision integration are incomplete.",
        "implementable_gaps": ["complete solve modes and seed policies", "collision-aware IK integration", "upstream config/result and convergence semantics"],
    },
    {
        "id": "trajectory.trajectory_optimization",
        "upstream": ("curobo/_src/solver/solver_trajopt.py", "TrajOptSolver"),
        "local": ("src/curobo_metal/ops/trajectory/core.py", "optimize_trajectory"),
        "tests": ["tests/contracts/test_trajectory.py", "tests/ops/trajectory/test_trajectory.py"],
        "classification": "partial",
        "boundary": "Portable trajectory optimization now includes particle and L-BFGS stages, limits, costs, and deterministic seeds; upstream rollout, solve modes, interpolation, and result/config semantics remain broader.",
        "implementable_gaps": ["upstream rollout/cost-manager composition", "complete solve modes, interpolation, and seed handling", "upstream config/result adapter completeness"],
    },
    {
        "id": "trajectory.dynamics_aware_bspline",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/ops/trajectory/dynamics_aware.py", "optimize_dynamics_aware"),
        "tests": ["tests/ops/trajectory/test_dynamics_aware.py"],
        "classification": "partial",
        "boundary": "B-spline sampling, velocity/acceleration/jerk/torque costs, optimization, and retiming exist; this is a portable contract rather than the upstream fused rollout API.",
        "implementable_gaps": ["adapter into upstream-style trajectory solver configuration/results", "full collision and task-space rollout composition"],
    },
    {
        "id": "graph.prm_planner",
        "upstream": ("curobo/_src/graph_planner/graph_planner_prm.py", "PRMGraphPlanner"),
        "local": ("src/curobo_metal/ops/graph_planning/core.py", "plan_graph"),
        "tests": ["tests/contracts/test_graph_planning.py", "tests/ops/graph_planning/test_graph_planning.py"],
        "classification": "partial",
        "boundary": "Deterministic sampled graph planning and path shortcutting exist; persistent graph mutation, all sampling/search modes, and upstream result/config behavior do not.",
        "implementable_gaps": ["persistent graph lifecycle and update API", "remaining sampling/search/connection modes", "upstream config/result adapters"],
    },
    {
        "id": "motion_generation.motion_gen",
        "upstream": ("curobo/_src/motion/motion_planner.py", "MotionPlanner"),
        "local": ("src/curobo_metal/api_compat/motion_gen.py", "MotionGen"),
        "tests": ["tests/compat/api_surface/test_motion_gen.py", "tests/integration/motion_gen/test_motion_gen.py"],
        "classification": "partial",
        "boundary": "Single/batch joint and pose planning, warmup/reset facade, world update, and composed IK/graph/trajectory execution exist; many upstream planning modes and operational semantics remain absent.",
        "implementable_gaps": ["complete pose/batch solve modes and retry policies", "world/attachment mutation semantics", "interpolation, timing, metrics, and debug-result compatibility"],
    },
    {
        "id": "dynamics.inverse_dynamics",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/reference/dynamics.py", "inverse_dynamics"),
        "tests": ["tests/contracts/test_whole_body_dynamics.py"],
        "classification": "evidence_blocked",
        "boundary": "A CPU reference RNEA is contract-tested, but it is not a production Metal operation and has not been replayed against pinned upstream CUDA dynamics.",
        "implementable_gaps": ["production CPU/MPS dynamics API and integration with RobotCfg"],
        "evidence_needed": "A pinned cuRobo CUDA dynamics replay plus a production local backend implementation.",
    },
    {
        "id": "whole_body.kinematics_dynamics",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/ops/whole_body/core.py", "WholeBodyModel"),
        "tests": ["tests/contracts/test_whole_body_dynamics.py", "tests/ops/whole_body/test_whole_body.py"],
        "classification": "partial",
        "boundary": "Tree kinematics, Jacobians, mass matrix, bias/gravity, and inverse dynamics are exposed through a portable whole-body contract; upstream robot/rollout APIs and paired numerical evidence are incomplete.",
        "implementable_gaps": ["upstream-compatible robot state/config/result adapters", "remaining dynamics quantities and rollout integration"],
    },
    {
        "id": "perception.depth_esdf_mapping",
        "upstream": ("curobo/_src/perception/mapper/mapper.py", "Mapper"),
        "local": ("src/curobo_metal/ops/perception/core.py", "PerceptionMapper"),
        "tests": ["tests/contracts/perception/test_perception_esdf.py", "tests/ops/perception/test_core.py"],
        "classification": "partial",
        "boundary": "Depth integration, deterministic dense ESDF, batched environments, reset, voxel export, and collision query exist; upstream block-hashed TSDF/ESDF mapping is substantially broader.",
        "implementable_gaps": ["sparse block allocation and TSDF integration", "mesh extraction, rendering, pose refinement, and checkpointing", "upstream camera/mapper configuration adapters"],
    },
    {
        "id": "runtime.cuda_graphs_and_fused_cuda_kernels",
        "upstream": ("curobo/_src/util/cuda_graph_util.py", "GraphExecutor"),
        "local": None,
        "tests": [],
        "classification": "intentionally_platform_inapplicable",
        "boundary": "CUDA graphs, CUDA streams, NVRTC extensions, and Warp/CUDA kernels are NVIDIA execution mechanisms, not portable Metal capabilities; observable behavior remains in scope.",
        "implementable_gaps": [],
    },
    {
        "id": "integration.isaac_ros_usd_visualization",
        "upstream": ("curobo/_src/util/usd_writer.py", "UsdWriter"),
        "local": None,
        "tests": [],
        "classification": "external_integration_only",
        "boundary": "Isaac Sim/Omniverse, ROS, USD authoring, and viewer integrations depend on external ecosystems and are outside this core package.",
        "implementable_gaps": [],
    },
]


def symbols(path: Path) -> set[str]:
    if not path.is_file() or path.suffix != ".py":
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


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


def checked_evidence(root: Path, evidence: tuple[str, str], label: str) -> dict[str, object]:
    relative, symbol = evidence
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing {label} evidence: {relative}")
    if symbol not in symbols(path):
        raise SystemExit(f"missing {label} symbol: {relative}:{symbol}")
    return {"path": relative, "symbol": symbol, "line": line_for(path, symbol)}


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
    for capability in CAPABILITIES:
        classification = capability["classification"]
        if classification not in CLASSES:
            raise AssertionError(classification)
        test_evidence = []
        for relative in capability["tests"]:
            path = repository / relative
            if not path.is_file():
                raise SystemExit(f"missing test evidence: {relative}")
            test_evidence.append(relative)
        local = capability["local"]
        records.append(
            {
                "id": capability["id"],
                "area": capability["id"].split(".", 1)[0],
                "capability": capability["id"].split(".", 1)[1],
                "classification": classification,
                "boundary": capability["boundary"],
                "implementable_gaps": capability["implementable_gaps"],
                "evidence_needed": capability.get("evidence_needed"),
                "upstream_evidence": checked_evidence(upstream, capability["upstream"], "upstream"),
                "local_evidence": None if local is None else checked_evidence(repository, local, "local"),
                "test_evidence": test_evidence,
            }
        )

    result = {
        "schema_version": 2,
        "generated_from": {"repository": REPOSITORY, "revision": PIN},
        "audit_scope": "Pinned upstream API/capability surface versus production implementations at local HEAD; classifications do not assert package-wide drop-in parity.",
        "classification_vocabulary": list(CLASSES),
        "summary": {name: sum(r["classification"] == name for r in records) for name in CLASSES},
        "capabilities": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

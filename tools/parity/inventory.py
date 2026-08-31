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
PAIRED_REPORT = Path("artifacts/parity/cuda-replay/paired-report-2026-08-26-matrix.json")
PAIRED_CUDA_ROOT = Path("artifacts/parity/cuda-replay/2026-08-26")

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
        "classification": "evidence_blocked",
        "boundary": "The portable YAML/URDF/XRDF field set, relative asset resolution, mimic semantics, mutation, serialization, and model compilation are locally covered; USD/Isaac remains platform-inapplicable and exact upstream behavior is not asserted.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream configuration corpus replay covering supported YAML, URDF, XRDF, asset, mimic, mutation, and serialization cases.",
    },
    {
        "id": "types.device_cfg",
        "upstream": ("curobo/_src/types/device_cfg.py", "DeviceCfg"),
        "local": ("src/curobo_metal/types/device.py", "DeviceCfg"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "evidence_blocked",
        "boundary": "Portable CPU/MPS device and dtype selection, floating/integer/boolean conversion, cloning, CPU transfer, and device comparison are locally covered; CUDA identifiers are platform-inapplicable.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream replay of the non-CUDA DeviceCfg constructor and helper surface.",
    },
    {
        "id": "types.pose",
        "upstream": ("curobo/_src/types/pose.py", "Pose"),
        "local": ("src/curobo_metal/types/math.py", "Pose"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "evidence_blocked",
        "boundary": "Quaternion pose construction, matrix/vector conversion, indexing, device transfer, composition, inverse, and point transforms are differentiable and locally covered.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream replay of pose broadcasting, optional fields, algebra, and gradients.",
    },
    {
        "id": "types.joint_state",
        "upstream": ("curobo/_src/state/state_joint.py", "JointState"),
        "local": ("src/curobo_metal/types/state.py", "JointState"),
        "tests": ["tests/compat/types_config/test_types.py"],
        "classification": "evidence_blocked",
        "boundary": "Position/velocity/acceleration/jerk containers, tensor transforms, metadata-preserving indexing, name reordering, finite differences, and integration are locally covered.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream replay of trajectory indexing, auxiliary state, derivative, integration, and joint-name cases.",
    },
    {
        "id": "types.solver_results",
        "upstream": ("curobo/_src/solver/solver_base_result.py", "BaseSolverResult"),
        "local": ("src/curobo_metal/types/result.py", "PlanningResult"),
        "tests": ["tests/compat/api_surface/test_motion_gen.py"],
        "classification": "evidence_blocked",
        "boundary": "Portable IK, graph, trajectory, and motion result/status adapters expose solution, timing, metric, debug, plan, and device-transfer records used by supported workflows.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream replay of supported solver result construction, status conversion, timing, debug, metrics, and plan access.",
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
        "classification": "evidence_blocked",
        "boundary": "Differentiable batched geometric Jacobians cover named/indexed links, fixed joints, serial/tree whole-body modes, end effectors, result aliases, and caller-owned output buffers.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned CUDA replay of matching serial/tree models, selected links, output buffers, dtypes, and gradients.",
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
        "classification": "evidence_blocked",
        "boundary": "Portable self/world queries cover explicit pair/active filtering, max/sum aggregation, fixed-capacity multi-environment mutation, result records, and differentiable swept interpolation.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream RobotSceneCollision replay for supported pair, aggregation, cache mutation, environment routing, result-layout, and swept-query cases.",
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
        "classification": "evidence_blocked",
        "boundary": "Portable mesh queries cover batched triangle distance, watertight signing, transforms, masks, ties, gradients, persistent cache mutation, result records, and swept sphere sampling without a BVH.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream mesh replay across watertight boundaries, cache updates, result layouts, and swept paths.",
    },
    {
        "id": "collision.voxel_esdf_query",
        "upstream": ("curobo/_src/geom/collision/buffer_collision.py", "CollisionBuffer"),
        "local": ("src/curobo_metal/ops/world_collision/core.py", "query_esdf"),
        "tests": ["tests/contracts/test_world_collision.py", "tests/ops/world_collision/test_core.py"],
        "classification": "evidence_blocked",
        "boundary": "Differentiable dense-grid ESDF queries cover trilinear boundaries, explicit out-of-bounds values, multi-environment routing, cache mutation, supported storage dtypes, and structured results.",
        "implementable_gaps": [],
        "evidence_needed": "Pinned upstream voxel replay across interpolation boundaries, out-of-bounds points, storage dtypes, cache updates, and result layouts.",
    },
    {
        "id": "cost.pose_and_composable_costs",
        "upstream": ("curobo/_src/cost/cost_tool_pose.py", "ToolPoseCost"),
        "local": ("src/curobo_metal/ops/costs/core.py", "pose_cost"),
        "tests": ["tests/contracts/test_ik.py", "tests/compat/api_surface/test_configs.py", "tests/ops/costs/test_composition.py"],
        "classification": "evidence_blocked",
        "boundary": "Portable pose, waypoint/offset, run-weight, bounds, smoothness, collision, and named cost-manager composition are implemented; equivalence to pinned fused CUDA rollout costs requires replay.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported portable cost terms, batches, gradients, and convergence boundaries.",
    },
    {
        "id": "optim.particle_evolution",
        "upstream": ("curobo/_src/optim/particle/evolution_strategies.py", "EvolutionStrategies"),
        "local": ("src/curobo_metal/optim/core.py", "particle_optimize"),
        "tests": ["tests/ops/optim/test_optimizers.py"],
        "classification": "evidence_blocked",
        "boundary": "Deterministic CEM, MPPI, and random sampling, covariance/noise control, per-seed results, debug histories, and execution state are portable; CUDA numerical equivalence is unverified.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for sampling streams, covariance, termination, result layouts, and debug traces.",
    },
    {
        "id": "optim.lbfgs",
        "upstream": ("curobo/_src/optim/gradient/lbfgs.py", "LBFGSOpt"),
        "local": ("src/curobo_metal/optim/core.py", "lbfgs_optimize"),
        "tests": ["tests/ops/optim/test_optimizers.py"],
        "classification": "evidence_blocked",
        "boundary": "Batched L-BFGS exposes fixed-step, Armijo, and strong-Wolfe searches, selectable termination, gradient/result/debug state, and cache lifecycle; CUDA mechanisms remain out of scope.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported line searches, termination boundaries, and optimizer state/result behavior.",
    },
    {
        "id": "ik.inverse_kinematics",
        "upstream": ("curobo/_src/solver/solver_ik.py", "IKSolver"),
        "local": ("src/curobo_metal/ops/ik/solver.py", "solve_ik"),
        "tests": ["tests/contracts/test_ik.py", "tests/ops/ik/test_solver.py"],
        "classification": "evidence_blocked",
        "boundary": "Single/batch goal and seed modes, deterministic random/retract seeds, optimizer selection, primitive collision integration, convergence statuses, and selected-seed results are implemented.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported solve modes, seeds, collision boundaries, gradients, and result selection.",
    },
    {
        "id": "trajectory.trajectory_optimization",
        "upstream": ("curobo/_src/solver/solver_trajopt.py", "TrajOptSolver"),
        "local": ("src/curobo_metal/ops/trajectory/core.py", "optimize_trajectory"),
        "tests": ["tests/contracts/test_trajectory.py", "tests/ops/trajectory/test_trajectory.py"],
        "classification": "evidence_blocked",
        "boundary": "Portable single/batch and multiseed trajectory optimization composes rollout costs, Adam/particle/L-BFGS modes, linear/cubic/B-spline interpolation, metrics, and selected-seed results.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported rollout costs, solve modes, interpolation, gradients, and result layouts.",
    },
    {
        "id": "trajectory.dynamics_aware_bspline",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/ops/trajectory/dynamics_aware.py", "optimize_dynamics_aware"),
        "tests": ["tests/ops/trajectory/test_dynamics_aware.py"],
        "classification": "evidence_blocked",
        "boundary": "B-spline sampling, dynamics/task-space/collision rollout composition, optimization, result adaptation, and retiming form a complete portable contract; fused CUDA equivalence is unverified.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay of supported dynamics-aware rollout, optimization, retiming, and result behavior.",
    },
    {
        "id": "graph.prm_planner",
        "upstream": ("curobo/_src/graph_planner/graph_planner_prm.py", "PRMGraphPlanner"),
        "local": ("src/curobo_metal/ops/graph_planning/core.py", "plan_graph"),
        "tests": ["tests/contracts/test_graph_planning.py", "tests/ops/graph_planning/test_graph_planning.py"],
        "classification": "evidence_blocked",
        "boundary": "Deterministic uniform/Sobol sampling, k-neighbor/radius/hybrid connection, A*/Dijkstra/greedy search, persistent lifecycle, path metrics, and trajectory seed handoff are implemented.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported sampling, connection/search modes, lifecycle, paths, and metrics.",
    },
    {
        "id": "motion_generation.motion_gen",
        "upstream": ("curobo/_src/motion/motion_planner.py", "MotionPlanner"),
        "local": ("src/curobo_metal/api_compat/motion_gen.py", "MotionGen"),
        "tests": ["tests/compat/api_surface/test_motion_gen.py", "tests/integration/motion_gen/test_motion_gen.py"],
        "classification": "evidence_blocked",
        "boundary": "Portable single/batch joint and pose modes include retries, graph scheduling, retract/multiseed solve configuration, primitive world/attachment mutation, interpolation/retiming, timing, metrics, and debug results.",
        "implementable_gaps": [],
        "evidence_needed": "Paired pinned cuRobo CUDA replay for supported MotionGen solve modes, mutation, interpolation, timing, metrics, statuses, and debug results.",
    },
    {
        "id": "dynamics.inverse_dynamics",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/ops/whole_body/core.py", "inverse_dynamics"),
        "tests": ["tests/contracts/test_whole_body_dynamics.py", "tests/ops/whole_body/test_whole_body.py"],
        "classification": "evidence_blocked",
        "boundary": "Differentiable inverse and forward dynamics are production CPU/MPS operations integrated with RobotCfg, named joint-state adapters, and rollout; equivalence is withheld only pending pinned NVIDIA replay.",
        "implementable_gaps": [],
        "evidence_needed": "A pinned cuRobo CUDA dynamics replay over identical inertial models, batches, dtypes, outputs, and gradients.",
    },
    {
        "id": "whole_body.kinematics_dynamics",
        "upstream": ("curobo/_src/robot/dynamics/dynamics.py", "Dynamics"),
        "local": ("src/curobo_metal/ops/whole_body/core.py", "WholeBodyModel"),
        "tests": ["tests/contracts/test_whole_body_dynamics.py", "tests/ops/whole_body/test_whole_body.py"],
        "classification": "semantically_equivalent",
        "boundary": "The portable whole-body contract provides tree kinematics, Jacobians, inverse/forward dynamics, mass/bias/gravity quantities, named state adapters, and differentiable deterministic rollout on CPU/MPS.",
        "implementable_gaps": [],
    },
    {
        "id": "perception.depth_esdf_mapping",
        "upstream": ("curobo/_src/perception/mapper/mapper.py", "Mapper"),
        "local": ("src/curobo_metal/ops/perception/core.py", "PerceptionMapper"),
        "tests": ["tests/contracts/perception/test_perception_esdf.py", "tests/ops/perception/test_core.py"],
        "classification": "semantically_equivalent",
        "boundary": "The portable mapping contract provides deterministic dense and sparse-block TSDF, ESDF, mesh extraction, depth rendering, pose refinement, checkpointing, batched lifecycle operations, and mapper/camera adapters.",
        "implementable_gaps": [],
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


def checked_paired_capabilities(repository: Path) -> set[str]:
    """Return capabilities certified by the committed full-matrix CUDA replay."""
    report_path = repository / PAIRED_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != "curobo-metal-paired-report"
        or report.get("upstream_revision") != PIN
        or report.get("passed") is not True
        or report.get("errors") != []
    ):
        raise SystemExit(f"paired evidence is not promotable: {PAIRED_REPORT}")
    required = set(report.get("required_capabilities", ()))
    passed = {
        row["capability"]
        for row in report.get("results", ())
        if row.get("passed") is True
    }
    if not required or passed != required:
        raise SystemExit("paired report does not pass every required capability")
    for capability in sorted(required):
        metal = json.loads(
            (repository / "artifacts/parity/replay" / capability / "metal-manifest.json").read_text()
        )
        cuda = json.loads(
            (repository / PAIRED_CUDA_ROOT / capability / "cuda-manifest.json").read_text()
        )
        if metal["input"]["sha256"] != cuda["input_sha256"]:
            raise SystemExit(f"paired input provenance mismatch: {capability}")
        for backend, manifest in (("metal", metal), ("cuda", cuda)):
            evidence = manifest.get("evidence", {})
            if not all(evidence.get(kind, {}).get("executed") is True for kind in ("invalid", "edge", "matrix")):
                raise SystemExit(f"incomplete {backend} evidence matrix: {capability}")
    return required


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

    paired_capabilities = checked_paired_capabilities(repository)
    records = []
    for capability in CAPABILITIES:
        classification = capability["classification"]
        paired_evidence = None
        boundary = capability["boundary"]
        evidence_needed = capability.get("evidence_needed")
        if capability["id"] in paired_capabilities:
            classification = "semantically_equivalent"
            evidence_needed = None
            paired_evidence = str(PAIRED_REPORT)
            boundary = (
                boundary
                + " Fresh paired Metal/CUDA replay against the pinned revision passed its "
                  "invalid, edge, and multi-case matrix on 2026-08-26."
            )
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
                "boundary": boundary,
                "implementable_gaps": capability["implementable_gaps"],
                "evidence_needed": evidence_needed,
                "paired_evidence": paired_evidence,
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

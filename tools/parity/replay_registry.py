"""Authoritative Wave 10 replay case registry.

The registry intentionally distinguishes a portable local probe from an
upstream equivalence claim.  ``cuda_adapter`` is ``None`` until a clean,
asset-complete adapter for the pinned upstream surface is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


PIN = "8e734f3ced1df898990bcd92de40abce475907db"


@dataclass(frozen=True)
class Case:
    capability: str
    operation: str
    rtol: float
    atol: float
    probe: str
    invalid_case: str
    edge_case: str
    cuda_constraint: str
    matrix_cases: tuple[str, ...] = ()


_ROWS = [
    ("configuration.robot_config_and_loaders", "robot_config", 0, 0, "serialization", "malformed_urdf", "two_joint_urdf", "real license-clean serialized URDF RobotCfg CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.device_cfg", "device_cfg", 0, 0, "device", "unsupported_device", "empty_tensor", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.pose", "pose", 1e-6, 1e-7, "pose", "zero_quaternion", "empty_points", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.joint_state", "joint_state", 1e-6, 1e-7, "joint_state", "name_width_mismatch", "noncontiguous_state", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.solver_results", "solver_results", 0, 0, "result", "invalid_status", "mixed_status", "real asset-independent BaseSolverResult CUDA adapter available; execution requires an NVIDIA runtime"),
    ("kinematics.forward_kinematics", "forward_kinematics", 8e-5, 8e-5, "fk", "wrong_dof", "empty_batch", "real compiled CUDA FK adapter available for the license-clean serialized robot"),
    ("kinematics.geometric_jacobian", "geometric_jacobian", 1e-4, 1e-5, "fk", "invalid_link", "noncontiguous_batch", "real compiled CUDA geometric-Jacobian adapter available for the license-clean serialized robot"),
    ("collision.robot_scene", "robot_scene_collision", 2e-5, 2e-6, "sphere", "bad_pair_index", "tangent_spheres", "real asset-independent CUDA self-collision adapter available; execution requires an NVIDIA runtime"),
    ("collision.mesh_world", "mesh_world", 2e-5, 2e-6, "mesh", "non_watertight_signed", "triangle_face", "real pinned Warp CUDA mesh query adapter available for the serialized unsigned triangle corpus"),
    ("collision.voxel_esdf_query", "voxel_esdf", 3e-5, 3e-6, "voxel", "bad_environment", "grid_boundary", "real pinned V2 scene-level CUDA voxel cache and Warp ESDF adapter available for the serialized identity-grid corpus"),
    ("cost.pose_and_composable_costs", "pose_costs", 2e-5, 2e-6, "cost", "invalid_pose_cost_input", "zero_pose_error", "real asset-independent upstream ToolPoseCost CUDA adapter available for position cost and gradient replay"),
    ("optim.particle_evolution", "evolution_strategies", 0, 0, "particle", "zero_problem_count", "facade_lifecycle", "real pinned EvolutionStrategies CUDA adapter; cross-device random samples and exact solutions are intentionally not compared"),
    ("optim.lbfgs", "lbfgs", 2e-4, 2e-5, "lbfgs", "unstable_mode_disabled", "action_range_batch_solve", "pinned upstream LBFGSOpt adapter uses eager PyTorch line search and two-loop recursion on CUDA; terminal freezing and hard projection are unsupported upstream"),
    ("ik.inverse_kinematics", "inverse_kinematics", 3e-4, 3e-5, "cost", "infeasible_goal", "two_pose_batch", "real pinned V2 CUDA IKSolver adapter available for the packaged Franka single-pose outcome/layout corpus; redundant-joint values are intentionally not compared"),
    ("trajectory.trajectory_optimization", "trajectory_optimization", 3e-4, 3e-5, "trajectory", "infeasible_limits", "exact_endpoints", "real pinned V2 CUDA TrajOptSolver adapter available for packaged Franka C-space outcome/layout replay; optimizer trajectories are intentionally not compared"),
    ("trajectory.dynamics_aware_bspline", "dynamics_aware_bspline", 3e-4, 3e-5, "bspline", "invalid_horizon", "endpoint_constraints", "real pinned CUDA cubic B-spline boundary-kernel adapter available"),
    ("graph.prm_planner", "prm_planner", 0, 0, "graph", "blocked_endpoints", "zero_length_edge", "real pinned CUDA PRMGraphPlanner adapter available for declarative 2-DoF forbidden-box outcome replay"),
    ("motion_generation.motion_gen", "motion_planner_cspace", 5e-4, 5e-5, "motion_planner", "zero_attempt_budget", "single_attempt_cspace_lifecycle", "real pinned MotionPlanner CUDA adapter available for packaged Franka C-space outcome/layout replay; planner trajectories are intentionally compared semantically"),
    ("dynamics.inverse_dynamics", "inverse_dynamics", 2e-4, 2e-5, "dynamics", "missing_acceleration", "two_batch_gradients", "real native-CUDA RNEA adapter available for the license-clean serialized inertial robot"),
]

_MATRIX_CASES = {
    "configuration.robot_config_and_loaders": ("single_tree", "branched_tree", "serialization_roundtrip"),
    "types.device_cfg": ("zero", "one", "many", "empty", "device_roundtrip"),
    "types.pose": ("zero_points", "one_point", "many_points", "noncontiguous_points", "vjp"),
    "types.joint_state": ("zero", "one", "many", "noncontiguous", "optional_channels", "mutation", "vjp"),
    "types.solver_results": ("zero", "one", "many", "mixed_status", "clone_mutation"),
    "kinematics.forward_kinematics": ("zero", "one", "many", "noncontiguous", "tree_links", "vjp"),
    "kinematics.geometric_jacobian": ("zero", "one", "many", "noncontiguous", "tree_links", "vjp"),
    "collision.robot_scene": ("separation_sign", "tie_winner", "active_pair", "reduced_aggregation", "vjp"),
    "collision.mesh_world": ("signed_distance", "tie_winner", "active_mesh", "multi_environment", "vjp"),
    "collision.voxel_esdf_query": ("in_bounds", "out_of_bounds", "tie_winner", "active_grid", "multi_environment", "cache_repeat", "vjp"),
    "cost.pose_and_composable_costs": ("zero_error", "weighted_pose", "composed_aggregation", "noncontiguous_error", "vjp"),
    "dynamics.inverse_dynamics": ("zero_motion", "batched_layout", "noncontiguous_state", "result_isolation", "vjp"),
    "optim.particle_evolution": ("five_seed_band", "fixed_seed_repeat", "shift_lifecycle", "debug_trace", "termination", "alternate_config"),
    "optim.lbfgs": ("bounded_config", "alternate_config", "reset_lifecycle", "debug_trace", "nonfinite_termination", "vjp"),
    "graph.prm_planner": ("direct", "detour", "infeasible", "batched", "reset", "debug_layout"),
    "trajectory.trajectory_optimization": ("endpoint", "infeasible", "alternate_config", "result_layout"),
    "trajectory.dynamics_aware_bspline": ("endpoint", "two_horizons", "constraint", "invalid_termination", "vjp"),
    "ik.inverse_kinematics": ("singleton", "batch", "success", "infeasible", "pose_cspace", "multiseed_selection", "status_layout", "residual_vjp"),
    "motion_generation.motion_gen": ("cspace", "success", "infeasible", "multiseed", "retry_schedule", "result_layout", "endpoints_metrics", "residual_vjp"),
}

CASES = tuple(Case(*row, _MATRIX_CASES.get(row[0], ())) for row in _ROWS)
BY_ID = {case.capability: case for case in CASES}

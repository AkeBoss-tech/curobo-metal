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
    cuda_constraint: str


_ROWS = [
    ("configuration.robot_config_and_loaders", "robot_config", 0, 0, "serialization", "malformed_yaml", "requires caller-supplied upstream robot assets and a supported RobotCfg constructor adapter"),
    ("types.device_cfg", "device_cfg", 0, 0, "device", "unsupported_device", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.pose", "pose", 1e-6, 1e-7, "pose", "zero_quaternion", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.joint_state", "joint_state", 1e-6, 1e-7, "joint_state", "name_width_mismatch", "real asset-independent CUDA adapter available; execution requires an NVIDIA runtime"),
    ("types.solver_results", "solver_results", 0, 0, "result", "invalid_status", "result constructors are solver-specific and require upstream solver-owned objects"),
    ("kinematics.forward_kinematics", "forward_kinematics", 8e-5, 8e-5, "fk", "wrong_dof", "requires a compiled upstream CUDA robot model and licensed/caller-provided robot configuration"),
    ("kinematics.geometric_jacobian", "geometric_jacobian", 1e-4, 1e-5, "fk", "invalid_link", "requires a compiled upstream CUDA robot model and caller-provided robot configuration"),
    ("collision.robot_scene", "robot_scene_collision", 2e-5, 2e-6, "sphere", "bad_pair_index", "requires upstream CUDA collision kernels plus robot/world cache configuration"),
    ("collision.mesh_world", "mesh_world", 2e-5, 2e-6, "mesh", "non_watertight_signed", "requires upstream Warp/CUDA mesh acceleration structures; no portable clean-runner constructor"),
    ("collision.voxel_esdf_query", "voxel_esdf", 3e-5, 3e-6, "voxel", "bad_environment", "requires upstream CUDA voxel cache allocation and collision buffers"),
    ("cost.pose_and_composable_costs", "pose_costs", 2e-5, 2e-6, "cost", "bad_weight_shape", "requires upstream fused CUDA rollout buffers and solver-owned cost configuration"),
    ("optim.particle_evolution", "particle_evolution", 2e-4, 2e-5, "particle", "bad_covariance", "upstream RNG stream and optimizer construction require CUDA rollout objects"),
    ("optim.lbfgs", "lbfgs", 2e-4, 2e-5, "lbfgs", "nonfinite_objective", "upstream LBFGSOpt requires CUDA graph/rollout configuration"),
    ("ik.inverse_kinematics", "inverse_kinematics", 3e-4, 3e-5, "cost", "infeasible_goal", "requires caller-provided robot/world assets and upstream CUDA IK solver compilation"),
    ("trajectory.trajectory_optimization", "trajectory_optimization", 3e-4, 3e-5, "trajectory", "infeasible_limits", "requires caller-provided robot/world assets and upstream CUDA rollout compilation"),
    ("trajectory.dynamics_aware_bspline", "dynamics_aware_bspline", 3e-4, 3e-5, "bspline", "too_few_knots", "requires upstream robot dynamics model and fused CUDA rollout"),
    ("graph.prm_planner", "prm_planner", 0, 0, "graph", "blocked_endpoints", "requires upstream CUDA collision checker and graph buffers"),
    ("motion_generation.motion_gen", "motion_gen", 5e-4, 5e-5, "trajectory", "ik_failed", "requires caller-provided robot/world assets and compiled upstream CUDA IK/graph/trajopt stack"),
    ("dynamics.inverse_dynamics", "inverse_dynamics", 2e-4, 2e-5, "dynamics", "bad_inertia", "requires a caller-provided inertial robot model compiled by upstream CUDA dynamics"),
]

CASES = tuple(Case(*row) for row in _ROWS)
BY_ID = {case.capability: case for case in CASES}

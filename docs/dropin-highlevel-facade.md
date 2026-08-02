# Public high-level facade closure

The pinned public modules are intentionally re-export-only modules.  This wave
keeps their object identities intact while installing the public calling
contracts that differ from the portable backend's extended private APIs.

`TrajectoryOptimizer.solve_pose(...)` now follows the pinned public signature
and composes portable IK with production joint-space trajectory optimization.
It accepts a `goal_state` for an explicit joint-space bypass and retains the IK
result in `result.debug_info["ik_result"]`.  Multi-link/goalset pose inputs are
still bounded by the portable IK implementation and reject precisely.

`RobotCollisionChecker` exposes the pinned `n`, `q`, and `batch` argument names
for sampling/validation while preserving device-resident CPU/MPS collision
queries, derivatives, and the private backend's extended helpers.  `Kinematics`
accepts the pinned `get_robot_as_mesh(joint_position)` call shape; mesh asset
construction remains an explicit CUDA/Isaac/asset boundary and raises
`NotImplementedError` rather than a misleading arity error.

This is source/API-shape closure, not an NVIDIA equivalence claim.  CUDA graph
capture, Warp/BVH mesh construction, Isaac/USD scene services, and paired CUDA
numerical replay remain outside the portable supported surface.

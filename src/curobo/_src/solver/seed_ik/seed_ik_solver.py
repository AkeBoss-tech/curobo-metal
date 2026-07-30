"""Seeded IK facade backed by the portable production IKSolver."""

from __future__ import annotations

from typing import Optional

from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from .seed_ik_solver_cfg import SeedIKSolverCfg


class SeedIKSolver:
    def __init__(self, config: SeedIKSolverCfg):
        if not isinstance(config, SeedIKSolverCfg):
            raise TypeError("config must be SeedIKSolverCfg")
        self.config = config
        self._solver = IKSolver(IKSolverCfg.create(
            config.robot_config, device_cfg=config.device_cfg,
            num_seeds=config.num_seeds,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=False, random_seed=config.sampler_seed,
            success_requires_convergence=True,
        ))

    @property
    def n_residuals(self): return self._solver.action_dim + 6 * len(self._solver.tool_frames)

    def _solve_impl(
        self, goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config=None, return_seeds: int = 1, batch_size: int = 1,
    ):
        if batch_size != goal_tool_poses.batch_size:
            raise ValueError("batch_size must match goal_tool_poses")
        return self._solver.solve_pose(
            goal_tool_poses, current_state=current_state,
            seed_config=seed_config, return_seeds=return_seeds,
        )

    def solve_single(self, goal_tool_poses, current_state=None, seed_config=None, return_seeds=1):
        if goal_tool_poses.batch_size != 1:
            raise ValueError("solve_single requires batch size 1")
        return self._solve_impl(
            goal_tool_poses, current_state, seed_config, return_seeds, 1
        )

    def solve_batch(self, goal_tool_poses, current_state=None, seed_config=None, return_seeds=1):
        return self._solve_impl(
            goal_tool_poses, current_state, seed_config, return_seeds,
            goal_tool_poses.batch_size,
        )

    @property
    def joint_limits(self): return self._solver._joint_limits()
    def get_default_joint_position(self): return self._solver.default_joint_position
    @property
    def kinematics(self): return self._solver.kinematics
    def compute_kinematics(self, joint_position):
        state = joint_position if isinstance(joint_position, JointState) else JointState.from_position(
            joint_position, self._solver.joint_names
        )
        return self._solver.compute_kinematics(state)
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self._solver.config.tool_pose_criteria = dict(tool_pose_criteria)
    def reset_seed(self): return self._solver.reset_seed()
    def destroy(self): return self._solver.destroy()


__all__ = ["SeedIKSolver"]

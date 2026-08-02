"""Public trajectory optimization facade.

The pinned public module is intentionally a small re-export layer.  The
portable backend retains that identity so applications which import both the
public and internal spelling receive the same solver class.  At import time we
install the public ``solve_pose`` contract on that class: the Metal/CPU
implementation composes its differentiable IK solver with the existing
joint-space trajectory optimizer rather than requiring callers to manufacture
an intermediate joint goal themselves.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Optional

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose

from curobo._src.solver.solver_trajopt import TrajOptSolver as TrajectoryOptimizer
from curobo._src.solver.solver_trajopt_cfg import (
    TrajOptSolverCfg as TrajectoryOptimizerCfg,
)
from curobo._src.solver.solver_trajopt_result import (
    TrajOptSolverResult as TrajectoryOptimizerResult,
)


def _solve_pose(
    self: TrajectoryOptimizer,
    goal_tool_poses: GoalToolPose,
    current_state: JointState,
    seed_config=None,
    seed_traj=None,
    return_seeds: int = 1,
    num_seeds: Optional[int] = None,
    dt=None,
    use_implicit_goal: bool = False,
    finetune_attempts: int = 1,
    goal_state: Optional[JointState] = None,
    initial_iters: Optional[int] = None,
    time_optimal_iters: Optional[int] = None,
    finetune_iters: Optional[int] = None,
    finetune_dt_scale: float = 0.55,
) -> TrajectoryOptimizerResult:
    """Plan to a tool pose using portable IK followed by trajectory optimization.

    This has the pinned cuRoboV2 public signature.  The CUDA implementation
    performs the two stages through compiled rollout objects; the portable
    implementation preserves the externally observable composition with the
    production PyTorch IK and trajectory paths.  A supplied ``goal_state``
    remains an efficient explicit joint-space bypass.
    """
    if not isinstance(current_state, JointState):
        raise TypeError("current_state must be a JointState")

    if goal_state is None:
        # ``acos`` has an undefined derivative at an exactly matching
        # quaternion.  Apple's MPS autograd exposes that singularity as NaNs
        # during an otherwise trivial hold-pose request.  Recognize that common
        # high-level command before entering IK and pass it directly to the
        # differentiable joint-space planner.
        hold_pose = False
        if (
            self.device_cfg.device.type == "mps"
            and isinstance(goal_tool_poses, GoalToolPose)
            and goal_tool_poses.num_links == 1
            and goal_tool_poses.num_goalset == 1
        ):
            current_pose = self.compute_kinematics(current_state).tool_poses
            target_position = goal_tool_poses.position[:, 0, 0, 0]
            target_quaternion = goal_tool_poses.quaternion[:, 0, 0, 0]
            actual_position = current_pose.position[:, 0, 0]
            actual_quaternion = current_pose.quaternion[:, 0, 0]
            position_error = (actual_position - target_position).abs().amax()
            alignment = (actual_quaternion * target_quaternion).sum(-1).abs()
            hold_pose = bool(
                (position_error <= self.config.position_tolerance).item()
                and bool((alignment >= 1 - 1e-6).all().item())
            )
        if hold_pose:
            goal_state = current_state.clone()
            ik_result = None
        else:
            # Constructing this short-lived solver keeps the public trajectory API
            # usable without silently changing the TrajOpt configuration's persistent
            # seed state.  Both solvers share robot, device, tolerances, and batch
            # bounds from the caller's configuration.
            ik_config = IKSolverCfg.create(
                self.config.robot_config,
                device_cfg=self.config.device_cfg,
                num_seeds=self.config.num_seeds if num_seeds is None else num_seeds,
                position_tolerance=self.config.position_tolerance,
                orientation_tolerance=self.config.orientation_tolerance,
                max_batch_size=self.config.max_batch_size,
                multi_env=self.config.multi_env,
                max_goalset=self.config.max_goalset,
                use_cuda_graph=False,
                random_seed=self.config.random_seed,
            )
            # A current-state seed is the normal high-level planning default and
            # avoids needlessly exploring random initial states when a caller asks
            # to hold (or make a small change to) the current tool pose.  Supplying
            # every portable seed also avoids an MPS-only NaN propagating from an
            # unrelated random candidate into batched autograd.
            if seed_config is None:
                seed_position = current_state.position
                if seed_position.ndim == 1:
                    seed_position = seed_position.unsqueeze(0)
                seed_count = ik_config.num_seeds
                seed_config = seed_position[:, None, :].expand(
                    -1, seed_count, -1
                ).clone()
            ik_result = IKSolver(ik_config).solve_pose(
                goal_tool_poses,
                current_state=current_state,
                seed_config=seed_config,
                return_seeds=1,
            )
            goal_position = ik_result.solution[:, 0]
            if current_state.position.ndim == 1:
                goal_position = goal_position[0]
            goal_state = JointState.from_position(goal_position, self.joint_names)
    else:
        ik_result = None

    result = self.solve_cspace(
        goal_state,
        current_state,
        seed_traj=seed_traj,
        return_seeds=return_seeds,
        num_seeds=num_seeds,
        dt=dt,
        finetune_attempts=finetune_attempts,
        initial_iters=initial_iters,
        time_optimal_iters=time_optimal_iters,
        finetune_iters=finetune_iters,
        finetune_dt_scale=finetune_dt_scale,
    )
    if ik_result is not None:
        # A trajectory can reach the IK candidate exactly even when that
        # candidate did not reach the requested Cartesian target.  Report this
        # honestly as a failed pose plan while retaining the diagnostic result.
        result.success = result.success & ik_result.success[:, :return_seeds]
        result.debug_info["ik_result"] = ik_result
    # This argument controls CUDA-specific goal rollout handling upstream.
    # The portable IK composition already makes the goal explicit.
    del use_implicit_goal
    return result


# Keep ``TrajectoryOptimizer is TrajOptSolver`` as upstream callers expect,
# while giving that public object the complete pinned ``solve_pose`` signature.
TrajectoryOptimizer.solve_pose = _solve_pose


_create_trajectory_optimizer_cfg = TrajectoryOptimizerCfg.create


@staticmethod
@wraps(_create_trajectory_optimizer_cfg)
def _create_public_trajectory_optimizer_cfg(robot, *args, **kwargs):
    """Resolve packaged robot names accepted by the pinned public example API."""
    if isinstance(robot, str) and not Path(robot).is_file():
        device_cfg = kwargs.get("device_cfg", DeviceCfg())
        kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg)
        robot = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
    return _create_trajectory_optimizer_cfg(robot, *args, **kwargs)


TrajectoryOptimizerCfg.create = _create_public_trajectory_optimizer_cfg

__all__ = [
    "TrajectoryOptimizer", "TrajectoryOptimizerCfg", "TrajectoryOptimizerResult",
]

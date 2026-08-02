"""Backend-neutral solver lifecycle shared by portable high-level facades."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.state.state_joint import JointState

from .solver_core_cfg import SolverCoreCfg


class SolverCore:
    def __init__(self, config: SolverCoreCfg, scene_collision_checker: Optional["SceneCollision"] = None):
        if not isinstance(config, SolverCoreCfg):
            raise TypeError("config must be SolverCoreCfg")
        self.config = config
        self._scene_collision_checker = scene_collision_checker
        robot = config.robot_config.kinematics
        self._kinematics = Kinematics(KinematicsCfg(
            config.device_cfg, list(robot.tool_frames), KinematicsParams(robot)
        ))
        joints = [
            joint for joint in robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        lower = config.device_cfg.to_device([joint.limits.lower for joint in joints])
        upper = config.device_cfg.to_device([joint.limits.upper for joint in joints])
        self._goal_manager = GoalManager(config.device_cfg)
        self._seed_manager = SeedManager(
            config.device_cfg, len(joints), lower, upper,
            config.random_seed, action_horizon=1,
        )

    kinematics = property(lambda self: self._kinematics)
    action_dim = property(lambda self: self._kinematics.dof)
    action_horizon = property(lambda self: 1)
    joint_names = property(lambda self: self._kinematics.joint_names)
    tool_frames = property(lambda self: self._kinematics.tool_frames)
    device_cfg = property(lambda self: self.config.device_cfg)
    goal_registry_manager = property(lambda self: self._goal_manager)
    seed_manager = property(lambda self: self._seed_manager)
    scene_collision_checker = property(lambda self: self._scene_collision_checker)
    optimizer = property(lambda self: None)
    metrics_rollout = property(lambda self: None)
    auxiliary_rollout = property(lambda self: None)
    transition_model = property(lambda self: None)
    solve_state = property(lambda self: None)

    @property
    def default_joint_position(self):
        return self.config.robot_config.kinematics.cspace.default_joint_position

    @property
    def default_joint_state(self):
        return JointState.from_position(
            self.device_cfg.to_device(self.default_joint_position), self.joint_names
        )

    def compute_kinematics(self, state: JointState) -> "KinematicsState":
        return self._kinematics.compute_kinematics(state)
    def get_active_js(self, full_js: JointState) -> JointState:
        return self._kinematics.get_active_js(full_js)
    def get_full_js(self, active_js: JointState) -> JointState:
        return active_js
    def prepare_action_seeds(
        self, batch_size: int, num_seeds: int, seed_config=None,
        current_state: Optional[JointState] = None, seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._seed_manager.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )
    def prepare_trajectory_seeds(
        self, batch_size: int, num_seeds: int, current_state: JointState,
        seed_config=None, seed_traj: Optional[torch.Tensor] = None,
    ) -> None:
        return self._seed_manager.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config, seed_traj
        )
    def reset_seed(self): return self._seed_manager.reset_seed()
    def reset_shape(self): return None
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")
    def destroy(self): return None
    def get_all_rollout_instances(
        self, include_optimizer_rollouts: bool = True, include_auxiliary_rollout: bool = True,
    ) -> List["RobotRollout"]:
        del include_optimizer_rollouts, include_auxiliary_rollout
        return []
    def update_rollout_params(
        self, goal_buffer: "GoalRegistry", include_auxiliary_rollout: bool = True,
    ) -> None:
        del goal_buffer, include_auxiliary_rollout
        return True
    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, "ToolPoseCriteria"]) -> None:
        self.config.tool_pose_criteria = dict(tool_pose_criteria)
    def enable_tool_pose_tracking(
        self, tool_frames: Optional[List[str]] = None, non_terminal_weight_factor: float = 0.0,
    ) -> None:
        del tool_frames, non_terminal_weight_factor
        return None
    def disable_tool_pose_tracking(self, tool_frames: Optional[List[str]] = None) -> None:
        del tool_frames
        return None
    def enable_joint_position_tracking(self) -> None: return None
    def disable_joint_position_tracking(self) -> None: return None
    def sample_configs(
        self, num_samples: int, rejection_ratio: int = 10,
        optimizer_collision_activation_distance: float = 0.01,
    ) -> torch.Tensor:
        del rejection_ratio, optimizer_collision_activation_distance
        return self._seed_manager.generate_random_actions(1, num_samples).squeeze(0)
    def prepare_goal_buffer(
        self, solve_state: "SolveState", goal_tool_poses: "GoalToolPose",
        current_state: Optional[JointState] = None, use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None, goal_state: Optional[JointState] = None,
    ) -> None:
        del use_implicit_goal
        return self._goal_manager.create_goal_buffer(
            solve_state,
            goal_tool_poses=goal_tool_poses,
            goal_js=goal_state,
            current_js=current_state,
            seed_goal_js=seed_goal_state,
        )
    def debug_dump(self, file_path: str) -> None:
        del file_path
        return {"backend": "portable", "cuda_graph": False}
    def update_link_inertial(
        self, link_name: str, mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None, inertia: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(
        self, link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)


__all__ = ["SolverCore"]

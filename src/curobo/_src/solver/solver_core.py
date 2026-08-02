"""Backend-neutral solver lifecycle shared by portable high-level facades."""

from __future__ import annotations

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.state.state_joint import JointState

from .solver_core_cfg import SolverCoreCfg


class SolverCore:
    def __init__(self, config: SolverCoreCfg, scene_collision_checker=None):
        if not isinstance(config, SolverCoreCfg):
            raise TypeError("config must be SolverCoreCfg")
        if scene_collision_checker is not None:
            raise NotImplementedError("external SceneCollision injection is unavailable")
        self.config = config
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
    scene_collision_checker = property(lambda self: None)
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

    def compute_kinematics(self, state): return self._kinematics.compute_kinematics(state)
    def get_active_js(self, full_js): return self._kinematics.get_active_js(full_js)
    def get_full_js(self, active_js): return active_js
    def prepare_action_seeds(self, *args, **kwargs):
        return self._seed_manager.prepare_action_seeds(*args, **kwargs)
    def prepare_trajectory_seeds(self, *args, **kwargs):
        return self._seed_manager.prepare_trajectory_seeds(*args, **kwargs)
    def reset_seed(self): return self._seed_manager.reset_seed()
    def reset_shape(self): return None
    def reset_cuda_graph(self):
        raise NotImplementedError("CUDA graph capture is unavailable on CPU/MPS")
    def destroy(self): return None
    def get_all_rollout_instances(self, **kwargs):
        del kwargs
        return []
    def update_rollout_params(self, **kwargs):
        del kwargs
        return True
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self.config.tool_pose_criteria = dict(tool_pose_criteria)
    def enable_tool_pose_tracking(self, tool_frames=None):
        del tool_frames
        return None
    def disable_tool_pose_tracking(self, tool_frames=None):
        del tool_frames
        return None
    def enable_joint_position_tracking(self): return None
    def disable_joint_position_tracking(self): return None
    def sample_configs(self, num_samples, rejection_ratio=10):
        del rejection_ratio
        return self._seed_manager.generate_random_actions(1, num_samples).squeeze(0)
    def prepare_goal_buffer(self, solve_state, **kwargs):
        return self._goal_manager.create_goal_buffer(solve_state, **kwargs)
    def debug_dump(self, *args, **kwargs):
        del args, kwargs
        return {"backend": "portable", "cuda_graph": False}
    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        raise NotImplementedError(f"runtime inertial mutation is unavailable for {link_name}")
    def update_links_inertial(self, link_properties):
        for name, values in link_properties.items():
            self.update_link_inertial(name, **values)


__all__ = ["SolverCore"]

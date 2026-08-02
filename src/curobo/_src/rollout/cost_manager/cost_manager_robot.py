"""Portable CPU/MPS cost-manager orchestration for robot rollouts.

The upstream implementation overlaps terms on CUDA streams.  This version
preserves the public lifecycle and differentiable cost composition while
evaluating eagerly on the caller's device; CUDA stream/graph mechanics are not
emulated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from curobo._src.cost.portable import (
    CSpaceDistCost,
    SceneCollisionCost,
    SelfCollisionCost,
    ToolPoseCost,
)
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


class RobotCostManager:
    def __init__(self, device_cfg: DeviceCfg = DeviceCfg()):
        self.device_cfg = device_cfg or DeviceCfg()
        self.costs: Dict[str, object] = {}
        self.config = None
        self._initialized = False
        self._batch_size: Optional[int] = None
        self._horizon: Optional[int] = None

    def register_cost(self, name: str, component) -> None:
        if name in self.costs:
            raise ValueError(f"Component {name} already registered")
        self.costs[name] = component

    def get_cost(self, name: str):
        return self.costs.get(name)

    def has_cost(self, name: str) -> bool:
        return name in self.costs

    def _require_cost(self, name: str):
        cost = self.get_cost(name)
        if cost is None:
            raise ValueError(f"Cost component {name} not found")
        return cost

    def enable_cost_component(self, name: str) -> None:
        self._require_cost(name).enable_cost()

    def disable_cost_component(self, name: str) -> None:
        self._require_cost(name).disable_cost()

    def get_enabled_costs(self) -> List[str]:
        return [name for name, cost in self.costs.items() if cost.enabled]

    def get_cost_component_names(self) -> List[str]:
        return list(self.costs)

    def get_cost_components(self) -> Dict[str, object]:
        return self.costs

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> None:
        if batch_size < 0 or horizon < 0:
            raise ValueError("batch_size and horizon must be non-negative")
        if (batch_size, horizon) == (self._batch_size, self._horizon):
            return
        for cost in self.costs.values():
            cost.setup_batch_tensors(batch_size, horizon)
        self._batch_size, self._horizon = int(batch_size), int(horizon)

    def reset(self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs) -> None:
        for cost in self.costs.values():
            cost.reset(reset_problem_ids=reset_problem_ids, **kwargs)

    def update_dt(self, dt: float) -> None:
        for cost in self.costs.values():
            cost.update_dt(dt)

    def initialize_from_config(self, config, transition_model=None, scene_collision_checker=None, **kwargs):
        """Instantiate configured cost terms, using only portable components."""
        if config is None:
            raise TypeError("config must be RobotCostManagerCfg, not None")
        self.config = config
        # Reconfiguration is a normal solver lifecycle operation (for
        # example when an MPC world changes).  CUDA replaces its component
        # instances at construction time; eagerly replacing them here avoids
        # stale weights/checkers and makes the portable path safely
        # idempotent.
        self.costs.clear()
        self._batch_size = self._horizon = None
        robot_model = getattr(transition_model, "robot_model", None)
        total_spheres = getattr(robot_model, "total_spheres", None)
        interpolation_steps = int(getattr(transition_model, "interpolation_steps", 1) or 1)

        if config.self_collision_cfg is not None:
            self_collision_kin_config = None
            if robot_model is not None and hasattr(robot_model, "get_self_collision_config"):
                self_collision_kin_config = robot_model.get_self_collision_config()
                config.self_collision_cfg.self_collision_kin_config = self_collision_kin_config
            # A configured self-collision cost is meaningful for the
            # standalone portable sphere API too.  A transition-backed
            # configuration follows V2 exactly and only registers when its
            # robot model supplies collision metadata.
            if transition_model is None or self_collision_kin_config is not None:
                if transition_model is not None and interpolation_steps > 1:
                    # Do not mutate caller-owned configuration on repeated
                    # initialization.  The cost owns a cloned execution
                    # weight, so scale that after construction instead.
                    component = SelfCollisionCost(config.self_collision_cfg)
                    component._weight.div_(interpolation_steps)
                else:
                    component = SelfCollisionCost(config.self_collision_cfg)
                if total_spheres == 0:
                    component.disable_cost()
                self.register_cost("self_collision", component)

        # V2 deliberately does not create an unusable scene cost without a
        # checker.  Retaining that rule keeps unconfigured planning rollouts
        # executable instead of failing later inside an otherwise optional
        # term.
        if config.scene_collision_cfg is not None and scene_collision_checker is not None:
            config.scene_collision_cfg.scene_collision_checker = scene_collision_checker
            if total_spheres is not None:
                config.scene_collision_cfg.update_num_spheres(total_spheres)
            component = SceneCollisionCost(config.scene_collision_cfg)
            if total_spheres == 0:
                component.disable_cost()
            self.register_cost("scene_collision", component)
        if config.cspace_cfg is not None:
            if transition_model is not None:
                config.cspace_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("cspace", config.cspace_cfg.class_type(config.cspace_cfg))
        if config.tool_pose_cfg is not None:
            if not config.tool_pose_cfg.tool_frames and robot_model is not None:
                config.tool_pose_cfg.set_tool_frames(robot_model.tool_frames)
            if not config.tool_pose_cfg.tool_frames:
                raise ValueError("tool_pose_cfg.tool_frames is required without a robot transition model")
            self.register_cost("tool_pose", ToolPoseCost(config.tool_pose_cfg))
        if config.start_cspace_dist_cfg is not None:
            if transition_model is not None:
                config.start_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("start_cspace_dist", CSpaceDistCost(config.start_cspace_dist_cfg))
        if config.target_cspace_dist_cfg is not None:
            if transition_model is not None:
                config.target_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            self.register_cost("target_cspace_dist", CSpaceDistCost(config.target_cspace_dist_cfg))
        self._initialized = True
        return self

    @staticmethod
    def _joint_state(state):
        return state.joint_state if isinstance(state, RobotState) else state

    def _shape(self, state):
        joint_state = self._joint_state(state)
        if not isinstance(joint_state, JointState) or joint_state.position.ndim < 3:
            raise ValueError("cost manager expects a JointState/RobotState shaped [batch,horizon,dof]")
        return joint_state, joint_state.position.shape[:2]

    def _validate_state_device(self, joint_state: JointState) -> None:
        """Reject accidental host/device mixing before an expensive rollout.

        This is intentionally a validation boundary rather than a hidden
        copy: callers planning on MPS must retain a device-resident autograd
        graph, and moving state inside a cost manager would silently break it.
        """
        expected = torch.device(self.device_cfg.device)
        if not self.device_cfg.is_same_torch_device(joint_state.position.device):
            raise ValueError(
                "robot state device does not match cost manager device: "
                f"{joint_state.position.device} != {expected}"
            )

    def _validate_collision_horizon(self, state, batch: int, horizon: int) -> None:
        """Ensure kinematics and trajectory buffers refer to the same rollout."""
        enabled_collision = any(
            self.has_cost(name) and self.get_cost(name).enabled
            for name in ("self_collision", "scene_collision")
        )
        if not enabled_collision:
            return
        spheres = getattr(state, "robot_spheres", None)
        if spheres is None:
            raise ValueError("enabled collision costs require state.robot_spheres")
        if not isinstance(spheres, torch.Tensor) or spheres.ndim != 4:
            raise ValueError("state.robot_spheres must have shape [batch,horizon,spheres,4]")
        if tuple(spheres.shape[:2]) != (batch, horizon):
            raise ValueError(
                "state.robot_spheres batch/horizon must match state.joint_state: "
                f"{tuple(spheres.shape[:2])} != {(batch, horizon)}"
            )
        if not self.device_cfg.is_same_torch_device(spheres.device):
            raise ValueError("state.robot_spheres device does not match cost manager device")

    def compute_costs(self, state, cost_collection: Optional[CostCollection] = None,
                      goal: Optional[GoalRegistry] = None, **kwargs) -> CostCollection:
        joint_state, (batch, horizon) = self._shape(state)
        self._validate_state_device(joint_state)
        self._validate_collision_horizon(state, batch, horizon)
        self.setup_batch_tensors(batch, horizon)
        output = CostCollection() if cost_collection is None else cost_collection

        cspace = self.get_cost("cspace")
        if cspace is not None and cspace.enabled:
            output.add(cspace.forward(
                joint_state,
                joint_torque=getattr(state, "joint_torque", None),
                target_joint_state=None if goal is None else goal.goal_js,
                idxs_target_joint_state=None if goal is None else goal.idxs_goal_js,
                current_joint_state=None if goal is None else goal.current_js,
                idxs_current_joint_state=None if goal is None else goal.idxs_current_js,
                current_state_dt=None if goal is None else goal.current_state_dt,
            ), "cspace")

        if goal is not None:
            tool = self.get_cost("tool_pose")
            if tool is not None and tool.enabled and goal.link_goal_poses is not None:
                poses = getattr(state, "tool_poses", None)
                if poses is not None:
                    value, _, _, _ = tool.forward(poses, goal.link_goal_poses, goal.idxs_link_pose)
                    output.add(value, "tool_pose")

        spheres = getattr(state, "robot_spheres", None)
        self_collision = self.get_cost("self_collision")
        if self_collision is not None and self_collision.enabled and spheres is not None:
            output.add(self_collision.forward(spheres), "self_collision")
        scene = self.get_cost("scene_collision")
        if scene is not None and scene.enabled and spheres is not None:
            idxs_env = None if goal is None or goal.idxs_env is None else goal.idxs_env.reshape(-1)
            output.add(scene.forward(state, idxs_env, trajectory_dt=joint_state.dt), "scene_collision")
        return output

    def compute_convergence(self, state, goal: Optional[GoalRegistry] = None, **kwargs) -> CostCollection:
        joint_state, (batch, horizon) = self._shape(state)
        self._validate_state_device(joint_state)
        self.setup_batch_tensors(batch, horizon)
        output = CostCollection()
        if goal is None:
            return output
        for name, target, indexes in (
            ("start_cspace_dist", goal.current_js, goal.idxs_current_js),
            ("target_cspace_dist", goal.goal_js, goal.idxs_goal_js),
        ):
            cost = self.get_cost(name)
            if cost is not None and cost.enabled and target is not None:
                output.add(cost.forward(joint_state.position, target.position,
                                        None if indexes is None else indexes.reshape(-1)),
                           f"{name}_tolerance")
        tool = self.get_cost("tool_pose")
        poses = getattr(state, "tool_poses", None)
        if tool is not None and tool.enabled and poses is not None and goal.link_goal_poses is not None:
            _, position, rotation, goalset = tool.forward(poses, goal.link_goal_poses, goal.idxs_link_pose)
            output.add(position, "tool_pose_position_tolerance")
            output.add(rotation, "tool_pose_orientation_tolerance")
            output.add(goalset, "tool_pose_goalset_index")
        return output

    def update_params(self, **kwargs) -> None:
        # Preserve the V2 lifecycle: early solver setup may broadcast update
        # requests before this manager has been configured.
        if not self._initialized:
            return
        if "dt" in kwargs:
            self.update_dt(kwargs["dt"])
        criteria = kwargs.get("tool_pose_criteria")
        tool = self.get_cost("tool_pose")
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise TypeError("tool_pose_criteria must be a dict")
            if tool is not None:
                tool.update_tool_pose_criteria(criteria)


__all__ = ["RobotCostManager"]

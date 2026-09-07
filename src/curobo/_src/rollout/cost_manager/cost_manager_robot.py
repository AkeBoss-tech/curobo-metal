"""Portable CPU/MPS cost-manager orchestration for robot rollouts.

The upstream implementation overlaps terms on CUDA streams.  This version
preserves the public lifecycle and differentiable cost composition while
evaluating eagerly on the caller's device; CUDA stream/graph mechanics are not
emulated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.portable import (
    CSpaceDistCost,
    SceneCollisionCost,
    SelfCollisionCost,
    ToolPoseCost,
)
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.util.logging import log_and_raise, log_info

if TYPE_CHECKING:
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


class RobotCostManager:
    def __init__(self, device_cfg: DeviceCfg = DeviceCfg()):
        self.device_cfg = device_cfg or DeviceCfg()
        self.costs: Dict[str, BaseCost] = {}
        self.config = None
        self._initialized = False
        self._batch_size: Optional[int] = None
        self._horizon: Optional[int] = None

    def register_cost(self, name: str, component: BaseCost) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("cost component name must be a non-empty string")
        if name in self.costs:
            raise ValueError(f"Component {name} already registered")
        for method in ("enable_cost", "disable_cost", "setup_batch_tensors", "reset", "update_dt"):
            if not callable(getattr(component, method, None)):
                raise TypeError(f"cost component {name!r} must provide {method}()")
        component_device = getattr(getattr(component, "device_cfg", None), "device", None)
        if component_device is not None and not self.device_cfg.is_same_torch_device(component_device):
            raise ValueError(
                f"cost component {name!r} is configured for {component_device}, "
                f"not manager device {self.device_cfg.device}"
            )
        self.costs[name] = component

    def get_cost(self, name: str) -> Optional[BaseCost]:
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

    def get_cost_components(self) -> Dict[str, BaseCost]:
        return self.costs

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> None:
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or not isinstance(horizon, int) or isinstance(horizon, bool)
                or batch_size < 0 or horizon < 0):
            raise ValueError("batch_size and horizon must be non-negative")
        if (batch_size, horizon) == (self._batch_size, self._horizon):
            return
        for cost in self.costs.values():
            cost.setup_batch_tensors(batch_size, horizon)
        self._batch_size, self._horizon = int(batch_size), int(horizon)

    def reset(self, reset_problem_ids: Optional[torch.Tensor] = None, **kwargs) -> None:
        if reset_problem_ids is not None:
            if not isinstance(reset_problem_ids, torch.Tensor):
                raise TypeError("reset_problem_ids must be a tensor")
            if reset_problem_ids.ndim != 1:
                raise ValueError("reset_problem_ids must have shape [batch]")
            if reset_problem_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("reset_problem_ids must use an integer dtype")
            if not self.device_cfg.is_same_torch_device(reset_problem_ids.device):
                raise ValueError("reset_problem_ids device does not match cost manager device")
        for cost in self.costs.values():
            cost.reset(reset_problem_ids=reset_problem_ids, **kwargs)

    def update_dt(self, dt: float) -> None:
        if isinstance(dt, torch.Tensor) and not self.device_cfg.is_same_torch_device(dt.device):
            raise ValueError("dt device does not match cost manager device")
        for cost in self.costs.values():
            cost.update_dt(dt)

    def _validate_config_devices(self, config) -> None:
        """Make cross-device configuration mistakes fail before partial setup.

        CUDA's packed buffers make mixed-device configs impossible in
        practice.  The eager backend has no such implicit allocation, so it
        must reject them explicitly instead of letting a later cost create a
        CPU tensor in an MPS rollout.
        """
        for name in (
            "self_collision_cfg", "scene_collision_cfg", "cspace_cfg",
            "start_cspace_dist_cfg", "target_cspace_dist_cfg", "tool_pose_cfg",
        ):
            value = getattr(config, name, None)
            cfg_device = getattr(getattr(value, "device_cfg", None), "device", None)
            if cfg_device is not None and not self.device_cfg.is_same_torch_device(cfg_device):
                raise ValueError(
                    f"{name} is configured for {cfg_device}, not cost manager device "
                    f"{self.device_cfg.device}"
                )

    def _initialize_from_config(
        self,
        config: RobotCostManagerCfg,
        transition_model: RobotStateTransition = None,
        scene_collision_checker: Optional[SceneCollision] = None,
        **kwargs,
    ) -> None:
        """Instantiate configured cost terms, using only portable components."""
        from .cost_manager_robot_cfg import RobotCostManagerCfg

        if not isinstance(config, RobotCostManagerCfg):
            raise TypeError("config must be a RobotCostManagerCfg")
        self._validate_config_devices(config)
        # Validate all predictable failure modes *before* replacing a usable
        # manager.  Reconfiguration happens during MPC/world updates; a bad
        # new config must not strand the old manager half rebuilt.
        if config.tool_pose_cfg is not None:
            configured_frames = config.tool_pose_cfg.tool_frames
            model_frames = getattr(getattr(transition_model, "robot_model", None), "tool_frames", None)
            if not configured_frames and not model_frames:
                raise ValueError("tool_pose_cfg.tool_frames is required without a robot transition model")

        # Reconfiguration is a normal solver lifecycle operation (for
        # example when an MPC world changes).  CUDA replaces its component
        # instances at construction time; eagerly replacing them here avoids
        # stale weights/checkers and makes the portable path safely
        # idempotent.
        robot_model = getattr(transition_model, "robot_model", None)
        total_spheres = getattr(robot_model, "total_spheres", None)
        interpolation_steps = int(getattr(transition_model, "interpolation_steps", 1) or 1)
        new_costs: Dict[str, object] = {}

        def register(name: str, component) -> None:
            if name in new_costs:
                raise ValueError(f"Component {name} already registered")
            component_device = getattr(getattr(component, "device_cfg", None), "device", None)
            if component_device is not None and not self.device_cfg.is_same_torch_device(component_device):
                raise ValueError(f"cost component {name!r} device does not match manager")
            new_costs[name] = component

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
                register("self_collision", component)

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
            register("scene_collision", component)
        if config.cspace_cfg is not None:
            if transition_model is not None:
                config.cspace_cfg.initialize_from_transition_model(transition_model)
            register("cspace", config.cspace_cfg.class_type(config.cspace_cfg))
        if config.tool_pose_cfg is not None:
            if not config.tool_pose_cfg.tool_frames and robot_model is not None:
                config.tool_pose_cfg.set_tool_frames(robot_model.tool_frames)
            register("tool_pose", ToolPoseCost(config.tool_pose_cfg))
        if config.start_cspace_dist_cfg is not None:
            if transition_model is not None:
                config.start_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            register("start_cspace_dist", CSpaceDistCost(config.start_cspace_dist_cfg))
        if config.target_cspace_dist_cfg is not None:
            if transition_model is not None:
                config.target_cspace_dist_cfg.initialize_from_transition_model(transition_model)
            register("target_cspace_dist", CSpaceDistCost(config.target_cspace_dist_cfg))
        self.costs = new_costs
        self.config = config
        self._batch_size = self._horizon = None
        self._initialized = True
        return self

    @staticmethod
    def _joint_state(state):
        return state.joint_state if isinstance(state, RobotState) else state

    def _shape(self, state):
        joint_state = self._joint_state(state)
        if not isinstance(joint_state, JointState) or joint_state.position.ndim != 3:
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

    def _validate_optional_state_tensors(self, state, joint_state: JointState) -> None:
        torque = getattr(state, "joint_torque", None)
        if torque is not None:
            if not isinstance(torque, torch.Tensor):
                raise TypeError("state.joint_torque must be a tensor")
            if torque.device != joint_state.position.device:
                raise ValueError("state.joint_torque must match joint_state device")
            if torque.shape != joint_state.position.shape:
                raise ValueError("state.joint_torque must match joint_state shape")

    def _validate_goal_device(self, goal) -> None:
        if goal is None:
            return
        for name in (
            "idxs_link_pose", "idxs_goal_js", "idxs_current_js", "idxs_env",
            "current_state_dt",
        ):
            value = getattr(goal, name, None)
            if isinstance(value, torch.Tensor) and not self.device_cfg.is_same_torch_device(value.device):
                raise ValueError(f"goal.{name} device does not match cost manager device")
        for name in ("goal_js", "current_js"):
            value = getattr(goal, name, None)
            if value is not None and not self.device_cfg.is_same_torch_device(value.position.device):
                raise ValueError(f"goal.{name} device does not match cost manager device")
        poses = getattr(goal, "link_goal_poses", None)
        if poses is not None and not self.device_cfg.is_same_torch_device(poses.position.device):
            raise ValueError("goal.link_goal_poses device does not match cost manager device")

    @staticmethod
    def _integer_index_dtype(value: torch.Tensor) -> bool:
        return value.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        )

    def _validate_goal_layout(self, goal, batch: int, horizon: int, dof: int,
                              dtype: torch.dtype) -> None:
        """Validate goal tables at the rollout boundary.

        The pinned registry stores selection tensors as ``[batch, 1]`` and
        the CUDA kernels flatten them internally. Validate that portable eager
        execution can make exactly that conversion before a later operation
        raises an incidental broadcasting or ``index_select`` error.
        """
        if goal is None:
            return
        for name in ("goal_js", "current_js"):
            value = getattr(goal, name, None)
            if value is None:
                continue
            if not isinstance(value, JointState):
                raise TypeError(f"goal.{name} must be a JointState")
            position = value.position
            if position.ndim not in (1, 2, 3) or position.shape[-1] != dof:
                raise ValueError(f"goal.{name}.position must end in the rollout DOF")
            if position.dtype != dtype:
                raise TypeError(f"goal.{name}.position dtype must match state.joint_state")

        for name in ("idxs_link_pose", "idxs_goal_js", "idxs_current_js", "idxs_env"):
            value = getattr(goal, name, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or not self._integer_index_dtype(value):
                raise TypeError(f"goal.{name} must be an integer tensor")
            if value.ndim not in (1, 2) or value.numel() != batch:
                raise ValueError(f"goal.{name} must contain one index per rollout batch")

        dt = getattr(goal, "current_state_dt", None)
        if dt is None:
            return
        if not isinstance(dt, torch.Tensor) or not (dt.is_floating_point() or dt.is_complex()):
            raise TypeError("goal.current_state_dt must be a floating tensor")
        if dt.dtype != dtype:
            raise TypeError("goal.current_state_dt dtype must match state.joint_state")
        valid_layout = (
            dt.ndim == 0
            or (dt.ndim == 1 and dt.numel() in (1, batch))
            or (dt.ndim == 2 and dt.shape[0] in (1, batch) and dt.shape[1] in (1, horizon))
        )
        if not valid_layout:
            raise ValueError(
                "goal.current_state_dt must be scalar, [batch], [batch,1], or [batch,horizon]"
            )

    @staticmethod
    def _rollout_current_state_dt(dt: Optional[torch.Tensor], batch: int) -> Optional[torch.Tensor]:
        """Turn a per-problem ``[batch]`` time step into ``[batch, 1]``.

        Costs append one final singleton DOF axis. Leaving a ``[batch]``
        tensor untouched therefore aligns it with the horizon axis, which is
        wrong whenever ``batch != horizon``. The shared registry is not
        mutated because it can be reused with a different horizon.
        """
        if dt is None or dt.ndim != 1 or dt.numel() != batch:
            return dt
        return dt.reshape(batch, 1)

    def compute_costs(self, state: RobotState, cost_collection: Optional[CostCollection] = None,
                      goal: Optional[GoalRegistry] = None, **kwargs) -> CostCollection:
        joint_state, (batch, horizon) = self._shape(state)
        self._validate_state_device(joint_state)
        self._validate_optional_state_tensors(state, joint_state)
        self._validate_goal_device(goal)
        self._validate_goal_layout(goal, batch, horizon, joint_state.position.shape[-1], joint_state.dtype)
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
                current_state_dt=None if goal is None else self._rollout_current_state_dt(
                    goal.current_state_dt, batch
                ),
            ), "cspace")

        if goal is not None:
            tool = self.get_cost("tool_pose")
            if tool is not None and tool.enabled and goal.link_goal_poses is not None:
                poses = getattr(state, "tool_poses", None)
                if poses is None:
                    raise ValueError("enabled tool_pose cost requires state.tool_poses")
                value, _, _, _ = tool.forward(poses, goal.link_goal_poses, goal.idxs_link_pose)
                output.add(value, "tool_pose")

        spheres = getattr(state, "robot_spheres", None)
        self_collision = self.get_cost("self_collision")
        if self_collision is not None and self_collision.enabled and spheres is not None:
            output.add(self_collision.forward(spheres), "self_collision")
        scene = self.get_cost("scene_collision")
        if scene is not None and scene.enabled and spheres is not None:
            idxs_env = None if goal is None or goal.idxs_env is None else goal.idxs_env.reshape(-1)
            if idxs_env is not None:
                # Registry indices follow the pinned int32 ABI, while the
                # portable tensor collision checker consumes PyTorch indices.
                idxs_env = idxs_env.to(dtype=torch.int64)
            output.add(scene.forward(state, idxs_env, trajectory_dt=joint_state.dt), "scene_collision")
        return output

    def compute_convergence(
        self, state: RobotState, goal: Optional[GoalRegistry] = None, **kwargs
    ) -> CostCollection:
        joint_state, (batch, horizon) = self._shape(state)
        self._validate_state_device(joint_state)
        self._validate_optional_state_tensors(state, joint_state)
        self._validate_goal_device(goal)
        self._validate_goal_layout(goal, batch, horizon, joint_state.position.shape[-1], joint_state.dtype)
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
        if tool is not None and tool.enabled and goal.link_goal_poses is not None:
            if poses is None:
                raise ValueError("enabled tool_pose cost requires state.tool_poses")
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
                # Match the public V2 contract: malformed runtime criteria are
                # a value-validation failure (rather than a Python call-shape
                # error).  Downstream callers rely on ValueError here.
                raise ValueError("tool_pose_criteria must be a dict")
            if tool is not None:
                tool.update_tool_pose_criteria(criteria)

    def initialize_from_config(
        self,
        config: RobotCostManagerCfg,
        transition_model: RobotStateTransition,
        scene_collision_checker: Optional[SceneCollision] = None,
        **kwargs,
    ) -> None:
        """Pinned declaration; portable runtime binds the optional implementation below."""
        return self._initialize_from_config(
            config, transition_model, scene_collision_checker, **kwargs
        )


# The eager backend permits a configuration-only setup, a useful portable
# extension used before a robot transition exists.  Retain it at runtime while
# leaving the declared V2 method shape available to static API clients.
RobotCostManager.initialize_from_config = RobotCostManager._initialize_from_config

__all__ = ["RobotCostManager"]

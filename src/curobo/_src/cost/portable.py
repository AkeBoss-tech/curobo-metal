"""Portable implementations of the public cuRobo V2 cost abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo_metal.ops.costs import collision_cost


class UnsupportedCostFeature(RuntimeError):
    """Raised only when the requested operation fundamentally requires Warp/CUDA."""


class CSpaceCostType(Enum):
    POSITION = 0
    STATE = 1


class PoseErrorType(Enum):
    SINGLE_GOAL = 0
    BATCH_GOAL = 1
    GOALSET = 2
    BATCH_GOALSET = 3


@dataclass
class BaseCostCfg:
    weight: Union[torch.Tensor, float, List[float]]
    class_type: Type["BaseCost"] = field(default_factory=lambda: BaseCost)
    device_cfg: DeviceCfg = DeviceCfg()
    convert_to_binary: bool = False
    use_grad_input: bool = False

    def __post_init__(self) -> None:
        self.weight = self.device_cfg.to_device(self.weight)

    def clone(self):
        return replace(self, weight=self.weight.clone())


class BaseCost:
    def __init__(self, config: BaseCostCfg):
        self.config = config
        self.weight = config.weight
        self.use_grad_input = config.use_grad_input
        self._enabled = bool(torch.any(self.weight != 0).item())
        self.batch_size = self.horizon = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        self.batch_size, self.horizon = batch_size, horizon
        return True

    def disable_cost(self):
        self._enabled = False

    def enable_cost(self):
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    def update_dt(self, dt):
        self.dt = self.config.device_cfg.to_device(dt)
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        return True

    def forward(self, *args, **kwargs):
        """Evaluate the cost.

        Concrete costs override this method.  Keeping the abstract-looking
        entrypoint is useful for callers which manage a heterogeneous list of
        cuRobo costs without knowing their concrete type.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement forward")

    def _apply_weight(self, value: torch.Tensor) -> torch.Tensor:
        result = value * self.weight
        if self.config.convert_to_binary:
            result = (result > 0).to(result.dtype)
        return result if self._enabled else result * 0


@dataclass
class CSpaceCostCfg(BaseCostCfg):
    dof: int = 0
    cost_type: Optional[CSpaceCostType] = None
    joint_limits: Optional[Any] = None
    squared_l2_regularization_weight: Optional[List[float]] = None
    retime_weights: bool = False
    retime_regularization_weights: bool = False
    activation_distance: Union[torch.Tensor, float] = 0.0
    cspace_target_weight: Optional[torch.Tensor] = None
    cspace_non_terminal_weight_factor: Optional[torch.Tensor] = None
    cspace_target_dof_weight: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("activation_distance", "cspace_target_weight",
                     "cspace_non_terminal_weight_factor", "cspace_target_dof_weight"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self.device_cfg.to_device(value))

    def set_bounds(self, bounds, teleport_mode: bool = False):
        self.joint_limits = bounds
        return self

    def initialize_from_transition_model(self, transition_model):
        return self

    def update_dof(self, dof: int):
        self.dof = dof
        return self


class BaseCSpaceCost(BaseCost):
    def __init__(self, config: CSpaceCostCfg):
        super().__init__(config)
        self.cspace_target_enabled = True

    def enable_cspace_target(self):
        self.cspace_target_enabled = True

    def disable_cspace_target(self):
        self.cspace_target_enabled = False

    def validate_input(self, state_batch: JointState, *args, **kwargs) -> bool:
        """Validate the portable joint-state portion of the cost contract."""
        if not isinstance(state_batch, JointState):
            raise TypeError("state_batch must be a JointState")
        if state_batch.position.shape[-1] != self.config.dof:
            raise ValueError(
                f"joint state must end in {self.config.dof} values, got "
                f"{state_batch.position.shape[-1]}"
            )
        return True


def _indexed_goal(goal: torch.Tensor, idx: Optional[torch.Tensor], current: torch.Tensor):
    if idx is None:
        return goal
    flat = idx.to(device=goal.device, dtype=torch.long)
    return goal.index_select(0, flat.reshape(-1)).reshape(current.shape)


class PositionCSpaceCost(BaseCSpaceCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def forward(self, state_batch: JointState, joint_torque=None,
                target_joint_state: Optional[JointState] = None,
                idxs_target_joint_state=None, current_joint_state=None,
                idxs_current_joint_state=None, current_state_dt=None):
        q = state_batch.position
        value = torch.zeros_like(q)
        if target_joint_state is not None and self.cspace_target_enabled:
            target = _indexed_goal(target_joint_state.position, idxs_target_joint_state, q)
            value = value + 0.5 * (q - target).square()
        limits = self.config.joint_limits
        if limits is not None:
            lower = getattr(limits, "position", limits)
            if hasattr(lower, "lower") and hasattr(lower, "upper"):
                value = value + 0.5 * (
                    (lower.lower - q).clamp_min(0).square()
                    + (q - lower.upper).clamp_min(0).square()
                )
        return self._apply_weight(value.sum(-1, keepdim=True))

    __call__ = forward


class StateCSpaceCost(PositionCSpaceCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def forward(self, state_batch: JointState, joint_torque=None, **kwargs):
        value = super().forward(state_batch, joint_torque, **kwargs)
        for name in ("velocity", "acceleration", "jerk"):
            tensor = getattr(state_batch, name)
            if tensor is not None:
                value = value + self._apply_weight(0.5 * tensor.square().sum(-1, keepdim=True))
        if joint_torque is not None:
            value = value + self._apply_weight(0.5 * joint_torque.square().sum(-1, keepdim=True))
        return value

    __call__ = forward


@dataclass
class CSpaceDistCostCfg(BaseCostCfg):
    class_type: Type["CSpaceDistCost"] = field(default_factory=lambda: CSpaceDistCost)
    use_null_space: bool = False
    only_terminal_cost: bool = True
    terminal_dof_weight: Optional[Union[torch.Tensor, List[float]]] = None
    non_terminal_dof_weight: Optional[Union[torch.Tensor, List[float]]] = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("terminal_dof_weight", "non_terminal_dof_weight"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self.device_cfg.to_device(value))

    def update_terminal_dof_weight(self, dof_weight):
        self.terminal_dof_weight = self.device_cfg.to_device(dof_weight)

    def update_non_terminal_dof_weight(self, non_terminal_dof_weight):
        self.non_terminal_dof_weight = self.device_cfg.to_device(non_terminal_dof_weight)

    def initialize_from_transition_model(self, transition_model):
        return self

    def update_dof(self, dof: int):
        return self


class CSpaceDistCost(BaseCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def validate_input(self, current_vec, goal_vec, *args, **kwargs) -> bool:
        if not isinstance(current_vec, torch.Tensor) or not isinstance(goal_vec, torch.Tensor):
            raise TypeError("current_vec and goal_vec must be tensors")
        if current_vec.shape[-1] != goal_vec.shape[-1]:
            raise ValueError("current_vec and goal_vec must have the same dof")
        return True

    def forward(self, current_vec, goal_vec, idxs_goal=None):
        goal = _indexed_goal(goal_vec, idxs_goal, current_vec)
        residual = current_vec - goal
        dof_weight = self.config.terminal_dof_weight
        if dof_weight is not None:
            residual = residual * dof_weight
        result = residual.square()
        if self.config.only_terminal_cost and result.ndim >= 3:
            mask = torch.zeros_like(result)
            mask[..., -1, :] = result[..., -1, :]
            result = mask
        return self._apply_weight(result)

    __call__ = forward

    def forward_out_distance(self, current_vec, goal_vec, idxs_goal=None):
        return self.forward(current_vec, goal_vec, idxs_goal).sqrt()

    def jit_squared_cost_to_l2(self, value):
        """Portable eager equivalent of the CUDA helper used by older callers."""
        return torch.as_tensor(value).clamp_min(0).sqrt()


@dataclass
class ToolPoseCostCfg(BaseCostCfg):
    class_type: Type["ToolPoseCost"] = field(default_factory=lambda: ToolPoseCost)
    tool_frames: Optional[List[str]] = None
    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)
    use_lie_group: bool = False

    def set_tool_frames(self, tool_frames):
        self.tool_frames = list(tool_frames)

    @property
    def num_links(self):
        return len(self.tool_frames or [])

    @property
    def rotation_method(self):
        return int(self.use_lie_group)


class ToolPoseCost(BaseCost):
    def update_tool_pose_criteria(self, tool_pose_criteria):
        self.config.tool_pose_criteria = dict(tool_pose_criteria)

    def forward(self, current_tool_poses: ToolPose, goal_tool_poses: GoalToolPose,
                idxs_goal=None, **kwargs):
        pos = current_tool_poses.position.unsqueeze(-2)
        quat = current_tool_poses.quaternion.unsqueeze(-2)
        p_err = (pos - goal_tool_poses.position).square().sum(-1)
        dot = (quat * goal_tool_poses.quaternion).sum(-1).abs().clamp_max(1)
        r_err = 1 - dot.square()
        distance, goal_idx = (p_err + r_err).min(-1)
        return self._apply_weight(distance), p_err.min(-1).values, r_err.min(-1).values, goal_idx

    __call__ = forward


@dataclass
class SceneCollisionCostCfg(BaseCostCfg):
    class_type: Type["SceneCollisionCost"] = field(default_factory=lambda: SceneCollisionCost)
    use_sweep: bool = False
    use_sweep_kernel: bool = False
    use_speed_metric: bool = False
    activation_distance: Union[torch.Tensor, float] = 0.0
    sum_distance: bool = True
    num_spheres: int = 0
    _num_scene_collision_checkers: int = 0
    _scene_collision_checker: Optional[Any] = None

    @property
    def scene_collision_checker(self):
        return self._scene_collision_checker

    @scene_collision_checker.setter
    def scene_collision_checker(self, value):
        self._scene_collision_checker = value

    def update_num_spheres(self, num_spheres):
        self.num_spheres = num_spheres

    def update_num_scene_collision_checkers(self, num_collision_checkers):
        self._num_scene_collision_checkers = int(num_collision_checkers)


class SceneCollisionCost(BaseCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def update_num_spheres(self, num_spheres):
        self.config.update_num_spheres(num_spheres)

    def validate_input(self, state, *args, **kwargs) -> bool:
        spheres = getattr(state, "link_spheres_tensor", getattr(state, "robot_spheres", state))
        if not isinstance(spheres, torch.Tensor) or spheres.shape[-1] != 4:
            raise ValueError("scene collision expects spheres ending in xyzw-radius")
        return True

    def get_gradient_buffer(self):
        checker = self.config.scene_collision_checker
        return getattr(checker, "collision_buffer", None) if checker is not None else None

    def jit_weight_distance(self, distance):
        return self._apply_weight(torch.as_tensor(distance))

    def jit_weight_collision(self, distance):
        return self.jit_weight_distance(distance)

    def forward(self, state, idxs_env_query=None, trajectory_dt=None):
        spheres = getattr(state, "link_spheres_tensor", getattr(state, "robot_spheres", state))
        checker = self.config.scene_collision_checker
        if checker is None:
            raise ValueError("scene_collision_checker is required")
        if hasattr(checker, "get_sphere_distance"):
            distance = checker.get_sphere_distance(spheres, env_query_idx=idxs_env_query)
        else:
            distance = checker(spheres, idxs_env_query)
        result = collision_cost(distance, activation_distance=float(
            torch.as_tensor(self.config.activation_distance).item()))
        return self._apply_weight(result)

    __call__ = forward


@dataclass
class SelfCollisionCostCfg(BaseCostCfg):
    class_type: Type["SelfCollisionCost"] = field(default_factory=lambda: SelfCollisionCost)
    self_collision_kin_config: Optional[Any] = None
    store_pair_distance: bool = False


class SelfCollisionCost(BaseCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def validate_input(self, robot_spheres, *args, **kwargs) -> bool:
        if not isinstance(robot_spheres, torch.Tensor) or robot_spheres.shape[-1] != 4:
            raise ValueError("self collision expects spheres ending in xyzw-radius")
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        return super().reset(reset_problem_ids, **kwargs)

    def forward(self, robot_spheres):
        xyz, radius = robot_spheres[..., :3], robot_spheres[..., 3]
        distance = torch.cdist(xyz, xyz) - radius[..., :, None] - radius[..., None, :]
        upper = torch.triu(torch.ones(distance.shape[-2:], device=distance.device,
                                      dtype=torch.bool), diagonal=1)
        penalty = (-distance[..., upper]).clamp_min(0).square()
        return self._apply_weight(0.5 * penalty.sum(-1, keepdim=True))

    __call__ = forward


@dataclass
class CostSupportPolygonCfg(BaseCostCfg):
    class_type: Type["CostSupportPolygon"] = field(default_factory=lambda: CostSupportPolygon)
    foot_sphere_indices: Optional[torch.Tensor] = None
    foot_link_names: Optional[List[str]] = None
    inside_cost_weight: float = 0.001


class CostSupportPolygon(BaseCost):
    def build_convex_hull(self, vertices, padding=None):
        self.vertices = vertices
        return vertices

    def forward(self, robot_com, robot_spheres):
        if not hasattr(self, "vertices"):
            raise ValueError("build_convex_hull must be called first")
        low, high = self.vertices[..., :2].amin(-2), self.vertices[..., :2].amax(-2)
        xy = robot_com[..., :2]
        outside = (low - xy).clamp_min(0) + (xy - high).clamp_min(0)
        return self._apply_weight(outside.square().sum(-1, keepdim=True))

    __call__ = forward


__all__ = [
    "UnsupportedCostFeature", "CSpaceCostType", "PoseErrorType", "BaseCostCfg",
    "BaseCost", "BaseCSpaceCost", "CSpaceCostCfg", "PositionCSpaceCost",
    "StateCSpaceCost", "CSpaceDistCostCfg", "CSpaceDistCost", "ToolPoseCostCfg",
    "ToolPoseCost", "SceneCollisionCostCfg", "SceneCollisionCost",
    "SelfCollisionCostCfg", "SelfCollisionCost", "CostSupportPolygonCfg",
    "CostSupportPolygon",
]

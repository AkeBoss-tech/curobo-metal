"""Portable implementations of the public cuRobo V2 cost abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch

from curobo._src.cost.tool_pose_criteria import StackedToolPoseCriteria, ToolPoseCriteria
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
        if isinstance(self.weight, bool) or isinstance(self.weight, int):
            raise TypeError("weight must be a float, tensor, or sequence of floats")
        self.weight = self.device_cfg.to_device(self.weight)
        if self.weight.ndim == 0:
            self.weight = self.weight.reshape(1)

    def clone(self):
        """Return an independent portable configuration of the same concrete type.

        Cost configs are often held by a rollout and then adjusted for a
        temporary solve (for example, a terminal target weight).  Returning a
        ``BaseCostCfg`` here, as the early portability shim did, silently
        discarded all derived configuration.  Copy tensor-backed fields while
        preserving explicit external resources such as collision checkers.
        """
        values = {
            item.name: _clone_cost_value(getattr(self, item.name))
            for item in fields(self)
        }
        return type(self)(**values)


def _clone_cost_value(value):
    """Clone mutable cost configuration data without duplicating backends."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_cost_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cost_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_cost_value(item) for key, item in value.items()}
    if isinstance(value, DeviceCfg):
        return value.clone()
    # Joint-limit records and criteria deliberately expose clone().  External
    # scene checkers do not, and must remain a shared query resource.
    clone = getattr(value, "clone", None)
    if callable(clone) and is_dataclass(value):
        return clone()
    return value


class BaseCost:
    def __init__(self, config: BaseCostCfg):
        self.config = config
        self.device_cfg = config.device_cfg
        # Keep a mutable execution weight just like the CUDA implementation.
        # Disabling a cost must not mutate its reusable configuration.
        self._weight = config.weight.clone()
        self.weight = self._weight  # legacy public attribute
        self.use_grad_input = config.use_grad_input
        self._enabled = bool(torch.any(self._weight != 0).item())
        self._batch_size = self._horizon = -1
        self.batch_size = self.horizon = None
        self._dt: Union[float, torch.Tensor] = 1.0

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        if batch_size < 0 or horizon < 0:
            raise ValueError("batch_size and horizon must be non-negative")
        self._batch_size = self.batch_size = int(batch_size)
        self._horizon = self.horizon = int(horizon)
        return True

    def disable_cost(self):
        if not self._enabled:
            return
        self._weight.zero_()
        self._enabled = False

    def enable_cost(self):
        if self._enabled:
            return
        self._weight.copy_(self.config.weight)
        self._enabled = bool(torch.any(self._weight != 0).item())

    @property
    def enabled(self):
        return self._enabled

    def update_dt(self, dt):
        self._dt = self.config.device_cfg.to_device(dt) if not isinstance(dt, torch.Tensor) else dt
        self.dt = self._dt
        return None

    def reset(self, reset_problem_ids=None, **kwargs):
        return None

    def forward(self, *args, **kwargs):
        """Evaluate the cost.

        Concrete costs override this method.  Keeping the abstract-looking
        entrypoint is useful for callers which manage a heterogeneous list of
        cuRobo costs without knowing their concrete type.
        """
        if self._batch_size < 0:
            return torch.zeros((0, 0, 1), **self.device_cfg.as_torch_dict())
        return torch.zeros((self._batch_size, self._horizon, 1), **self.device_cfg.as_torch_dict())

    def _apply_weight(self, value: torch.Tensor) -> torch.Tensor:
        # A component weight is applied by concrete costs.  Here retain the
        # scalar/general-cost behavior without accidentally adding a dimension
        # for vector-valued configurations.
        weight = self._weight if self._weight.numel() == 1 else self._weight.mean()
        result = value * weight
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
        if isinstance(self.cost_type, str):
            self.cost_type = CSpaceCostType[self.cost_type.upper()]
        # Older portable callers omitted cost_type; POSITION is the only
        # backwards-compatible interpretation of a single scalar weight.
        if self.cost_type is None:
            self.cost_type = CSpaceCostType.POSITION
        if self.dof < 0:
            raise ValueError("dof must be non-negative")
        for name in ("activation_distance", "cspace_target_weight",
                     "cspace_non_terminal_weight_factor", "cspace_target_dof_weight",
                     "squared_l2_regularization_weight"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self.device_cfg.to_device(value))
        terms = 2 if self.cost_type is CSpaceCostType.POSITION else 5
        if self.activation_distance.ndim == 0:
            self.activation_distance = self.activation_distance.reshape(1)
        if self.activation_distance.numel() == 1:
            self.activation_distance = self.activation_distance.expand(terms).clone()
        if self.activation_distance.numel() != terms:
            raise ValueError(f"activation_distance must have {terms} values for {self.cost_type.name}")
        if self.weight.numel() == 1:
            self.weight = self.weight.expand(terms).clone()
        if self.weight.numel() != terms:
            raise ValueError(f"weight must have {terms} values for {self.cost_type.name}")
        if self.squared_l2_regularization_weight is None:
            self.squared_l2_regularization_weight = torch.zeros(terms, **self.device_cfg.as_torch_dict())
        elif self.squared_l2_regularization_weight.numel() != terms:
            raise ValueError(f"squared_l2_regularization_weight must have {terms} values")
        if self.cspace_target_weight is None:
            self.cspace_target_weight = torch.zeros(1, **self.device_cfg.as_torch_dict())
        if self.cspace_target_weight.numel() != 1:
            raise ValueError("cspace_target_weight must be scalar")
        if self.cspace_non_terminal_weight_factor is None:
            self.cspace_non_terminal_weight_factor = torch.ones(1, **self.device_cfg.as_torch_dict())
        if self.cspace_non_terminal_weight_factor.numel() != 1:
            raise ValueError("cspace_non_terminal_weight_factor must be scalar")
        self.update_dof(self.dof)
        self.class_type = PositionCSpaceCost if self.cost_type is CSpaceCostType.POSITION else StateCSpaceCost

    def set_bounds(self, bounds, teleport_mode: bool = False):
        self.joint_limits = bounds.clone() if hasattr(bounds, "clone") else bounds
        if teleport_mode and self.cost_type is CSpaceCostType.STATE:
            self.cost_type = CSpaceCostType.POSITION
            self.weight = self.weight[[0, -1]].clone()
            self.activation_distance = self.activation_distance[[0, -1]].clone()
            self.squared_l2_regularization_weight = self.squared_l2_regularization_weight[:2].clone()
            self.class_type = PositionCSpaceCost
        return self

    def initialize_from_transition_model(self, transition_model):
        self.update_dof(transition_model.action_dim)
        if hasattr(transition_model, "get_state_bounds"):
            self.set_bounds(transition_model.get_state_bounds(), bool(getattr(transition_model, "teleport_mode", False)))
        return self

    def update_dof(self, dof: int):
        self.dof = int(dof)
        if self.cspace_target_dof_weight is None or self.cspace_target_dof_weight.numel() != self.dof:
            self.cspace_target_dof_weight = torch.ones(self.dof, **self.device_cfg.as_torch_dict())
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
        if self.config.dof and state_batch.position.shape[-1] != self.config.dof:
            raise ValueError(
                f"joint state must end in {self.config.dof} values, got "
                f"{state_batch.position.shape[-1]}"
            )
        return True


def _indexed_goal(goal: torch.Tensor, idx: Optional[torch.Tensor], current: torch.Tensor):
    if idx is None:
        return goal
    flat = idx.to(device=goal.device, dtype=torch.long)
    selected = goal.index_select(0, flat.reshape(-1))
    # cuRobo goal tables are [goal-batch,dof], while trajectory state is
    # [batch,horizon,dof].  Preserve any existing horizon-shaped goal table.
    while selected.ndim < current.ndim:
        selected = selected.unsqueeze(-2)
    return selected.expand_as(current)


def _term_weight(config: CSpaceCostCfg, index: int, value: torch.Tensor) -> torch.Tensor:
    return config.weight[index].to(device=value.device, dtype=value.dtype)


def _limit_penalty(value: torch.Tensor, limits: Optional[Any], activation: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
    """Squared hinge penalty in the activation band around a [2,dof] range."""
    if limits is None:
        return torch.zeros_like(value)
    if hasattr(limits, "lower") and hasattr(limits, "upper"):
        lower, upper = limits.lower, limits.upper
    elif isinstance(limits, torch.Tensor):
        if limits.shape[0] != 2:
            raise ValueError("limits must have lower/upper in dimension zero")
        lower, upper = limits[0], limits[1]
    else:
        raise TypeError("joint limit entries must expose lower/upper or be [2,dof] tensors")
    lower = torch.as_tensor(lower, device=value.device, dtype=value.dtype)
    upper = torch.as_tensor(upper, device=value.device, dtype=value.dtype)
    if lower.shape[-1] != value.shape[-1] or upper.shape[-1] != value.shape[-1]:
        raise ValueError("joint limits and state have different dof")
    # cuRobo defines activation as a fraction of the joint range.
    margin = torch.as_tensor(activation, device=value.device, dtype=value.dtype) * (upper - lower)
    below = (lower + margin - value).clamp_min(0)
    above = (value - (upper - margin)).clamp_min(0)
    return 0.5 * weight * (below.square() + above.square())


class PositionCSpaceCost(BaseCSpaceCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def forward(self, state_batch: JointState, joint_torque=None,
                target_joint_state: Optional[JointState] = None,
                idxs_target_joint_state=None, current_joint_state=None,
                idxs_current_joint_state=None, current_state_dt=None):
        self.validate_input(state_batch)
        q = state_batch.position
        value = _limit_penalty(q, getattr(self.config.joint_limits, "position", None),
                               self.config.activation_distance[0], _term_weight(self.config, 0, q))
        if target_joint_state is not None and self.cspace_target_enabled:
            target = _indexed_goal(target_joint_state.position, idxs_target_joint_state, q)
            target_weight = self.config.cspace_target_weight.to(q) * self.config.cspace_target_dof_weight.to(q)
            target_value = 0.5 * (q - target).square() * target_weight
            if q.ndim >= 3 and q.shape[-2] > 1:
                target_value = target_value.clone()
                target_value[..., :-1, :] *= self.config.cspace_non_terminal_weight_factor.to(q)
            value = value + target_value
        # Position mode additionally penalizes implied velocity/acceleration
        # when a prior state is supplied, exactly where a trajectory rollout
        # has enough information to form them.
        if current_joint_state is not None and current_joint_state.velocity is not None:
            dt = current_state_dt if current_state_dt is not None else self._dt
            dt = torch.as_tensor(dt, device=q.device, dtype=q.dtype).clamp_min(torch.finfo(q.dtype).eps)
            prior = _indexed_goal(current_joint_state.position, idxs_current_joint_state, q)
            implied_velocity = (q - prior) / dt[..., None]
            value = value + 0.5 * self.config.squared_l2_regularization_weight[0].to(q) * implied_velocity.square()
            prior_velocity = _indexed_goal(current_joint_state.velocity, idxs_current_joint_state, q)
            implied_acceleration = (implied_velocity - prior_velocity) / dt[..., None]
            value = value + 0.5 * self.config.squared_l2_regularization_weight[1].to(q) * implied_acceleration.square()
        return value.sum(-1, keepdim=True) if self.enabled else value.sum(-1, keepdim=True) * 0

    __call__ = forward


class StateCSpaceCost(PositionCSpaceCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def forward(self, state_batch: JointState, joint_torque=None, **kwargs):
        # State bounds are evaluated for every available state component.
        q = state_batch.position
        value = _limit_penalty(q, getattr(self.config.joint_limits, "position", None),
                               self.config.activation_distance[0], _term_weight(self.config, 0, q))
        for index, name in enumerate(("velocity", "acceleration", "jerk"), start=1):
            tensor = getattr(state_batch, name)
            if tensor is not None:
                limit = getattr(self.config.joint_limits, name, None)
                value = value + _limit_penalty(tensor, limit, self.config.activation_distance[index],
                                               _term_weight(self.config, index, tensor))
                value = value + 0.5 * self.config.squared_l2_regularization_weight[index - 1].to(tensor) * tensor.square()
        if joint_torque is not None:
            value = value + _limit_penalty(joint_torque, getattr(self.config.joint_limits, "effort", None),
                                           self.config.activation_distance[4], _term_weight(self.config, 4, joint_torque))
            value = value + 0.5 * self.config.squared_l2_regularization_weight[3].to(joint_torque) * joint_torque.square()
            if state_batch.velocity is not None:
                value = value + self.config.squared_l2_regularization_weight[4].to(joint_torque) * (joint_torque * state_batch.velocity * torch.as_tensor(self._dt, device=q.device, dtype=q.dtype)[..., None]).abs()
        return value.sum(-1, keepdim=True) if self.enabled else value.sum(-1, keepdim=True) * 0

    __call__ = forward


@dataclass
class CSpaceDistCostCfg(BaseCostCfg):
    class_type: Type["CSpaceDistCost"] = field(default_factory=lambda: CSpaceDistCost)
    # Upstream initializes this lazily from RobotStateTransition.  Retaining a
    # concrete default makes direct portable construction safe as well.
    dof: int = 0
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
        if self.terminal_dof_weight is not None:
            self.dof = int(self.terminal_dof_weight.numel())
        elif self.non_terminal_dof_weight is not None:
            self.dof = int(self.non_terminal_dof_weight.numel())

    def update_terminal_dof_weight(self, dof_weight):
        value = self.device_cfg.to_device(dof_weight)
        if self.terminal_dof_weight is not None and self.terminal_dof_weight.shape == value.shape:
            self.terminal_dof_weight.copy_(value)
        else:
            self.terminal_dof_weight = value
        self.dof = int(value.numel())

    def update_non_terminal_dof_weight(self, non_terminal_dof_weight):
        value = self.device_cfg.to_device(non_terminal_dof_weight)
        if self.only_terminal_cost:
            value = torch.zeros_like(value)
        if self.non_terminal_dof_weight is not None and self.non_terminal_dof_weight.shape == value.shape:
            self.non_terminal_dof_weight.copy_(value)
        else:
            self.non_terminal_dof_weight = value

    def initialize_from_transition_model(self, transition_model):
        self.update_dof(transition_model.action_dim)
        attr = "null_space_weight" if self.use_null_space else "cspace_distance_weight"
        if hasattr(transition_model, attr):
            self.update_terminal_dof_weight(getattr(transition_model, attr))
            self.update_non_terminal_dof_weight(getattr(transition_model, attr))
        return self

    def update_dof(self, dof: int):
        self.dof = int(dof)
        self.terminal_dof_weight = torch.ones(self.dof, **self.device_cfg.as_torch_dict())
        self.non_terminal_dof_weight = torch.zeros(self.dof, **self.device_cfg.as_torch_dict()) if self.only_terminal_cost else torch.ones(self.dof, **self.device_cfg.as_torch_dict())
        return self


class CSpaceDistCost(BaseCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        return super().setup_batch_tensors(batch_size, horizon)

    def validate_input(self, current_vec, goal_vec, *args, **kwargs) -> bool:
        if not isinstance(current_vec, torch.Tensor) or not isinstance(goal_vec, torch.Tensor):
            raise TypeError("current_vec and goal_vec must be tensors")
        if current_vec.shape[-1] != goal_vec.shape[-1]:
            raise ValueError("current_vec and goal_vec must have the same dof")
        if current_vec.ndim < 2 or goal_vec.ndim < 1:
            raise ValueError("current_vec and goal_vec must include a dof dimension")
        idxs_goal = args[0] if args else kwargs.get("idxs_goal")
        if idxs_goal is not None and (idxs_goal.ndim != 1 or idxs_goal.shape[0] != current_vec.shape[0]):
            raise ValueError("idxs_goal must have shape [batch]")
        if self.config.dof and current_vec.shape[-1] != self.config.dof:
            raise ValueError("current_vec dof does not match configured dof")
        return True

    def forward(self, current_vec, goal_vec, idxs_goal=None):
        self.validate_input(current_vec, goal_vec, idxs_goal)
        goal = _indexed_goal(goal_vec, idxs_goal, current_vec)
        residual = current_vec - goal
        terminal_weight = self.config.terminal_dof_weight
        if terminal_weight is None:
            terminal_weight = torch.ones(current_vec.shape[-1], device=current_vec.device, dtype=current_vec.dtype)
        terminal_weight = terminal_weight.to(current_vec)
        non_terminal_weight = self.config.non_terminal_dof_weight
        if non_terminal_weight is None:
            non_terminal_weight = torch.zeros_like(terminal_weight) if self.config.only_terminal_cost else terminal_weight
        dof_weight = terminal_weight
        if current_vec.ndim >= 3:
            dof_weight = terminal_weight.expand_as(current_vec)
            dof_weight = dof_weight.clone()
            dof_weight[..., :-1, :] = non_terminal_weight.to(current_vec)
        # Keep component-wise output, matching the portable trajectory stack
        # and preserving gradients all the way to the goal when requested.
        return residual.square() * dof_weight * self._weight.reshape(-1)[0]

    __call__ = forward

    def forward_out_distance(self, current_vec, goal_vec, idxs_goal=None):
        cost = self.forward(current_vec, goal_vec, idxs_goal)
        return cost, self.jit_squared_cost_to_l2(cost, self._weight, torch.ones_like(cost))

    @staticmethod
    def jit_squared_cost_to_l2(value, weight=None, run_weight_vec=None):
        """Portable eager equivalent of the CUDA helper used by older callers."""
        value = torch.as_tensor(value)
        if weight is None:
            return value.clamp_min(0).sqrt()
        weight = torch.as_tensor(weight, device=value.device, dtype=value.dtype)
        if run_weight_vec is not None:
            weight = weight * torch.as_tensor(run_weight_vec, device=value.device, dtype=value.dtype)
        return torch.sqrt(value * torch.nan_to_num(weight.reciprocal(), nan=0.0, posinf=0.0))


@dataclass
class ToolPoseCostCfg(BaseCostCfg):
    class_type: Type["ToolPoseCost"] = field(default_factory=lambda: ToolPoseCost)
    tool_frames: Optional[List[str]] = None
    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)
    use_lie_group: bool = False
    _terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    _non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None
    _terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    _non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None
    _project_distance_to_goal: Union[torch.Tensor, bool] = False
    _pose_criteria: Optional[ToolPoseCriteria] = None

    def __post_init__(self):
        self._pose_criteria = self._pose_criteria or ToolPoseCriteria(
            terminal_pose_axes_weight_factor=self._terminal_pose_axes_weight_factor,
            non_terminal_pose_axes_weight_factor=self._non_terminal_pose_axes_weight_factor,
            terminal_pose_convergence_tolerance=self._terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance=self._non_terminal_pose_convergence_tolerance,
            project_distance_to_goal=self._project_distance_to_goal,
            device_cfg=self.device_cfg,
        )
        super().__post_init__()
        if self.tool_frames is not None:
            self.set_tool_frames(self.tool_frames)

    def clone(self):
        frames = None if self.tool_frames is None else self.tool_frames.copy()
        criteria = {name: value.clone() for name, value in self.tool_pose_criteria.items()}
        clone = replace(self, weight=self.weight.clone(), tool_frames=None,
                        tool_pose_criteria={}, _pose_criteria=None if self._pose_criteria is None else self._pose_criteria.clone())
        clone.tool_frames, clone.tool_pose_criteria = frames, criteria
        return clone

    def set_tool_frames(self, tool_frames):
        self.tool_frames = list(tool_frames)
        if self._pose_criteria is None:
            raise ValueError("pose criteria must be initialized before setting tool frames")
        self.tool_pose_criteria = {name: self._pose_criteria.clone() for name in self.tool_frames}

    @property
    def num_links(self):
        return len(self.tool_frames or [])

    @property
    def rotation_method(self):
        return int(self.use_lie_group)


class ToolPoseCost(BaseCost):
    def __init__(self, config: ToolPoseCostCfg):
        if not config.tool_frames:
            raise ValueError("tool_frames must be set before creating ToolPoseCost")
        super().__init__(config)
        self.tool_frames = list(config.tool_frames)
        self.num_links = len(self.tool_frames)
        self._stacked_tool_pose_criteria = StackedToolPoseCriteria.from_tool_pose_criteria(config.tool_pose_criteria)

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        super().setup_batch_tensors(batch_size, horizon)
        shape = (batch_size, horizon, self.num_links)
        self._out_distance = torch.zeros((*shape, 2), **self.device_cfg.as_torch_dict())
        self._out_position_distance = torch.zeros(shape, **self.device_cfg.as_torch_dict())
        self._out_rotation_distance = torch.zeros(shape, **self.device_cfg.as_torch_dict())
        self._out_goalset_idx = torch.zeros(shape, device=self.device_cfg.device, dtype=torch.int32)
        # These public buffers are useful to consumers that inspect the
        # latest cost diagnostics.  The portable path uses native autograd for
        # the returned value, so buffers are detached observations rather than
        # a custom CUDA backward workspace.
        self._out_position_gradient = torch.zeros((*shape, 3), **self.device_cfg.as_torch_dict())
        self._out_rotation_gradient = torch.zeros((*shape, 4), **self.device_cfg.as_torch_dict())

    def update_tool_pose_criteria(self, tool_pose_criteria):
        if not isinstance(tool_pose_criteria, dict):
            raise TypeError("tool_pose_criteria must be a mapping of tool frame names to criteria")
        unknown = set(tool_pose_criteria).difference(self.tool_frames)
        if unknown:
            raise ValueError(f"tool pose criteria contains unknown configured frames: {sorted(unknown)}")
        # Match the upstream partial-update lifecycle: callers can alter one
        # tool without resetting the criteria for every other configured tool.
        for name, criteria in tool_pose_criteria.items():
            if not isinstance(criteria, ToolPoseCriteria):
                raise TypeError(f"criterion for {name!r} must be a ToolPoseCriteria")
            self.config.tool_pose_criteria[name].copy_(criteria)
        self._stacked_tool_pose_criteria.update_tool_pose_criteria(tool_pose_criteria)

    def forward(self, current_tool_poses: ToolPose, goal_tool_poses: GoalToolPose,
                idxs_goal=None, **kwargs):
        if current_tool_poses.tool_frames != goal_tool_poses.tool_frames:
            raise ValueError("current_tool_poses and goal_tool_poses must use identical tool frames")
        if current_tool_poses.tool_frames != self.tool_frames:
            raise ValueError("tool poses do not match configured tool frames")
        current_pos, current_quat = current_tool_poses.position, current_tool_poses.quaternion
        goal_pos, goal_quat = goal_tool_poses.position, goal_tool_poses.quaternion
        if idxs_goal is not None:
            if idxs_goal.ndim != 1 or idxs_goal.shape[0] != current_pos.shape[0]:
                raise ValueError("idxs_goal must have shape [batch]")
            goal_pos = goal_pos.index_select(0, idxs_goal.to(device=goal_pos.device, dtype=torch.long))
            goal_quat = goal_quat.index_select(0, idxs_goal.to(device=goal_quat.device, dtype=torch.long))
        if goal_pos.shape[:3] != current_pos.shape[:3]:
            raise ValueError("goal pose batch, horizon and tool dimensions must match current poses")
        criteria = self._stacked_tool_pose_criteria
        terminal_axes = criteria.terminal_pose_axes_weight_factor.to(current_pos)
        running_axes = criteria.non_terminal_pose_axes_weight_factor.to(current_pos)
        axes = terminal_axes.expand(current_pos.shape[0], current_pos.shape[1], -1, -1).clone()
        if current_pos.shape[1] > 1:
            axes[:, :-1] = running_axes
        pos_axes, rotation_axes = axes[..., :3], axes[..., 3:]
        position_delta = current_pos.unsqueeze(-2) - goal_pos
        # ``project_distance_to_goal`` measures anisotropic translational
        # errors in the goal frame.  This matters when callers intentionally
        # relax a Cartesian axis (linear approach/retreat criteria); Euclidean
        # norms alone would otherwise hide the requested frame convention.
        project = criteria.project_distance_to_goal.to(current_pos).bool()
        if bool(project.any().item()):
            goal_unit = torch.nn.functional.normalize(goal_quat, dim=-1)
            qw, qx, qy, qz = goal_unit.unbind(-1)
            rotation = torch.stack((
                1 - 2 * (qy.square() + qz.square()), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                2 * (qx * qy + qz * qw), 1 - 2 * (qx.square() + qz.square()), 2 * (qy * qz - qx * qw),
                2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx.square() + qy.square()),
            ), dim=-1).reshape(*goal_unit.shape[:-1], 3, 3)
            projected = torch.matmul(position_delta.unsqueeze(-2), rotation).squeeze(-2)
            position_delta = torch.where(project.reshape(1, 1, -1, 1, 1), projected, position_delta)
        p_err = (position_delta.square() * pos_axes.unsqueeze(-2)).sum(-1)
        current_quat = torch.nn.functional.normalize(current_quat, dim=-1)
        goal_quat = torch.nn.functional.normalize(goal_quat, dim=-1)
        dot = (current_quat.unsqueeze(-2) * goal_quat).sum(-1).abs().clamp_max(1)
        # The scalar quaternion distance is distributed across rotation axes,
        # preserving sign invariance and differentiability without Warp.
        r_err = (1 - dot.square()) * rotation_axes.mean(-1).unsqueeze(-1)
        position_tolerance = criteria.terminal_pose_convergence_tolerance[:, 0].to(current_pos)
        rotation_tolerance = criteria.terminal_pose_convergence_tolerance[:, 1].to(current_pos)
        if current_pos.shape[1] > 1:
            position_tolerance = position_tolerance.expand(current_pos.shape[0], current_pos.shape[1], -1).clone()
            rotation_tolerance = rotation_tolerance.expand(current_pos.shape[0], current_pos.shape[1], -1).clone()
            position_tolerance[:, :-1] = criteria.non_terminal_pose_convergence_tolerance[:, 0].to(current_pos)
            rotation_tolerance[:, :-1] = criteria.non_terminal_pose_convergence_tolerance[:, 1].to(current_pos)
        # Offset the square root by the same epsilon.  ``sqrt(0)`` has an
        # infinite derivative in PyTorch; the algebraically zero residual at
        # an exact goal must instead contribute a finite zero gradient.
        eps = torch.finfo(current_pos.dtype).eps
        p_norm = (p_err.clamp_min(0) + eps).sqrt() - eps**0.5
        r_norm = (r_err.clamp_min(0) + eps).sqrt() - eps**0.5
        p_err = (p_norm - position_tolerance.unsqueeze(-1)).clamp_min(0).square()
        r_err = (r_norm - rotation_tolerance.unsqueeze(-1)).clamp_min(0).square()
        total = p_err + r_err
        distance, goal_idx = total.min(-1)
        p_min = p_err.gather(-1, goal_idx.unsqueeze(-1)).squeeze(-1)
        r_min = r_err.gather(-1, goal_idx.unsqueeze(-1)).squeeze(-1)
        output = self._apply_weight(distance)
        # Update diagnostics only when a batch workspace was explicitly
        # allocated.  Copying detached data keeps a reusable output buffer
        # without retaining a graph across solver iterations.
        if getattr(self, "_out_goalset_idx", None) is not None and self._out_goalset_idx.shape == goal_idx.shape:
            self._out_position_distance.copy_(p_min.detach())
            self._out_rotation_distance.copy_(r_min.detach())
            self._out_distance[..., 0].copy_(p_min.detach())
            self._out_distance[..., 1].copy_(r_min.detach())
            self._out_goalset_idx.copy_(goal_idx.detach().to(torch.int32))
        return output, p_min, r_min, goal_idx.to(torch.int32)

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

    def __post_init__(self):
        if isinstance(self.activation_distance, (float, int)):
            self.activation_distance = self.device_cfg.to_device([float(self.activation_distance)])
        else:
            self.activation_distance = self.device_cfg.to_device(self.activation_distance)
        super().__post_init__()

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
    def __init__(self, config: SceneCollisionCostCfg):
        super().__init__(config)
        self._collision_buffer = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        super().setup_batch_tensors(batch_size, horizon)
        if self.config.num_spheres:
            from curobo._src.geom.collision.buffer_collision import CollisionBuffer
            self._collision_buffer = CollisionBuffer.from_shape(
                torch.Size((batch_size, horizon, self.config.num_spheres, 4)), self.device_cfg
            )

    def update_num_spheres(self, num_spheres, batch_size=None, horizon=None):
        self.config.update_num_spheres(num_spheres)
        if batch_size is not None or horizon is not None:
            self.setup_batch_tensors(self._batch_size if batch_size is None else batch_size,
                                     self._horizon if horizon is None else horizon)

    def validate_input(self, state, *args, **kwargs) -> bool:
        spheres = getattr(state, "link_spheres_tensor", getattr(state, "robot_spheres", state))
        if not isinstance(spheres, torch.Tensor) or spheres.shape[-1] != 4:
            raise ValueError("scene collision expects spheres ending in xyzw-radius")
        if spheres.ndim != 4:
            raise ValueError("scene collision expects [batch,horizon,spheres,4]")
        if self.config.num_spheres and spheres.shape[-2] != self.config.num_spheres:
            raise ValueError("sphere count does not match configured num_spheres")
        idxs_env_query = args[0] if args else kwargs.get("idxs_env_query")
        if idxs_env_query is not None and (idxs_env_query.ndim != 1 or idxs_env_query.shape[0] != spheres.shape[0]):
            raise ValueError("idxs_env_query must have shape [batch]")
        return True

    def get_gradient_buffer(self):
        if self._collision_buffer is not None:
            return self._collision_buffer.gradient
        checker = self.config.scene_collision_checker
        return getattr(checker, "collision_buffer", None) if checker is not None else None

    @staticmethod
    def jit_weight_distance(distance, sum_cost=True):
        distance = torch.as_tensor(distance)
        return distance.sum(-1) if sum_cost else distance.max(-1).values

    @staticmethod
    def jit_weight_collision(distance, sum_cost=True):
        value = SceneCollisionCost.jit_weight_distance(distance, sum_cost)
        return torch.where(value > 0, value + 1.0, value)

    def forward(self, state, idxs_env_query=None, trajectory_dt=None):
        spheres = getattr(state, "link_spheres_tensor", getattr(state, "robot_spheres", state))
        self.validate_input(state, idxs_env_query)
        checker = self.config.scene_collision_checker
        if checker is None:
            raise ValueError("scene_collision_checker is required")
        if self.config.use_sweep and hasattr(checker, "get_swept_sphere_distance"):
            if trajectory_dt is None:
                raise ValueError("trajectory_dt is required when use_sweep=True")
            try:
                distance = checker.get_swept_sphere_distance(
                    state, self._collision_buffer, self._weight,
                    activation_distance=self.config.activation_distance,
                    trajectory_dt=trajectory_dt, enable_speed_metric=self.config.use_speed_metric,
                    env_query_idx=idxs_env_query, return_loss=self.use_grad_input,
                )
            except TypeError:
                # Lightweight user checkers commonly only accept raw spheres.
                # Preserve that duck-typed contract while full SceneCollision
                # receives the complete portable lifecycle above.
                distance = checker.get_swept_sphere_distance(spheres, trajectory_dt=trajectory_dt,
                                                             env_query_idx=idxs_env_query)
        elif hasattr(checker, "get_sphere_distance"):
            try:
                distance = checker.get_sphere_distance(spheres, env_query_idx=idxs_env_query)
            except TypeError:
                distance = checker.get_sphere_distance(state, self._collision_buffer, self._weight,
                                                       activation_distance=self.config.activation_distance,
                                                       env_query_idx=idxs_env_query, return_loss=self.use_grad_input)
        else:
            distance = checker(spheres, idxs_env_query)
        distance = torch.as_tensor(distance, device=spheres.device, dtype=spheres.dtype)
        # Checkers either return signed clearance or an already-positive loss.
        # A signed-clearance tensor has a sphere dimension matching the query.
        if distance.shape[-1:] == spheres.shape[-2:-1]:
            activation = self.config.activation_distance.reshape(-1)[0].to(distance)
            # collision_cost is a trajectory-level helper and reduces its
            # final dimension.  Keep sphere-resolved values here so this
            # cuRobo facade can honor sum_distance vs max_distance.
            result = 0.5 * (activation - distance).clamp_min(0).square()
            result = self.jit_weight_collision(result, self.config.sum_distance) if self.config.convert_to_binary else self.jit_weight_distance(result, self.config.sum_distance)
        else:
            result = distance
        return self._apply_weight(result)

    __call__ = forward


@dataclass
class SelfCollisionCostCfg(BaseCostCfg):
    class_type: Type["SelfCollisionCost"] = field(default_factory=lambda: SelfCollisionCost)
    self_collision_kin_config: Optional[Any] = None
    store_pair_distance: bool = False


class SelfCollisionCost(BaseCost):
    def setup_batch_tensors(self, batch_size: int, horizon: int):
        super().setup_batch_tensors(batch_size, horizon)
        cfg = self.config.self_collision_kin_config
        if cfg is not None and self.config.store_pair_distance:
            pairs = getattr(cfg, "collision_pairs", torch.empty((0, 2), dtype=torch.long))
            self._pair_distance = torch.zeros((batch_size, horizon, len(pairs)), **self.device_cfg.as_torch_dict())

    def validate_input(self, robot_spheres, *args, **kwargs) -> bool:
        if not isinstance(robot_spheres, torch.Tensor) or robot_spheres.ndim != 4 or robot_spheres.shape[-1] != 4:
            raise ValueError("self collision expects spheres ending in xyzw-radius")
        cfg = self.config.self_collision_kin_config
        if cfg is not None and hasattr(cfg, "num_spheres") and robot_spheres.shape[-2] != cfg.num_spheres:
            raise ValueError("sphere count does not match self collision kinematics config")
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        return super().reset(reset_problem_ids, **kwargs)

    def forward(self, robot_spheres):
        self.validate_input(robot_spheres)
        xyz, radius = robot_spheres[..., :3], robot_spheres[..., 3]
        cfg = self.config.self_collision_kin_config
        padding = 0.0 if cfg is None else getattr(cfg, "sphere_padding", 0.0)
        distance = torch.cdist(xyz, xyz) - radius[..., :, None] - radius[..., None, :] - torch.as_tensor(padding, device=xyz.device, dtype=xyz.dtype)
        pairs = None if cfg is None else getattr(cfg, "collision_pairs", None)
        if pairs is None:
            upper = torch.triu(torch.ones(distance.shape[-2:], device=distance.device, dtype=torch.bool), diagonal=1)
            pair_distance = distance[..., upper]
        else:
            pairs = torch.as_tensor(pairs, device=distance.device, dtype=torch.long)
            pair_distance = distance[..., pairs[:, 0], pairs[:, 1]]
        if self.config.store_pair_distance:
            self._pair_distance = pair_distance
        # Upstream returns the largest penetration, not the sum of all pairs.
        penalty = (-pair_distance).clamp_min(0)
        result = penalty.amax(-1, keepdim=True)
        if self.config.convert_to_binary:
            result = torch.where(result > 0, result.clamp_max(1) + 1, result)
        return self._apply_weight(result)

    __call__ = forward


@dataclass
class CostSupportPolygonCfg(BaseCostCfg):
    class_type: Type["CostSupportPolygon"] = field(default_factory=lambda: CostSupportPolygon)
    foot_sphere_indices: Optional[torch.Tensor] = None
    foot_link_names: Optional[List[str]] = None
    inside_cost_weight: float = 0.001


class CostSupportPolygon(BaseCost):
    def __init__(self, config: CostSupportPolygonCfg):
        super().__init__(config)
        from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
        self._polygon_helper = ConvexPolygon2DHelper(config.device_cfg)
    def build_convex_hull(self, vertices, padding=None):
        self._polygon_helper.build_convex_hull(vertices, padding)
        self.vertices = self._polygon_helper._cached_convex_hulls
        return self.vertices

    def forward(self, robot_com, robot_spheres):
        if robot_com.ndim != 3 or robot_com.shape[-1] != 3:
            raise ValueError("robot_com must have shape [batch,horizon,3]")
        if robot_spheres.ndim != 4 or robot_spheres.shape[-1] != 4:
            raise ValueError("robot_spheres must have shape [batch,horizon,spheres,4]")
        if self._polygon_helper._cached_convex_hulls is None:
            indices = self.config.foot_sphere_indices
            if indices is None:
                raise ValueError("foot_sphere_indices must be configured or build_convex_hull called")
            self.build_convex_hull(robot_spheres[:, 0, indices, :2].detach(), padding=0.05)
        idx = torch.arange(robot_com.shape[0], device=robot_com.device)
        signed = self._polygon_helper.compute_point_hull_distance(robot_com[..., :2].unsqueeze(-2), idx).squeeze(-1)
        outside = signed.clamp_min(0)
        inside = self.config.inside_cost_weight * (signed + 0.1).clamp(0, 0.1)
        value = torch.where(signed < 0, inside, outside)
        return self._apply_weight(value)

    __call__ = forward


__all__ = [
    "UnsupportedCostFeature", "CSpaceCostType", "PoseErrorType", "BaseCostCfg",
    "BaseCost", "BaseCSpaceCost", "CSpaceCostCfg", "PositionCSpaceCost",
    "StateCSpaceCost", "CSpaceDistCostCfg", "CSpaceDistCost", "ToolPoseCostCfg",
    "ToolPoseCost", "SceneCollisionCostCfg", "SceneCollisionCost",
    "SelfCollisionCostCfg", "SelfCollisionCost", "CostSupportPolygonCfg",
    "CostSupportPolygon",
]

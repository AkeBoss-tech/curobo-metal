"""Portable multi-link tool-pose goalset cost.

The upstream module launches a specialised Warp kernel and exposes an
interleaved ``[position, rotation]`` cost for each configured tool link.  This
implementation deliberately keeps that public tensor contract while composing
ordinary PyTorch operations so it works on CPU and MPS and retains gradients
through both the current and goal poses.

Raw Warp kernel objects and CUDA custom-backward/stream semantics are not
portable and are intentionally not emulated here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg

from curobo._src.cost.cost_base import BaseCost
from .tool_pose_criteria import StackedToolPoseCriteria, ToolPoseCriteria
from .wp_tool_pose import ToolPoseDistance, create_goalset_pose_distance_kernel_with_constants
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import wp
from .portable import BaseCost as _PortableBaseCost


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first quaternions with broadcastable leading shapes."""
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def _quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized scalar-first quaternions to active rotation matrices."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y.square() + z.square()), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x.square() + y.square()),
    ), dim=-1).reshape(*quaternion.shape[:-1], 3, 3)


class _ToolPoseCostPortable(_PortableBaseCost):
    """Differentiable CPU/MPS implementation of the cuRobo ToolPoseCost API.

    The returned cost has shape ``[batch, horizon, 2 * num_links]`` with
    position and rotation costs interleaved per link, matching the upstream
    Warp-facing contract.  The two diagnostic distance tensors have shape
    ``[batch, horizon, num_links]`` and report geometric distances rather than
    weighted costs, as their CUDA counterparts do.
    """

    def __init__(self, config):
        if not config.tool_frames:
            raise ValueError("tool_frames must be set before creating ToolPoseCost")
        super().__init__(config)
        self.tool_frames = list(config.tool_frames)
        self.num_links = len(self.tool_frames)
        self._stacked_tool_pose_criteria = StackedToolPoseCriteria.from_tool_pose_criteria(
            config.tool_pose_criteria
        )
        self._out_distance = None
        self._out_position_distance = None
        self._out_rotation_distance = None
        self._out_goalset_idx = None
        self._out_position_gradient = None
        self._out_rotation_gradient = None

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        del kwargs
        if batch_size < 0 or horizon < 0:
            raise ValueError("batch_size and horizon must be non-negative")
        needs_allocation = (
            self._out_distance is None
            or self._out_distance.shape != (batch_size, horizon, 2 * self.num_links)
        )
        if needs_allocation:
            spec = self.device_cfg.as_torch_dict()
            self._out_distance = torch.zeros((batch_size, horizon, 2 * self.num_links), **spec)
            self._out_position_distance = torch.zeros((batch_size, horizon, self.num_links), **spec)
            self._out_rotation_distance = torch.zeros((batch_size, horizon, self.num_links), **spec)
            self._out_goalset_idx = torch.zeros(
                (batch_size, horizon, self.num_links), device=self.device_cfg.device, dtype=torch.int32
            )
            self._out_position_gradient = torch.zeros((batch_size, horizon, self.num_links, 3), **spec)
            self._out_rotation_gradient = torch.zeros((batch_size, horizon, self.num_links, 4), **spec)
        return super().setup_batch_tensors(batch_size, horizon)

    def update_tool_pose_criteria(self, tool_pose_criteria):
        if not isinstance(tool_pose_criteria, dict):
            raise TypeError("tool_pose_criteria must be a mapping of tool frame names to criteria")
        unknown = set(tool_pose_criteria).difference(self.tool_frames)
        if unknown:
            raise ValueError(f"tool pose criteria contains unknown configured frames: {sorted(unknown)}")
        for name, criteria in tool_pose_criteria.items():
            if not isinstance(criteria, ToolPoseCriteria):
                raise TypeError(f"criterion for {name!r} must be a ToolPoseCriteria")
            self.config.tool_pose_criteria[name].copy_(criteria)
        self._stacked_tool_pose_criteria.update_tool_pose_criteria(tool_pose_criteria)

    @staticmethod
    def _validate_poses(current: ToolPose, goal: GoalToolPose, frames) -> None:
        if current is None or goal is None:
            raise ValueError("current_tool_poses and goal_tool_poses must be provided")
        if current.tool_frames != goal.tool_frames:
            raise ValueError("current_tool_poses and goal_tool_poses must use identical tool frames")
        if current.tool_frames != frames:
            raise ValueError("tool poses do not match configured tool frames")
        if current.position.shape != (*current.position.shape[:-1], 3):
            raise ValueError("current tool positions must end in dimension 3")
        if current.quaternion.shape != (*current.quaternion.shape[:-1], 4):
            raise ValueError("current tool quaternions must end in dimension 4")
        if current.position.shape[:3] != current.quaternion.shape[:3]:
            raise ValueError("current tool position/quaternion batch, horizon, and link dimensions must match")
        if goal.position.shape[2] != current.position.shape[2] or goal.position.shape[3] < 1:
            raise ValueError("goal pose link dimension must match current poses and goalset must be non-empty")
        if current.position.device != current.quaternion.device or goal.position.device != goal.quaternion.device:
            raise ValueError("position and quaternion tensors must share a device")
        if current.position.device != goal.position.device:
            raise ValueError("current and goal tool poses must share a device")
        if current.position.dtype != current.quaternion.dtype or goal.position.dtype != goal.quaternion.dtype:
            raise ValueError("position and quaternion tensors must share a dtype")
        if current.position.dtype != goal.position.dtype:
            raise ValueError("current and goal tool poses must share a dtype")

    def _resolve_goals(
        self, current: ToolPose, goal: GoalToolPose, idxs_goal: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select goal batches and broadcast the upstream single goal horizon."""
        batch, horizon = current.position.shape[:2]
        if goal.position.shape[1] not in (1, horizon):
            raise ValueError("goal horizon must be one or match the current pose horizon")
        goal_position, goal_quaternion = goal.position, goal.quaternion
        if goal.position.shape[1] == 1 and horizon != 1:
            goal_position = goal_position.expand(-1, horizon, -1, -1, -1)
            goal_quaternion = goal_quaternion.expand(-1, horizon, -1, -1, -1)
        if idxs_goal is None:
            if goal_position.shape[0] != batch:
                raise ValueError("goal batch must match current poses when idxs_goal is omitted")
            return goal_position, goal_quaternion
        indices = torch.as_tensor(idxs_goal, device=goal_position.device)
        if indices.ndim == 2 and indices.shape[1] == 1:
            indices = indices[:, 0]
        if indices.ndim != 1 or indices.numel() != batch:
            raise ValueError("idxs_goal must have shape [batch] or [batch, 1]")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("idxs_goal must use int32 or int64")
        if bool(((indices < 0) | (indices >= goal_position.shape[0])).any().item()):
            raise IndexError("idxs_goal contains an out-of-range goal batch index")
        return (
            goal_position.index_select(0, indices.to(torch.long)),
            goal_quaternion.index_select(0, indices.to(torch.long)),
        )

    def _weights(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self._weight.to(value)
        if weights.numel() == 1:
            return weights.reshape(()), weights.reshape(())
        if weights.numel() != 2:
            raise ValueError("ToolPoseCost weight must contain one scalar or [position, rotation]")
        return weights[0], weights[1]

    def forward(self, current_tool_poses: ToolPose, goal_tool_poses: GoalToolPose,
                idxs_goal: Optional[torch.Tensor] = None, **kwargs):
        del kwargs
        self._validate_poses(current_tool_poses, goal_tool_poses, self.tool_frames)
        current_pos, current_quat = current_tool_poses.position, current_tool_poses.quaternion
        goal_pos, goal_quat = self._resolve_goals(current_tool_poses, goal_tool_poses, idxs_goal)
        batch, horizon, links = current_pos.shape[:3]
        if (batch, horizon, links) != (current_quat.shape[0], current_quat.shape[1], current_quat.shape[2]):
            raise ValueError("current position and quaternion shapes are incompatible")

        criteria = self._stacked_tool_pose_criteria
        terminal = criteria.terminal_pose_axes_weight_factor.to(current_pos)
        running = criteria.non_terminal_pose_axes_weight_factor.to(current_pos)
        axes = terminal.reshape(1, 1, links, 6).expand(batch, horizon, -1, -1)
        if horizon > 1:
            axes = axes.clone()
            axes[:, :-1] = running.reshape(1, 1, links, 6)
        position_axes, rotation_axes = axes[..., :3], axes[..., 3:]

        terminal_tolerance = criteria.terminal_pose_convergence_tolerance.to(current_pos)
        running_tolerance = criteria.non_terminal_pose_convergence_tolerance.to(current_pos)
        tolerance = terminal_tolerance.reshape(1, 1, links, 2).expand(batch, horizon, -1, -1)
        if horizon > 1:
            tolerance = tolerance.clone()
            tolerance[:, :-1] = running_tolerance.reshape(1, 1, links, 2)

        current_unit = torch.nn.functional.normalize(current_quat, dim=-1)
        goal_unit = torch.nn.functional.normalize(goal_quat, dim=-1)
        position_delta = current_pos.unsqueeze(-2) - goal_pos
        goal_rotation = _quat_to_matrix(goal_unit)
        project = criteria.project_distance_to_goal.to(current_pos).bool().reshape(1, 1, links, 1, 1)
        projected_delta = torch.matmul(position_delta.unsqueeze(-2), goal_rotation).squeeze(-2)
        position_delta = torch.where(project, projected_delta, position_delta)

        position_weight, rotation_weight = self._weights(current_pos)
        position_cost = 0.5 * position_weight * (position_delta * position_axes.unsqueeze(-2)).square().sum(-1)
        position_cost = torch.where(
            position_cost < tolerance[..., 0].unsqueeze(-1).square(), torch.zeros_like(position_cost), position_cost
        )

        relative = _quat_multiply(
            current_unit.unsqueeze(-2),
            torch.cat((goal_unit[..., :1], -goal_unit[..., 1:]), dim=-1),
        )
        # The quaternion double cover must not change a pose cost.  The
        # positive-hemisphere convention also keeps the log map continuous at
        # the identity for the Lie-group option.
        relative = torch.where(relative[..., :1] < 0, -relative, relative)
        vector, scalar = relative[..., 1:], relative[..., :1].abs()
        vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        geometric_angle = 2 * torch.atan2(vector_norm, scalar)
        tangent = torch.where(
            vector_norm > torch.finfo(current_pos.dtype).eps,
            geometric_angle * vector / vector_norm.clamp_min(torch.finfo(current_pos.dtype).eps),
            2 * vector,
        )
        if self.config.use_lie_group:
            rotation_vector = tangent
            angular_distance = torch.linalg.vector_norm(tangent * rotation_axes.unsqueeze(-2), dim=-1)
        else:
            weighted_vector = vector * rotation_axes.unsqueeze(-2)
            weighted_norm = torch.linalg.vector_norm(weighted_vector, dim=-1, keepdim=True)
            axis = weighted_vector / weighted_norm.clamp_min(torch.finfo(current_pos.dtype).eps)
            rotation_vector = 2 * torch.atan2(weighted_norm, scalar) * axis
            angular_distance = torch.linalg.vector_norm(rotation_vector, dim=-1)
        rotation_cost = rotation_weight * rotation_vector.square().sum(-1)
        rotation_cost = torch.where(
            rotation_cost < tolerance[..., 1].unsqueeze(-1).square(), torch.zeros_like(rotation_cost), rotation_cost
        )
        angular_distance = torch.where(rotation_weight == 0, torch.zeros_like(angular_distance), angular_distance)

        total = position_cost + rotation_cost
        _, goal_idx = total.min(dim=-1)
        selected_position_cost = position_cost.gather(-1, goal_idx.unsqueeze(-1)).squeeze(-1)
        selected_rotation_cost = rotation_cost.gather(-1, goal_idx.unsqueeze(-1)).squeeze(-1)
        selected_angle = angular_distance.gather(-1, goal_idx.unsqueeze(-1)).squeeze(-1)
        selected_position_distance = torch.where(
            position_weight == 0,
            torch.zeros_like(selected_position_cost),
            torch.sqrt((2 * selected_position_cost / position_weight).clamp_min(0)),
        )

        output = torch.stack((selected_position_cost, selected_rotation_cost), dim=-1).flatten(-2)
        if self.config.convert_to_binary:
            output = (output > 0).to(output.dtype)
        if not self._enabled:
            output = output * 0

        if self._out_distance is not None and self._out_distance.shape == output.shape:
            self._out_distance.copy_(output.detach())
            self._out_position_distance.copy_(selected_position_distance.detach())
            self._out_rotation_distance.copy_(selected_angle.detach())
            self._out_goalset_idx.copy_(goal_idx.detach().to(torch.int32))
            # Native PyTorch autograd supplies the authoritative VJP.  These
            # buffers remain detached diagnostic workspaces; position is
            # useful to legacy observers, while rotation is intentionally not
            # a second, approximate quaternion backward implementation.
            selected_goal_pos = goal_pos.gather(
                -2, goal_idx[..., None, None].expand(*goal_idx.shape, 1, 3)
            ).squeeze(-2)
            position_grad = position_weight * position_axes.square() * (current_pos - selected_goal_pos)
            if bool(project.any().item()):
                selected_goal_rotation = goal_rotation.gather(
                    -3, goal_idx[..., None, None, None].expand(*goal_idx.shape, 1, 3, 3)
                ).squeeze(-3)
                projected_grad = torch.matmul(position_grad.unsqueeze(-2), selected_goal_rotation.transpose(-1, -2)).squeeze(-2)
                selected_project = criteria.project_distance_to_goal.to(current_pos).bool().reshape(
                    1, 1, links
                ).expand(batch, horizon, -1)
                position_grad = torch.where(selected_project.unsqueeze(-1), projected_grad, position_grad)
            self._out_position_gradient.copy_(position_grad.detach())
            self._out_rotation_gradient.zero_()
        return output, selected_position_distance, selected_angle, goal_idx.to(torch.int32)

    __call__ = forward


class ToolPoseCost(BaseCost):
    """Pinned cuRoboV2 declaration surface for portable tool-pose scoring."""

    def __init__(self, config: ToolPoseCostCfg):
        raise NotImplementedError

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: GoalToolPose,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update_tool_pose_criteria(
        self,
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
    ):
        raise NotImplementedError


# The runtime implementation preserves differentiable CPU/MPS goalset scoring
# and diagnostics.  The facade above retains the V2 declaration contract for
# static API consumers without pretending Warp is present.
if not TYPE_CHECKING:
    ToolPoseCost = _ToolPoseCostPortable


__all__ = [
    "BaseCost", "ToolPoseCost", "ToolPose", "GoalToolPose", "StackedToolPoseCriteria",
    "ToolPoseDistance", "create_goalset_pose_distance_kernel_with_constants",
]

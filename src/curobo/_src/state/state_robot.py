"""Portable robot-state record compatible with cuRobo's state lifecycle.

The CUDA implementation associates a packed robot-model result with a joint
state.  The Metal port deliberately uses the regular :class:`KinematicsState`
value object instead: all public data remains ordinary PyTorch tensors and can
therefore be indexed, cloned, differentiated, and moved between CPU and MPS
without pretending to expose CUDA buffer handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose

from .state_base import State
from .state_joint import JointState


def _model_tensor_fields(model_state: Any) -> tuple[str, ...]:
    """The tensor fields shared by the portable FK state implementations."""
    return tuple(
        name
        for name in ("robot_spheres", "tool_jacobians", "robot_com")
        if isinstance(getattr(model_state, name, None), torch.Tensor)
    )


def _model_uses_merged_batch_seed_dim(
    joint_position: torch.Tensor, model_state: Any
) -> bool:
    """Recognize FK ``[B*S,H,...]`` storage for a joint ``[B,S,...]`` state.

    Some solver buffers retain the seed dimension on their joint state while
    flattening it for forward kinematics.  Detecting that representation from
    actual tensor extents gives the same copy semantics on CPU and MPS without
    depending on a CUDA model-state ABI.
    """
    if joint_position.ndim < 3:
        return False
    batch, seeds = joint_position.shape[:2]
    tensors = [getattr(model_state, name, None) for name in _model_tensor_fields(model_state)]
    poses = getattr(model_state, "tool_poses", None)
    if poses is not None:
        tensors.append(getattr(poses, "position", None))
    return any(
        isinstance(value, torch.Tensor)
        and value.ndim >= 1
        and value.shape[0] == batch * seeds
        for value in tensors
    )


def _index_model_state(model_state: Any, index: Union[int, torch.Tensor]) -> Any:
    """Index a KinematicsState, with a narrow legacy tensor-record fallback."""
    if model_state is None:
        return None
    try:
        return model_state[index]
    except (TypeError, AttributeError):
        # A few public integrations construct a minimal ``SimpleNamespace``
        # carrying sphere tensors.  Preserve that data-only compatibility
        # rather than treating it as a CUDA state object.
        values = {}
        for name in _model_tensor_fields(model_state):
            values[name] = getattr(model_state, name)[index]
        poses = getattr(model_state, "tool_poses", None)
        if poses is not None:
            values["tool_poses"] = poses[index]
        return SimpleNamespace(**values)


@dataclass
class RobotState(State):
    """Joint, torque, and optional forward-kinematics state.

    ``cuda_robot_model_state`` keeps its upstream field name for source
    compatibility.  In this backend it is a portable ``KinematicsState`` (or
    a small tensor record accepted for backwards-compatible integrations), not
    a CUDA object.
    """

    joint_state: JointState
    joint_torque: Optional[torch.Tensor] = None
    cuda_robot_model_state: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_state, JointState):
            raise TypeError("joint_state must be a JointState")
        if self.joint_torque is not None:
            if not isinstance(self.joint_torque, torch.Tensor):
                raise TypeError("joint_torque must be a torch.Tensor")
            if self.joint_torque.device != self.joint_state.device:
                raise ValueError("joint_torque must reside on the joint_state device")

    def data_ptr(self) -> int:
        return self.joint_state.data_ptr()

    def __len__(self) -> int:
        return len(self.joint_state)

    @property
    def device(self) -> torch.device:
        return self.joint_state.device

    @property
    def dtype(self) -> torch.dtype:
        return self.joint_state.dtype

    @property
    def shape(self) -> torch.Size:
        return self.joint_state.shape

    @property
    def robot_spheres(self) -> Optional[torch.Tensor]:
        return None if self.cuda_robot_model_state is None else getattr(
            self.cuda_robot_model_state, "robot_spheres", None
        )

    @property
    def link_poses(self) -> Optional[ToolPose]:
        return None if self.cuda_robot_model_state is None else getattr(
            self.cuda_robot_model_state, "tool_poses", None
        )

    @property
    def tool_poses(self) -> Optional[ToolPose]:
        return self.link_poses

    @property
    def tool_frames(self) -> List[str]:
        return [] if self.tool_poses is None else self.tool_poses.tool_frames

    def get_link_pose(self, link_name: str) -> Pose:
        if self.tool_poses is None:
            raise ValueError("Link poses are not set")
        return self.tool_poses.get_link_pose(link_name)

    def __getitem__(self, index: Union[int, torch.Tensor]) -> "RobotState":
        return type(self)(
            self.joint_state[index],
            None if self.joint_torque is None else self.joint_torque[index],
            _index_model_state(self.cuda_robot_model_state, index),
        )

    def clone(self) -> "RobotState":
        model_state = self.cuda_robot_model_state
        if model_state is not None:
            clone = getattr(model_state, "clone", None)
            if not callable(clone):
                raise TypeError("robot model state must provide clone()")
            model_state = clone()
        return type(self)(
            self.joint_state.clone(),
            None if self.joint_torque is None else self.joint_torque.clone(),
            model_state,
        )

    def detach(self) -> "RobotState":
        model_state = self.cuda_robot_model_state
        if model_state is not None:
            detach = getattr(model_state, "detach", None)
            if not callable(detach):
                raise TypeError("robot model state must provide detach()")
            model_state = detach()
        return type(self)(
            # ``JointState.detach`` is an upstream-compatible in-place helper;
            # detach a clone so RobotState.detach has normal value semantics.
            self.joint_state.clone().detach(),
            None if self.joint_torque is None else self.joint_torque.detach(),
            model_state,
        )

    def contiguous(self) -> "RobotState":
        model_state = self.cuda_robot_model_state
        if model_state is not None:
            contiguous = getattr(model_state, "contiguous", None)
            if callable(contiguous):
                model_state = contiguous()
            else:
                raise TypeError("robot model state must provide contiguous()")
        joint_state = self.joint_state.clone()
        for name in joint_state._tensor_fields():
            value = getattr(joint_state, name)
            if value is not None:
                setattr(joint_state, name, value.contiguous())
        return type(self)(
            joint_state,
            None if self.joint_torque is None else self.joint_torque.contiguous(),
            model_state,
        )

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "RobotState":
        """Return a state moved through normal PyTorch CPU/MPS operations."""
        if device_cfg is not None and (device is not None or dtype is not None):
            raise ValueError("pass either device_cfg or device/dtype to RobotState.to()")
        if device_cfg is None:
            if device is None and dtype is None:
                return self
            device_cfg = DeviceCfg(
                torch.device(self.device if device is None else device),
                self.dtype if dtype is None else dtype,
            )
        model_state = self.cuda_robot_model_state
        if model_state is not None:
            move = getattr(model_state, "to", None)
            if not callable(move):
                raise TypeError("robot model state must provide to()")
            model_state = move(device_cfg)
        return type(self)(
            self.joint_state.to(device_cfg),
            None if self.joint_torque is None else self.joint_torque.to(**device_cfg.as_torch_dict()),
            model_state,
        )

    def requires_grad_(self, requires_grad: bool = True) -> "RobotState":
        """Enable/disable first-order gradients for every floating payload."""
        for name in self.joint_state._tensor_fields():
            value = getattr(self.joint_state, name)
            if value is not None and (value.is_floating_point() or value.is_complex()):
                value.requires_grad_(requires_grad)
        if self.joint_torque is not None and (
            self.joint_torque.is_floating_point() or self.joint_torque.is_complex()
        ):
            self.joint_torque.requires_grad_(requires_grad)
        if self.cuda_robot_model_state is not None:
            require = getattr(self.cuda_robot_model_state, "requires_grad_", None)
            if callable(require):
                require(requires_grad)
        return self

    def copy_(self, other: "RobotState") -> "RobotState":
        if not isinstance(other, RobotState):
            raise TypeError("other must be a RobotState")
        self.joint_state.copy_(other.joint_state)
        if self.joint_torque is not None:
            if other.joint_torque is None:
                raise ValueError("cannot copy missing joint_torque")
            self.joint_torque.copy_(other.joint_torque)
        if self.cuda_robot_model_state is not None:
            if other.cuda_robot_model_state is None:
                raise ValueError("cannot copy missing robot model state")
            copy = getattr(self.cuda_robot_model_state, "copy_", None)
            if not callable(copy):
                raise TypeError("robot model state must provide copy_()")
            copy(other.cuda_robot_model_state)
        return self

    def copy_only_index(self, other: "RobotState", index: Union[int, torch.Tensor]) -> "RobotState":
        """Copy selected leading entries into an existing solver buffer."""
        if not isinstance(other, RobotState):
            raise TypeError("other must be a RobotState")
        self.joint_state.copy_only_index(other.joint_state, index)
        self._copy_model_indices(other, index, None)
        if self.joint_torque is not None:
            if other.joint_torque is None:
                raise ValueError("cannot copy missing joint_torque")
            self.joint_torque[index] = other.joint_torque[index]
        return self

    def copy_at_batch_seed_indices(
        self, other: "RobotState", batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ) -> "RobotState":
        """Copy selected ``[batch, seed]`` entries, including flattened FK buffers."""
        if not isinstance(other, RobotState):
            raise TypeError("other must be a RobotState")
        if batch_idx.shape != seed_idx.shape:
            raise ValueError("batch_idx and seed_idx must have the same shape")
        if batch_idx.device != self.device or seed_idx.device != self.device:
            raise ValueError("batch_idx and seed_idx must reside on the RobotState device")
        self.joint_state.copy_at_batch_seed_indices(other.joint_state, batch_idx, seed_idx)
        self._copy_model_indices(other, batch_idx, seed_idx)
        if self.joint_torque is not None:
            if other.joint_torque is None:
                raise ValueError("cannot copy missing joint_torque")
            self.joint_torque[batch_idx, seed_idx] = other.joint_torque[batch_idx, seed_idx]
        return self

    def _copy_model_indices(
        self, other: "RobotState", index: Union[int, torch.Tensor], seed_idx: Optional[torch.Tensor]
    ) -> None:
        target, source = self.cuda_robot_model_state, other.cuda_robot_model_state
        if target is None:
            return
        if source is None:
            raise ValueError("cannot copy missing robot model state")
        merged = seed_idx is not None and _model_uses_merged_batch_seed_dim(
            self.joint_state.position, target
        )
        model_index: Union[int, torch.Tensor] = index
        if merged:
            assert seed_idx is not None
            model_index = index * self.joint_state.position.shape[1] + seed_idx
        for name in ("robot_spheres", "tool_jacobians", "robot_com"):
            target_value, source_value = getattr(target, name, None), getattr(source, name, None)
            if target_value is not None:
                if source_value is None:
                    raise ValueError(f"cannot copy missing robot-model source field: {name}")
                if seed_idx is not None and not merged:
                    target_value[index, seed_idx] = source_value[index, seed_idx]
                else:
                    target_value[model_index] = source_value[model_index]
        target_pose, source_pose = getattr(target, "tool_poses", None), getattr(source, "tool_poses", None)
        if target_pose is not None:
            if source_pose is None:
                raise ValueError("cannot copy missing robot-model source field: tool_poses")
            if seed_idx is not None and not merged:
                target_pose.position[index, seed_idx] = source_pose.position[index, seed_idx]
                target_pose.quaternion[index, seed_idx] = source_pose.quaternion[index, seed_idx]
            else:
                target_pose.position[model_index] = source_pose.position[model_index]
                target_pose.quaternion[model_index] = source_pose.quaternion[model_index]


__all__ = ["RobotState"]

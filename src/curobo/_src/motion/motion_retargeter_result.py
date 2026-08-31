"""Portable result lifecycle for motion retargeting.

The pinned V2 ``RetargetResult`` deliberately declares only a final
``JointState`` and an optional MPC trajectory.  CUDA cuRobo normally keeps
those values in solver-owned buffers; the Metal port materialises them as
ordinary PyTorch states instead.  The small lifecycle helpers below make a
retargeting result safe to retain, batch-select, or move between CPU and MPS
without claiming CUDA result-buffer compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


_BatchIndex = Union[int, slice, torch.Tensor, list[int], tuple[int, ...]]


class _RetargetResultPortableMixin:
    """Result from :meth:`MotionRetargeter.solve_frame` or ``solve_sequence``.

    ``joint_state.position`` has shape ``[environment, dof]`` for a frame and
    ``[environment, frame, dof]`` for a sequence.  In MPC mode, ``trajectory``
    is a rank-three ``[environment, intermediate-frame, dof]`` endpoint
    stream; it is ``None`` for IK retargeting.  Both states remain ordinary
    differentiable CPU/MPS tensors until :meth:`detach` is requested.
    """

    joint_state: JointState
    trajectory: JointState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_state, JointState):
            raise TypeError("joint_state must be a JointState")
        position = self.joint_state.position
        if position.ndim not in (2, 3):
            raise ValueError(
                "joint_state.position must have shape [environment, dof] or "
                "[environment, frame, dof]"
            )
        if position.shape[0] < 1 or position.shape[-1] < 1:
            raise ValueError("joint_state must contain at least one environment and degree of freedom")

        if self.trajectory is None:
            return
        if not isinstance(self.trajectory, JointState):
            raise TypeError("trajectory must be a JointState or None")
        trajectory_position = self.trajectory.position
        if trajectory_position.ndim != 3:
            raise ValueError(
                "trajectory.position must have shape [environment, intermediate-frame, dof]"
            )
        if trajectory_position.shape[0] != position.shape[0]:
            raise ValueError("joint_state and trajectory must share the environment batch size")
        if trajectory_position.shape[-1] != position.shape[-1]:
            raise ValueError("joint_state and trajectory must share the final DOF dimension")
        if trajectory_position.device != position.device:
            raise ValueError("joint_state and trajectory must share a device")
        if trajectory_position.dtype != position.dtype:
            raise TypeError("joint_state and trajectory must share a dtype")
        left_names, right_names = self.joint_state.joint_names, self.trajectory.joint_names
        if left_names is not None and right_names is not None and left_names != right_names:
            raise ValueError("joint_state and trajectory must have identical joint_names when declared")

    @property
    def batch_size(self) -> int:
        """Number of independent retargeting environments in this result."""
        return int(self.joint_state.position.shape[0])

    @property
    def num_dof(self) -> int:
        """Number of controllable joints in the result's final dimension."""
        return int(self.joint_state.position.shape[-1])

    @property
    def num_frames(self) -> int:
        """Number of final-state frames (one for a ``solve_frame`` result)."""
        return 1 if self.joint_state.position.ndim == 2 else int(self.joint_state.position.shape[1])

    @property
    def num_trajectory_frames(self) -> int:
        """Number of MPC endpoint frames, or zero when using IK retargeting."""
        return 0 if self.trajectory is None else int(self.trajectory.position.shape[1])

    @property
    def is_sequence(self) -> bool:
        """Whether ``joint_state`` contains a time/frame dimension."""
        return self.joint_state.position.ndim == 3

    @property
    def is_mpc(self) -> bool:
        """Whether the result carries a materialised MPC endpoint stream."""
        return self.trajectory is not None

    @property
    def device(self) -> torch.device:
        """Device that owns the materialised joint-state tensors."""
        return self.joint_state.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the materialised joint-state tensors."""
        return self.joint_state.dtype

    def clone(self) -> "RetargetResult":
        """Return a result with independent state tensors and metadata."""
        return type(self)(
            joint_state=self.joint_state.clone(),
            trajectory=None if self.trajectory is None else self.trajectory.clone(),
        )

    def detach(self) -> "RetargetResult":
        """Return a cloned result detached from the caller's autograd graph."""
        output = self.clone()
        output.joint_state.detach()
        if output.trajectory is not None:
            output.trajectory.detach()
        return output

    def to(self, device_cfg: DeviceCfg | torch.device | str) -> "RetargetResult":
        """Move floating state channels to CPU/MPS while preserving their layout.

        Passing a :class:`DeviceCfg` also applies its declared floating dtype.
        A ``torch.device`` or string retains this result's floating dtype.
        """
        if not isinstance(device_cfg, DeviceCfg):
            device_cfg = DeviceCfg(torch.device(device_cfg), self.dtype)
        return type(self)(
            joint_state=self.joint_state.to(device_cfg),
            trajectory=None if self.trajectory is None else self.trajectory.to(device_cfg),
        )

    def select_batch(self, index: _BatchIndex) -> "RetargetResult":
        """Select environments while preserving a leading batch dimension.

        An integer selects a one-environment result instead of returning a
        rank-reduced state.  This is useful for forwarding a chosen retarget
        result directly into a solver that expects ``[batch, ...]`` tensors.
        """
        indices = self._normalise_batch_index(index)
        return type(self)(
            joint_state=self.joint_state[indices],
            trajectory=None if self.trajectory is None else self.trajectory[indices],
        )

    def __getitem__(self, index: _BatchIndex) -> "RetargetResult":
        """Alias for :meth:`select_batch` with batch-preserving integer semantics."""
        return self.select_batch(index)

    def _normalise_batch_index(self, index: _BatchIndex) -> torch.Tensor:
        device = self.device
        if isinstance(index, slice):
            indices = torch.arange(self.batch_size, device=device, dtype=torch.long)[index]
        elif isinstance(index, int):
            normalised = index + self.batch_size if index < 0 else index
            if normalised < 0 or normalised >= self.batch_size:
                raise IndexError("retarget result batch index is out of range")
            indices = torch.tensor([normalised], device=device, dtype=torch.long)
        elif isinstance(index, (list, tuple)):
            indices = torch.as_tensor(index, device=device)
        elif isinstance(index, torch.Tensor):
            indices = index.to(device=device)
        else:
            raise TypeError("batch index must be an int, slice, sequence, or tensor")
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.ndim != 1:
            raise IndexError("batch index must be one-dimensional")
        if indices.dtype == torch.bool:
            if indices.numel() != self.batch_size:
                raise IndexError("boolean batch index must have batch_size elements")
            return indices
        if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("batch index tensor must be integer or boolean")
        indices = indices.to(dtype=torch.long)
        indices = torch.where(indices < 0, indices + self.batch_size, indices)
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.batch_size):
            raise IndexError("retarget result batch index is out of range")
        return indices


@dataclass
class RetargetResult(_RetargetResultPortableMixin):
    joint_state: JointState
    trajectory: Optional[JointState] = None


__all__ = ["RetargetResult"]

"""Portable lifecycle helpers for the pinned MPC solver result model.

The V2 public result is a small dataclass populated by an MPC solve.  CUDA
cuRobo additionally keeps graph-resident command/result buffers behind this
object.  Those buffers are deliberately *not* represented here: this module
operates only on materialized PyTorch tensors and state values, so it behaves
the same on CPU and Apple MPS.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Union

import torch

from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


_BatchIndex = Union[torch.Tensor, Sequence[int], int]


def _clone_value(value: Any) -> Any:
    """Clone nested, tensor-bearing diagnostic values without aliasing them."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value.clone() if hasattr(value, "clone") else value


def _move_tensor(value: torch.Tensor, device_cfg: DeviceCfg) -> torch.Tensor:
    """Move floating data without corrupting boolean or integer result fields."""
    dtype = device_cfg.dtype if value.is_floating_point() or value.is_complex() else value.dtype
    return value.to(device=device_cfg.device, dtype=dtype)


def _move_value(value: Any, device_cfg: DeviceCfg) -> Any:
    if isinstance(value, torch.Tensor):
        return _move_tensor(value, device_cfg)
    if isinstance(value, JointState):
        return value.to(device_cfg)
    if isinstance(value, RobotState):
        # A CUDA robot-model state can contain opaque packed buffer handles.
        # It has no portable device-transfer contract, so rejecting it is
        # safer than returning an object with mixed-device tensor fields.
        if value.cuda_robot_model_state is not None:
            raise NotImplementedError(
                "moving RobotState CUDA model buffers is unavailable in the portable backend"
            )
        torque = None if value.joint_torque is None else _move_tensor(value.joint_torque, device_cfg)
        return type(value)(value.joint_state.to(device_cfg), torque)
    if isinstance(value, Mapping):
        return type(value)((key, _move_value(item, device_cfg)) for key, item in value.items())
    if isinstance(value, list):
        return [_move_value(item, device_cfg) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device_cfg) for item in value)
    return value


def _select_batch_value(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select values that have the public MPC leading batch dimension."""
    if isinstance(value, torch.Tensor):
        return value[indices] if value.ndim > 0 and value.shape[0] == batch_size else value
    if isinstance(value, JointState):
        return value[indices] if value.position.ndim > 0 and value.position.shape[0] == batch_size else value
    if isinstance(value, RobotState):
        if value.joint_state.position.ndim > 0 and value.joint_state.position.shape[0] == batch_size:
            return value[indices]
        return value
    if isinstance(value, Mapping):
        return type(value)(
            (key, _select_batch_value(item, indices, batch_size)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_select_batch_value(item, indices, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_select_batch_value(item, indices, batch_size) for item in value)
    return value


def _select_action_channel(state: JointState, index: int) -> JointState:
    """Extract one command while retaining the leading batch axis."""
    if state.position.ndim != 3:
        raise ValueError("action sequences must have shape [batch, horizon, dof]")
    if index < 0 or index >= state.position.shape[1]:
        raise IndexError("action index is outside the available action horizon")

    def select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        # State derivative channels use the sequence layout.  A scalar or
        # per-batch timing value remains valid metadata for the command.
        if value.ndim >= 2 and value.shape[:2] == state.position.shape[:2]:
            return value[:, index].clone()
        return value.clone()

    return type(state)(
        state.position[:, index].clone(), select(state.velocity), select(state.acceleration),
        None if state.joint_names is None else state.joint_names.copy(), select(state.jerk),
        state.device_cfg, select(state.dt), aux_data=dict(state.aux_data),
        knot=select(state.knot), knot_dt=select(state.knot_dt), control_space=state.control_space,
    )


def _copy_batch_tensor(
    target: Optional[torch.Tensor], source: Optional[torch.Tensor], mask: torch.Tensor, name: str
) -> None:
    """Copy a complete MPC batch selection with explicit layout failures."""
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError(f"both {name} fields must be set or both must be None")
    if target.device != source.device or target.shape != source.shape:
        raise ValueError(f"{name} tensors must share shape and device")
    if target.ndim < 1 or target.shape[0] != mask.numel():
        raise ValueError(f"{name} must have the MPC batch as its leading dimension")
    target[mask] = source[mask]


def _copy_batch_joint_state(
    target: Optional[JointState], source: Optional[JointState], mask: torch.Tensor, name: str
) -> None:
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError(f"both {name} fields must be set or both must be None")
    for field_name in target._tensor_fields():
        _copy_batch_tensor(
            getattr(target, field_name), getattr(source, field_name), mask, f"{name}.{field_name}"
        )


def _copy_batch_robot_state(
    target: Optional[RobotState], source: Optional[RobotState], mask: torch.Tensor
) -> None:
    if target is None and source is None:
        return
    if target is None or source is None:
        raise ValueError("both robot_state_sequence fields must be set or both must be None")
    # RobotState owns portable model-state copy semantics (including flattened
    # batch/seed FK payloads).  Reuse that public lifecycle instead of
    # guessing which model fields happen to be materialised.
    target.copy_only_index(source, mask)


class _MPCSolverResultPortableMixin:
    """Result specific to the portable MPC solver.

    Command data use ``[batch, horizon, dof]``.  The helpers retain the batch
    axis when slicing so callers can pass selected results back into batched
    planners.  CUDA graph handles, packed result buffers, and CUDA robot-model
    state transfer remain explicit unsupported boundaries.
    """

    next_action: Optional[JointState] = None
    action_sequence: Optional[JointState] = None
    full_action_sequence: Optional[JointState] = None
    robot_state_sequence: Optional[RobotState] = None
    action_buffer: Optional[torch.Tensor] = None
    action_dt: Optional[float] = None

    @property
    def device(self) -> Optional[torch.device]:
        """Device of the first materialized portable result field."""
        for name in ("action_buffer", "solution", "success", "seed_cost", "cspace_error"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                return value.device
        for value in (self.next_action, self.action_sequence, self.full_action_sequence, self.js_solution):
            if value is not None:
                return value.device
        if self.robot_state_sequence is not None:
            return self.robot_state_sequence.joint_state.device
        return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Floating dtype of command/result data, when materialized."""
        for name in ("action_buffer", "solution", "seed_cost", "cspace_error"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                return value.dtype
        for value in (self.next_action, self.action_sequence, self.full_action_sequence, self.js_solution):
            if value is not None:
                return value.dtype
        return None

    @property
    def action_horizon(self) -> int:
        """Number of materialized MPC commands, or zero when no plan exists."""
        for value in (self.action_buffer, None if self.action_sequence is None else self.action_sequence.position):
            if isinstance(value, torch.Tensor):
                if value.ndim != 3:
                    raise ValueError("MPC action data must have shape [batch, horizon, dof]")
                return int(value.shape[1])
        return 0

    @property
    def result_batch_size(self) -> int:
        """Infer the result batch from metadata or any materialized payload."""
        if self.batch_size > 0:
            return int(self.batch_size)
        for value in (
            self.success, self.action_buffer,
            None if self.action_sequence is None else self.action_sequence.position,
            None if self.next_action is None else self.next_action.position,
        ):
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        return 0

    def clone(self) -> "MPCSolverResult":
        """Deep-clone result tensors, state channels, and nested diagnostics."""
        return type(self)(**{item.name: _clone_value(getattr(self, item.name)) for item in fields(self)})

    def to(self, device_cfg: DeviceCfg) -> "MPCSolverResult":
        """Return a portable copy on ``device_cfg``.

        Boolean and integral values keep their dtype.  Moving opaque CUDA
        robot-model state is explicitly unsupported rather than fabricated.
        """
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        return type(self)(
            **{item.name: _move_value(getattr(self, item.name), device_cfg) for item in fields(self)}
        )

    def validate_action_layout(self) -> None:
        """Validate the materialized command sequence contract without mutation."""
        batch = self.result_batch_size
        device = self.device
        for name in ("action_buffer",):
            value = getattr(self, name)
            if value is None:
                continue
            if value.ndim != 3 or (batch and value.shape[0] != batch):
                raise ValueError(f"{name} must have shape [batch, horizon, dof]")
            if value.shape[1] < 1:
                raise ValueError(f"{name} must contain at least one action")
            if device is not None and value.device != device:
                raise ValueError(f"{name} must reside on the result device")
        for name in ("next_action", "action_sequence", "full_action_sequence"):
            value = getattr(self, name)
            if value is None:
                continue
            position = value.position
            expected_rank = 2 if name == "next_action" else 3
            if position.ndim != expected_rank or (batch and position.shape[0] != batch):
                shape = "[batch, dof]" if name == "next_action" else "[batch, horizon, dof]"
                raise ValueError(f"{name}.position must have shape {shape}")
            if device is not None and position.device != device:
                raise ValueError(f"{name} must reside on the result device")
        if self.action_buffer is not None and self.action_sequence is not None:
            if self.action_buffer.shape != self.action_sequence.position.shape:
                raise ValueError("action_buffer and action_sequence.position must share shape")
        if self.action_sequence is not None and self.full_action_sequence is not None:
            if self.action_sequence.position.shape != self.full_action_sequence.position.shape:
                raise ValueError("action_sequence and full_action_sequence must share shape")
        if self.robot_state_sequence is not None:
            position = self.robot_state_sequence.joint_state.position
            if position.ndim != 3 or (batch and position.shape[0] != batch):
                raise ValueError("robot_state_sequence.joint_state.position must have shape [batch, horizon, dof]")
            if device is not None and position.device != device:
                raise ValueError("robot_state_sequence must reside on the result device")
        if self.action_dt is not None and (not isinstance(self.action_dt, (float, int)) or self.action_dt <= 0):
            raise ValueError("action_dt must be a positive scalar when provided")

    def select_batch(self, indices: _BatchIndex) -> "MPCSolverResult":
        """Select batch rows while preserving a leading batch dimension."""
        batch = self.result_batch_size
        if batch < 1:
            raise ValueError("a positive batch size is required for batch selection")
        device = self.device if self.device is not None else torch.device("cpu")
        tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        if tensor.numel() and (tensor.min() < 0 or tensor.max() >= batch):
            raise IndexError("batch index is outside the available result rows")
        values = {
            item.name: _select_batch_value(getattr(self, item.name), tensor, batch)
            for item in fields(self)
        }
        values["batch_size"] = int(tensor.numel())
        return type(self)(**values)

    def __getitem__(self, indices: _BatchIndex) -> "MPCSolverResult":
        """Alias for :meth:`select_batch`, matching result-style indexing."""
        return self.select_batch(indices)

    def action_at(self, index: int) -> JointState:
        """Return one command from the current portable action plan.

        ``action_buffer`` takes precedence because it is the current
        receding-horizon command stream.  The method has no execution side
        effects; advancing CUDA graph/action-buffer cursors is unsupported.
        """
        if not isinstance(index, int):
            raise TypeError("action index must be an integer")
        if self.action_buffer is not None:
            if self.action_buffer.ndim != 3:
                raise ValueError("action_buffer must have shape [batch, horizon, dof]")
            if index < 0 or index >= self.action_buffer.shape[1]:
                raise IndexError("action index is outside the available action horizon")
            names = None if self.action_sequence is None else self.action_sequence.joint_names
            dt = None
            if self.action_sequence is not None and self.action_sequence.dt is not None:
                sequence_dt = self.action_sequence.dt
                if sequence_dt.ndim >= 2 and sequence_dt.shape[:2] == self.action_buffer.shape[:2]:
                    dt = sequence_dt[:, index].clone()
                else:
                    dt = sequence_dt.clone()
            elif self.action_dt is not None:
                dt = torch.full(
                    (self.action_buffer.shape[0],), float(self.action_dt),
                    device=self.action_buffer.device, dtype=self.action_buffer.dtype,
                )
            return JointState(
                self.action_buffer[:, index].clone(),
                torch.zeros_like(self.action_buffer[:, index]),
                torch.zeros_like(self.action_buffer[:, index]),
                None if names is None else names.copy(),
                torch.zeros_like(self.action_buffer[:, index]),
                dt=dt,
            )
        if self.action_sequence is not None:
            return _select_action_channel(self.action_sequence, index)
        if index == 0 and self.next_action is not None:
            return self.next_action.clone()
        raise ValueError("no materialized MPC action sequence is available")

    def successful(self) -> "MPCSolverResult":
        """Return only successful batch rows, retaining batched result ranks."""
        batch = self.result_batch_size
        if batch < 1 or self.success.ndim < 1 or self.success.shape[0] != batch:
            raise ValueError("success must have a leading result batch dimension")
        success = self.success.reshape(batch, -1).all(dim=1)
        return self.select_batch(success.nonzero(as_tuple=False).flatten())

    def _copy_mpc_at_batch_indices(self, other: "MPCSolverResult", mask: torch.Tensor) -> None:
        """Copy MPC-only plan/action payloads selected by a batch mask."""
        _copy_batch_joint_state(self.next_action, other.next_action, mask, "next_action")
        _copy_batch_joint_state(self.action_sequence, other.action_sequence, mask, "action_sequence")
        _copy_batch_joint_state(
            self.full_action_sequence, other.full_action_sequence, mask, "full_action_sequence"
        )
        _copy_batch_robot_state(self.robot_state_sequence, other.robot_state_sequence, mask)
        _copy_batch_tensor(self.action_buffer, other.action_buffer, mask, "action_buffer")

    def copy_at_batch_indices(self, other: "MPCSolverResult", mask: torch.Tensor) -> None:
        """Copy complete materialised MPC result rows, including actions."""
        if not isinstance(other, MPCSolverResult):
            raise TypeError("other must be an MPCSolverResult")
        super().copy_at_batch_indices(other, mask)
        self._copy_mpc_at_batch_indices(other, mask)

    def copy_successful_solutions(self, other: "MPCSolverResult") -> None:
        """Merge successful MPC rows, including current and full action plans."""
        if not isinstance(other, MPCSolverResult):
            raise TypeError("other must be an MPCSolverResult")
        if other.success.ndim != 1:
            raise ValueError("MPC success must have shape [batch] for action-plan merging")
        super().copy_successful_solutions(other)
        self._copy_mpc_at_batch_indices(other, other.success)


@dataclass
class MPCSolverResult(_MPCSolverResultPortableMixin, BaseSolverResult):
    next_action: Optional[JointState] = None
    action_sequence: Optional[JointState] = None
    full_action_sequence: Optional[JointState] = None
    robot_state_sequence: Optional[RobotState] = None
    action_buffer: Optional[torch.Tensor] = None
    action_dt: Optional[float] = None

    def clone(self) -> MPCSolverResult:
        return _MPCSolverResultPortableMixin.clone(self)


__all__ = ["MPCSolverResult"]

"""Shared portable lifecycle for C-space rollout costs.

The production evaluators use regular PyTorch operations on CPU and MPS, but
callers also use this public base class to allocate and validate a rollout
before choosing a position or state implementation.  This module therefore
preserves the useful V2 lifecycle invariants without pretending to expose the
CUDA/Warp launch workspace.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import init_warp

from .cost_base import BaseCost
from .portable import BaseCSpaceCost as _PortableBaseCSpaceCost

if TYPE_CHECKING:
    from .cost_cspace_cfg import CSpaceCostCfg


class _BaseCSpaceCostPortableMixin(_PortableBaseCSpaceCost):
    """Validated C-space cost lifecycle usable on CPU and Apple Metal.

    Upstream allocates a CUDA-side copy of the target weight and starts that
    term disabled.  The target copy below deliberately remains a normal torch
    tensor: it has the same enable/disable and buffer-reuse semantics while
    retaining autograd and avoiding a false CUDA graph compatibility claim.
    Concrete portable position/state costs retain their composed-PyTorch
    forward implementations in :mod:`curobo._src.cost.portable`.
    """

    def __init__(self, config: "CSpaceCostCfg") -> None:
        if not hasattr(config, "dof") or isinstance(config.dof, bool) or config.dof < 0:
            raise ValueError("dof must be a non-negative integer")
        if getattr(config, "joint_limits", None) is None:
            raise ValueError("joint_limits must be set before creating BaseCSpaceCost")
        validate_shape = getattr(config.joint_limits, "validate_shape", None)
        if callable(validate_shape):
            validate_shape(config.dof, check_effort=True)
        super().__init__(config)
        self._cspace_target_weight = config.cspace_target_weight.detach().clone()
        self.disable_cspace_target()

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> bool:
        if isinstance(batch_size, bool) or isinstance(horizon, bool):
            raise TypeError("batch_size and horizon must be integers")
        if not isinstance(batch_size, int) or not isinstance(horizon, int):
            raise TypeError("batch_size and horizon must be integers")
        if batch_size < 0 or horizon < 0:
            raise ValueError("batch_size and horizon must be non-negative")
        return bool(super().setup_batch_tensors(batch_size, horizon))

    def enable_cspace_target(self) -> None:
        self._cspace_target_weight.copy_(self.config.cspace_target_weight)
        self.cspace_target_enabled = bool(torch.any(self._cspace_target_weight != 0).item())

    def disable_cspace_target(self) -> None:
        self._cspace_target_weight.zero_()
        self.cspace_target_enabled = False

    @property
    def cspace_target_weight(self) -> torch.Tensor:
        """The mutable, evaluator-owned target-weight buffer.

        The configuration remains reusable; users that inspect this property
        after disable/enable observe the current rollout state, not config
        mutation.
        """
        return self._cspace_target_weight

    def validate_input(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> bool:
        del args, kwargs
        if not isinstance(state_batch, JointState):
            raise TypeError("state_batch must be a JointState")
        position = state_batch.position
        if not isinstance(position, torch.Tensor) or position.ndim != 3:
            raise ValueError("state_batch.position must have shape [batch, horizon, dof]")
        batch, horizon, dof = position.shape
        if dof != self.config.dof:
            raise ValueError(f"state_batch dof mismatch: {dof} != {self.config.dof}")
        if self._batch_size >= 0 and batch != self._batch_size:
            raise ValueError(f"state_batch batch size mismatch: {batch} != {self._batch_size}")
        if self._horizon >= 0 and horizon != self._horizon:
            raise ValueError(f"state_batch horizon mismatch: {horizon} != {self._horizon}")
        if not self.device_cfg.is_same_torch_device(position.device):
            raise ValueError("state_batch must reside on config.device_cfg.device")
        if joint_torque is not None:
            if not isinstance(joint_torque, torch.Tensor) or joint_torque.shape != position.shape:
                raise ValueError("joint_torque must have shape [batch, horizon, dof]")
            if joint_torque.device != position.device or joint_torque.dtype != position.dtype:
                raise ValueError("joint_torque must share state_batch device and dtype")
        if target_joint_state is not None:
            if not isinstance(target_joint_state, JointState):
                raise TypeError("target_joint_state must be a JointState")
            target = target_joint_state.position
            if target.ndim not in (2, 3) or target.shape[-1] != dof:
                raise ValueError("target_joint_state.position must end in the configured dof")
            if target.device != position.device or target.dtype != position.dtype:
                raise ValueError("target_joint_state must share state_batch device and dtype")
        if idxs_target_joint_state is not None:
            if not isinstance(idxs_target_joint_state, torch.Tensor):
                raise TypeError("idxs_target_joint_state must be a tensor")
            if idxs_target_joint_state.ndim != 1 or idxs_target_joint_state.shape[0] != batch:
                raise ValueError("idxs_target_joint_state must have shape [batch]")
            if idxs_target_joint_state.dtype not in (torch.int32, torch.int64):
                raise TypeError("idxs_target_joint_state must use int32 or int64")
            if idxs_target_joint_state.device != position.device:
                raise ValueError("idxs_target_joint_state must share state_batch device")
            if target_joint_state is None:
                raise ValueError("idxs_target_joint_state requires target_joint_state")
            targets = target_joint_state.position.shape[0]
            if bool(((idxs_target_joint_state < 0) | (idxs_target_joint_state >= targets)).any().item()):
                raise ValueError("idxs_target_joint_state contains an out-of-range target index")
        return True


class BaseCSpaceCost(_BaseCSpaceCostPortableMixin):
    """Pinned C-space base declaration with portable concrete behavior.

    The upstream ``forward`` is abstract because its CUDA/Warp subclasses
    implement it.  This class keeps that declared marker, while delegating to
    the private eager base so the public base remains instantiable for the
    portable lifecycle callers that use it for validation and allocation.
    """

    def __init__(self, config: CSpaceCostCfg):
        _BaseCSpaceCostPortableMixin.__init__(self, config)

    def validate_input(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
    ):
        return _BaseCSpaceCostPortableMixin.validate_input(
            self, state_batch, joint_torque, target_joint_state, idxs_target_joint_state
        )

    @abstractmethod
    def forward(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
    ):
        return _BaseCSpaceCostPortableMixin.forward(
            self,
            state_batch,
            joint_torque,
            target_joint_state,
            idxs_target_joint_state,
            current_joint_state,
            idxs_current_joint_state,
        )

    def enable_cspace_target(self):
        return _BaseCSpaceCostPortableMixin.enable_cspace_target(self)

    def disable_cspace_target(self):
        return _BaseCSpaceCostPortableMixin.disable_cspace_target(self)


__all__ = ["BaseCost", "BaseCSpaceCost", "JointState"]

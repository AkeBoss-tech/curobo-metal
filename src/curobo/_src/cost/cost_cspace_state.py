"""Pinned declaration surface for portable configuration-space state cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_state import StateCSpaceFunction
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise
from .portable import StateCSpaceCost as _StateCSpaceCostPortable


class StateCSpaceCost(BaseCSpaceCost):
    """Pinned cuRoboV2 declaration surface for the portable implementation."""

    def __init__(self, config: CSpaceCostCfg):
        raise NotImplementedError

    def setup_batch_tensors(self, batch: int, horizon: int):
        raise NotImplementedError

    def forward(
        self,
        state_batch: JointState,
        joint_torque: Optional[torch.Tensor] = None,
        target_joint_state: Optional[JointState] = None,
        idxs_target_joint_state: Optional[torch.Tensor] = None,
        current_joint_state: Optional[JointState] = None,
        idxs_current_joint_state: Optional[torch.Tensor] = None,
        current_state_dt: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError


# The public runtime continues to use the CPU/MPS implementation, retaining
# portable state-limit and regularization behavior without a Warp ABI claim.
if not TYPE_CHECKING:
    StateCSpaceCost = _StateCSpaceCostPortable


__all__ = ["BaseCSpaceCost", "CSpaceCostType", "JointState", "StateCSpaceCost", "StateCSpaceFunction"]

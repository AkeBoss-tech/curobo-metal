"""Pinned declaration surface for portable configuration-space position cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.wp_cspace_position import PositionCSpaceFunction
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise
from .portable import PositionCSpaceCost as _PositionCSpaceCostPortable


class PositionCSpaceCost(BaseCSpaceCost):
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


# Runtime uses the CPU/MPS tensor implementation.  It retains additional
# diagnostics and velocity-aware extensions without advertising a Warp ABI.
if not TYPE_CHECKING:
    PositionCSpaceCost = _PositionCSpaceCostPortable


__all__ = ["BaseCSpaceCost", "CSpaceCostType", "JointState", "PositionCSpaceCost", "PositionCSpaceFunction"]

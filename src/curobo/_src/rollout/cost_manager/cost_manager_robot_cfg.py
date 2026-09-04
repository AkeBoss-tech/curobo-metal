"""Portable configuration for robot rollout cost orchestration.

This keeps V2's serializable mapping factory while binding tensors to an
explicit CPU/MPS :class:`DeviceCfg`.  CUDA stream and Warp kernel options
belong to the backend and are intentionally not represented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg as _PublicSelfCollisionCostCfg
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.portable import CSpaceCostCfg as _PortableCSpaceCostCfg
from curobo._src.cost.portable import CSpaceDistCostCfg as _PortableCSpaceDistCostCfg
from curobo._src.cost.portable import SelfCollisionCostCfg
from curobo._src.cost.portable import ToolPoseCostCfg as _PortableToolPoseCostCfg
from curobo._src.types.device_cfg import DeviceCfg

if TYPE_CHECKING:
    from curobo._src.geom.collision.collision_scene import SceneCollision


_COST_CONFIG_TYPES = {
    "self_collision_cfg": SelfCollisionCostCfg,
    "scene_collision_cfg": SceneCollisionCostCfg,
    "cspace_cfg": CSpaceCostCfg,
    "start_cspace_dist_cfg": CSpaceDistCostCfg,
    "target_cspace_dist_cfg": CSpaceDistCostCfg,
    "tool_pose_cfg": ToolPoseCostCfg,
}

# The public self-collision config carries extra portable validation and does
# not subclass the original lightweight record.  Both forms are accepted by
# the cost implementation, so preserve that source-compatible distinction.
# Portable base classes are accepted alongside the validated public subclasses
# so callers can use either form without an unnecessary conversion.
_ACCEPTED_COST_CONFIG_TYPES = {
    **_COST_CONFIG_TYPES,
    # ``portable.CSpaceCostCfg`` is the backend-neutral base record while
    # ``cost_cspace_cfg.CSpaceCostCfg`` is the validated public subclass.
    # RobotCostManager accepts either form so callers can use the portable
    # config directly without an unnecessary conversion.
    "cspace_cfg": (CSpaceCostCfg, _PortableCSpaceCostCfg),
    "self_collision_cfg": (SelfCollisionCostCfg, _PublicSelfCollisionCostCfg),
    "start_cspace_dist_cfg": (CSpaceDistCostCfg, _PortableCSpaceDistCostCfg),
    "target_cspace_dist_cfg": (CSpaceDistCostCfg, _PortableCSpaceDistCostCfg),
    "tool_pose_cfg": (ToolPoseCostCfg, _PortableToolPoseCostCfg),
}


def _config_update_tensor(value, target: torch.Tensor, name: str) -> torch.Tensor:
    """Create an update on the configured device without hidden tensor moves."""
    if isinstance(value, torch.Tensor) and value.device != target.device:
        raise ValueError(f"{name} tensor device must match the configured cost device")
    return torch.as_tensor(value, device=target.device, dtype=target.dtype)


@dataclass
class RobotCostManagerCfg:
    class_type: type = None
    self_collision_cfg: Optional[SelfCollisionCostCfg] = None
    scene_collision_cfg: Optional[SceneCollisionCostCfg] = None
    cspace_cfg: Optional[CSpaceCostCfg] = None
    start_cspace_dist_cfg: Optional[CSpaceDistCostCfg] = None
    target_cspace_dist_cfg: Optional[CSpaceDistCostCfg] = None
    tool_pose_cfg: Optional[ToolPoseCostCfg] = None
    def __post_init__(self):
        from .cost_manager_robot import RobotCostManager
        for name, config_type in _ACCEPTED_COST_CONFIG_TYPES.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, config_type):
                description = (
                    "/".join(item.__name__ for item in config_type)
                    if isinstance(config_type, tuple)
                    else config_type.__name__
                )
                raise TypeError(f"{name} must be a {description}")
        self.class_type = RobotCostManager
    @staticmethod
    def create(
        data_dict: Dict,
        scene_collision_checker: Optional[SceneCollision] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> RobotCostManagerCfg:
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dictionary")
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        values = {}
        for name, kind in _COST_CONFIG_TYPES.items():
            raw = data_dict.get(name)
            if raw is not None:
                if isinstance(raw, _ACCEPTED_COST_CONFIG_TYPES[name]):
                    if not device_cfg.is_same_torch_device(raw.device_cfg.device):
                        raise ValueError(f"{name} device does not match device_cfg")
                    values[name] = raw
                elif isinstance(raw, dict):
                    payload = dict(raw)
                    declared_device_cfg = payload.pop("device_cfg", None)
                    if declared_device_cfg is not None:
                        if not isinstance(declared_device_cfg, DeviceCfg):
                            raise TypeError(f"{name}.device_cfg must be a DeviceCfg")
                        if not device_cfg.is_same_torch_device(declared_device_cfg.device):
                            raise ValueError(f"{name}.device_cfg does not match device_cfg")
                    values[name] = kind(device_cfg=device_cfg, **payload)
                else:
                    raise TypeError(f"{name} must be a dict or {kind.__name__}")
        if values.get("scene_collision_cfg") is not None:
            values["scene_collision_cfg"].scene_collision_checker = scene_collision_checker
        return RobotCostManagerCfg(**values)
    def update_collision_activation_distance(self, distance: float):
        if self.scene_collision_cfg:
            value = self.scene_collision_cfg.activation_distance
            update = _config_update_tensor(distance, value, "collision activation distance")
            if update.numel() not in (1, value.numel()):
                raise ValueError("collision activation distance must be scalar or match configured shape")
            value.copy_(update.reshape(-1).expand_as(value))
        return self
    def disable_self_collision(self):
        if self.self_collision_cfg is not None:
            self.self_collision_cfg.weight.zero_()
        return self
    def update_regularization_weight(
        self, l2_weight: Optional[float] = None, distance_weight: Optional[float] = None
    ):
        """Update c-space L2 and distance weights without rebuilding the rollout."""
        if self.cspace_cfg is not None and l2_weight is not None:
            target = self.cspace_cfg.squared_l2_regularization_weight
            value = _config_update_tensor(l2_weight, target, "l2_weight")
            if value.numel() not in (1, self.cspace_cfg.squared_l2_regularization_weight.numel()):
                raise ValueError("l2_weight must be scalar or match cspace regularization shape")
            self.cspace_cfg.squared_l2_regularization_weight.copy_(value.expand_as(self.cspace_cfg.squared_l2_regularization_weight))
        for cfg in (self.start_cspace_dist_cfg, self.target_cspace_dist_cfg):
            if cfg is not None and distance_weight is not None:
                value = _config_update_tensor(distance_weight, cfg.weight, "distance_weight")
                if value.numel() not in (1, cfg.weight.numel()):
                    raise ValueError("distance_weight must be scalar or match cspace-distance weight shape")
                cfg.weight.copy_(value.expand_as(cfg.weight))
        return self
__all__ = ["RobotCostManagerCfg"]

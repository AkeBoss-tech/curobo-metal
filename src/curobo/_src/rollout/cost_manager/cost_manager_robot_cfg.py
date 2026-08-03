"""Portable configuration for robot rollout cost orchestration.

This keeps V2's serializable mapping factory while binding tensors to an
explicit CPU/MPS :class:`DeviceCfg`.  CUDA stream and Warp kernel options
belong to the backend and are intentionally not represented here.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from curobo._src.cost.portable import *
from curobo._src.types.device_cfg import DeviceCfg
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
        self.class_type = RobotCostManager
    @classmethod
    def create(cls, data_dict: Dict, scene_collision_checker=None, device_cfg: DeviceCfg = DeviceCfg()):
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dictionary")
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        mapping = {"self_collision_cfg": SelfCollisionCostCfg, "scene_collision_cfg": SceneCollisionCostCfg,
                   "cspace_cfg": CSpaceCostCfg, "start_cspace_dist_cfg": CSpaceDistCostCfg,
                   "target_cspace_dist_cfg": CSpaceDistCostCfg, "tool_pose_cfg": ToolPoseCostCfg}
        values = {}
        for name, kind in mapping.items():
            raw = data_dict.get(name)
            if raw is not None:
                if isinstance(raw, kind):
                    if not device_cfg.is_same_torch_device(raw.device_cfg.device):
                        raise ValueError(f"{name} device does not match device_cfg")
                    values[name] = raw
                elif isinstance(raw, dict):
                    values[name] = kind(device_cfg=device_cfg, **raw)
                else:
                    raise TypeError(f"{name} must be a dict or {kind.__name__}")
        if values.get("scene_collision_cfg") is not None:
            values["scene_collision_cfg"].scene_collision_checker = scene_collision_checker
        return cls(**values)
    def update_collision_activation_distance(self, distance):
        if self.scene_collision_cfg:
            value = self.scene_collision_cfg.activation_distance
            update = torch.as_tensor(distance, device=value.device, dtype=value.dtype)
            if update.numel() not in (1, value.numel()):
                raise ValueError("collision activation distance must be scalar or match configured shape")
            value.copy_(update.reshape(-1).expand_as(value))
        return self
    def disable_self_collision(self):
        if self.self_collision_cfg is not None:
            self.self_collision_cfg.weight.zero_()
        return self
    def update_regularization_weight(self, l2_weight=None, distance_weight=None):
        """Update c-space L2 and distance weights without rebuilding the rollout."""
        if self.cspace_cfg is not None and l2_weight is not None:
            value = torch.as_tensor(l2_weight, device=self.cspace_cfg.squared_l2_regularization_weight.device,
                                    dtype=self.cspace_cfg.squared_l2_regularization_weight.dtype)
            if value.numel() not in (1, self.cspace_cfg.squared_l2_regularization_weight.numel()):
                raise ValueError("l2_weight must be scalar or match cspace regularization shape")
            self.cspace_cfg.squared_l2_regularization_weight.copy_(value.expand_as(self.cspace_cfg.squared_l2_regularization_weight))
        for cfg in (self.start_cspace_dist_cfg, self.target_cspace_dist_cfg):
            if cfg is not None and distance_weight is not None:
                value = torch.as_tensor(distance_weight, device=cfg.weight.device, dtype=cfg.weight.dtype)
                if value.numel() not in (1, cfg.weight.numel()):
                    raise ValueError("distance_weight must be scalar or match cspace-distance weight shape")
                cfg.weight.copy_(value.expand_as(cfg.weight))
        return self
__all__ = ["RobotCostManagerCfg"]

from dataclasses import dataclass
from typing import Optional
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
    @classmethod
    def create(cls, data_dict, scene_collision_checker=None, device_cfg=DeviceCfg()):
        mapping = {"self_collision_cfg": SelfCollisionCostCfg, "scene_collision_cfg": SceneCollisionCostCfg,
                   "cspace_cfg": CSpaceCostCfg, "start_cspace_dist_cfg": CSpaceDistCostCfg,
                   "target_cspace_dist_cfg": CSpaceDistCostCfg, "tool_pose_cfg": ToolPoseCostCfg}
        values = {}
        for name, kind in mapping.items():
            raw = data_dict.get(name)
            if raw is not None:
                values[name] = kind(device_cfg=device_cfg, **raw)
        if values.get("scene_collision_cfg") is not None:
            values["scene_collision_cfg"].scene_collision_checker = scene_collision_checker
        return cls(**values)
    def update_collision_activation_distance(self, distance):
        if self.scene_collision_cfg: self.scene_collision_cfg.activation_distance = distance
    def disable_self_collision(self): self.self_collision_cfg = None
    def update_regularization_weight(self, l2_weight=None, distance_weight=None): return self
__all__ = ["RobotCostManagerCfg"]

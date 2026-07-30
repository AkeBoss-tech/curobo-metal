from dataclasses import dataclass
from typing import Optional, Type
from curobo._src.types.device_cfg import DeviceCfg
from .cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
@dataclass
class RobotRolloutCfg:
    device_cfg: DeviceCfg
    sum_horizon: bool = False
    sampler_seed: int = 1312
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg
    transition_model_config_instance_type: type = object
    transition_model_cfg: Optional[object] = None
    cost_cfg: Optional[RobotCostManagerCfg] = None
    constraint_cfg: Optional[RobotCostManagerCfg] = None
    hybrid_cost_constraint_cfg: Optional[RobotCostManagerCfg] = None
    convergence_cfg: Optional[RobotCostManagerCfg] = None
    scene_collision_cfg: Optional[object] = None
    @classmethod
    def create_with_component_types(cls, data_dict, robot_cfg, device_cfg=DeviceCfg(),
                                    transition_model_config_instance_type=object,
                                    cost_manager_config_instance_type=RobotCostManagerCfg):
        return cls(device_cfg=device_cfg,
                   sum_horizon=data_dict.get("sum_horizon", False),
                   sampler_seed=data_dict.get("sampler_seed", 1312),
                   transition_model_config_instance_type=transition_model_config_instance_type,
                   cost_manager_config_instance_type=cost_manager_config_instance_type)
    def get_cost_manager_configs(self, include_constraint_cfg=True):
        values=[self.cost_cfg, self.hybrid_cost_constraint_cfg, self.convergence_cfg]
        if include_constraint_cfg: values.insert(1,self.constraint_cfg)
        return [x for x in values if x is not None]
__all__=["RobotRolloutCfg"]

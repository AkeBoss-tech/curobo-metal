"""Portable subset of cuRoboV2 solver-core configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


@dataclass
class SolverCoreCfg:
    robot_config: Any
    device_cfg: DeviceCfg = DeviceCfg()
    optimizer_configs: List[Any] = field(default_factory=list)
    optimizer_rollout_configs: List[Any] = field(default_factory=list)
    metrics_rollout_config: Any = None
    scene_collision_cfg: Any = None
    use_cuda_graph: bool = False
    random_seed: int = 123
    store_debug: bool = False

    def __post_init__(self):
        if self.use_cuda_graph:
            raise NotImplementedError(
                "CUDA Graph capture has no Metal equivalent; set use_cuda_graph=False"
            )


def resolve_yaml_configs(
    robot: Union[str, Dict[str, Any], RobotCfg],
    optimizer_configs: List[Union[str, Dict[str, Any]]],
    metrics_rollout: Union[str, Dict[str, Any]],
    transition_model: Union[str, Dict[str, Any]],
    scene_model: Optional[Union[str, Dict[str, Any]]],
    device_cfg: DeviceCfg,
    load_collision_spheres: bool = True,
    num_envs: int = 1,
) -> Tuple[RobotCfg, List[Dict], Dict, Dict, Optional[Dict]]:
    del num_envs
    if isinstance(robot, RobotCfg):
        robot_cfg = robot
    else:
        kin = KinematicsCfg.from_robot_yaml_file(
            robot, device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres,
        )
        robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
    optimizers = [
        dict(item) if isinstance(item, dict) else {"config": item}
        for item in optimizer_configs
    ]
    metrics = dict(metrics_rollout) if isinstance(metrics_rollout, dict) else {
        "config": metrics_rollout
    }
    transition = dict(transition_model) if isinstance(transition_model, dict) else {
        "config": transition_model
    }
    scene = None if scene_model is None else (
        dict(scene_model) if isinstance(scene_model, dict) else {"config": scene_model}
    )
    return robot_cfg, optimizers, metrics, transition, scene


def create_scene_collision_cfg(
    scene_model_dict: Optional[Dict],
    collision_cache: Optional[Dict[str, int]],
    device_cfg: DeviceCfg,
):
    del collision_cache, device_cfg
    return scene_model_dict


def create_rollout_configs(
    optimization_dicts: List[Dict], transition_model_dict: Dict,
    robot_config: RobotCfg, device_cfg: DeviceCfg,
    optimizer_collision_activation_distance: Optional[float],
    transition_model_config_instance_type: Type = object,
    cost_manager_config_instance_type: Type = object,
    self_collision_check: bool = True,
) -> List[Dict]:
    del (
        transition_model_config_instance_type,
        cost_manager_config_instance_type,
    )
    return [
        {
            **item, "transition_model": transition_model_dict,
            "robot_config": robot_config, "device_cfg": device_cfg,
            "collision_activation_distance": optimizer_collision_activation_distance,
            "self_collision_check": self_collision_check,
        }
        for item in optimization_dicts
    ]


def create_metrics_rollout_config(
    metrics_rollout_dict: Dict, transition_model_dict: Dict,
    robot_config: RobotCfg, device_cfg: DeviceCfg,
    transition_model_config_instance_type: Type = object,
    cost_manager_config_instance_type: Type = object,
):
    del transition_model_config_instance_type, cost_manager_config_instance_type
    return {
        **metrics_rollout_dict, "transition_model": transition_model_dict,
        "robot_config": robot_config, "device_cfg": device_cfg,
    }


def create_solver_core_cfg(
    robot_config: RobotCfg, optimizer_dicts: List[Dict],
    metrics_rollout_dict: Dict, transition_model_dict: Dict,
    scene_model_dict: Optional[Dict], device_cfg: DeviceCfg,
    collision_cache: Optional[Dict[str, int]] = None,
    self_collision_check: bool = True,
    optimizer_collision_activation_distance: float = 0.01,
    use_cuda_graph: bool = True, random_seed: int = 123,
    store_debug: bool = False,
    override_optimizer_num_iters: Optional[Dict[str, Optional[int]]] = None,
    transition_model_config_instance_type: Type = object,
    cost_manager_config_instance_type: Type = object,
) -> SolverCoreCfg:
    del override_optimizer_num_iters
    rollout = create_rollout_configs(
        optimizer_dicts, transition_model_dict, robot_config, device_cfg,
        optimizer_collision_activation_distance,
        transition_model_config_instance_type,
        cost_manager_config_instance_type, self_collision_check,
    )
    metrics = create_metrics_rollout_config(
        metrics_rollout_dict, transition_model_dict, robot_config, device_cfg,
        transition_model_config_instance_type,
        cost_manager_config_instance_type,
    )
    return SolverCoreCfg(
        robot_config, device_cfg, optimizer_dicts, rollout, metrics,
        create_scene_collision_cfg(scene_model_dict, collision_cache, device_cfg),
        use_cuda_graph=False if use_cuda_graph else False,
        random_seed=random_seed, store_debug=store_debug,
    )


__all__ = [
    "SolverCoreCfg", "resolve_yaml_configs", "create_scene_collision_cfg",
    "create_rollout_configs", "create_metrics_rollout_config",
    "create_solver_core_cfg",
]

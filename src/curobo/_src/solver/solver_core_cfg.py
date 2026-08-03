"""Portable solver-core configuration and YAML assembly helpers.

The NVIDIA implementation assembles CUDA rollouts and graphs from a group of
YAML files.  This module retains that useful configuration boundary but emits
ordinary CPU/MPS rollout records.  ``use_cuda_graph`` remains an accepted
upstream option so applications do not need a platform branch; it is recorded
and disabled rather than being mistaken for an eager graph implementation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.optim.optim_factory import create_optimization_config
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo._src.util.logging import log_and_raise, log_warn
from curobo.content import (
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)


@dataclass
class SolverCoreCfg:
    # ``robot_config`` was carried directly by early portable releases.  Keep
    # it first for source compatibility while exposing the same rollout fields
    # consumed by the pinned V2 SolverCore.
    robot_config: Any
    device_cfg: DeviceCfg = DeviceCfg()
    optimizer_configs: List[Any] = field(default_factory=list)
    optimizer_rollout_configs: List[Any] = field(default_factory=list)
    metrics_rollout_config: Any = None
    scene_collision_cfg: Any = None
    use_cuda_graph: bool = False
    random_seed: int = 123
    store_debug: bool = False
    requested_use_cuda_graph: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be DeviceCfg")
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be RobotCfg")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if not isinstance(self.use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        if not isinstance(self.store_debug, bool):
            raise TypeError("store_debug must be a bool")
        if self.optimizer_configs is None:
            self.optimizer_configs = []
        if not isinstance(self.optimizer_configs, list):
            raise TypeError("optimizer_configs must be a list")
        if self.optimizer_rollout_configs is None:
            self.optimizer_rollout_configs = []
        if not isinstance(self.optimizer_rollout_configs, list):
            raise TypeError("optimizer_rollout_configs must be a list")
        if self.scene_collision_cfg is not None and not isinstance(
            self.scene_collision_cfg, SceneCollisionCfg
        ):
            # A pre-assembled SceneCfg is a common direct-construction form.
            self.scene_collision_cfg = create_scene_collision_cfg(
                self.scene_collision_cfg, None, self.device_cfg
            )
        self.requested_use_cuda_graph = bool(self.use_cuda_graph)
        # CUDA graph capture is a device ABI, not a performance hint.  Keep
        # normal high-level configurations executable on Metal while exposing
        # explicit failure through SolverCore.reset_cuda_graph().
        self.use_cuda_graph = False

    def clone(self, **updates: Any) -> "SolverCoreCfg":
        """Return an independent portable configuration copy.

        Solver facades regularly adjust a seed or scene setting between
        solves.  Deep-copy transport records here so a clone cannot mutate the
        rollout/optimizer configuration owned by its source core.
        """
        valid = {
            "robot_config", "device_cfg", "optimizer_configs",
            "optimizer_rollout_configs", "metrics_rollout_config",
            "scene_collision_cfg", "use_cuda_graph", "random_seed", "store_debug",
        }
        unknown = set(updates).difference(valid)
        if unknown:
            raise TypeError(f"unknown SolverCoreCfg field(s): {sorted(unknown)}")
        values = {
            "robot_config": self.robot_config,
            "device_cfg": self.device_cfg,
            "optimizer_configs": deepcopy(self.optimizer_configs),
            "optimizer_rollout_configs": deepcopy(self.optimizer_rollout_configs),
            "metrics_rollout_config": deepcopy(self.metrics_rollout_config),
            "scene_collision_cfg": deepcopy(self.scene_collision_cfg),
            # Preserve user intent, not the eagerly compiled false value.
            "use_cuda_graph": self.requested_use_cuda_graph,
            "random_seed": self.random_seed,
            "store_debug": self.store_debug,
        }
        values.update(updates)
        return SolverCoreCfg(**values)


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
    if isinstance(robot, RobotCfg):
        robot_cfg = robot
    elif isinstance(robot, str):
        # KinematicsCfg resolves bundled robot-relative URDF/XRDF resources;
        # passing the parsed YAML directly to RobotCfg would make those paths
        # relative to the application's working directory.
        kin = KinematicsCfg.from_robot_yaml_file(
            robot, device_cfg=device_cfg, load_collision_spheres=load_collision_spheres
        )
        robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
    else:
        robot_value = resolve_config(join_path(get_robot_configs_path(), robot))
        robot_cfg = RobotCfg.create(
            robot_value, device_cfg=device_cfg,
            load_collision_spheres=load_collision_spheres, num_envs=num_envs,
        )
    optimizers = [
        resolve_config(join_path(get_task_configs_path(), item))
        for item in optimizer_configs
    ]
    metrics = resolve_config(join_path(get_task_configs_path(), metrics_rollout))
    transition = resolve_config(join_path(get_task_configs_path(), transition_model))
    scene = None if scene_model is None else resolve_config(
        join_path(get_scene_configs_path(), scene_model)
    )
    return robot_cfg, optimizers, metrics, transition, scene


def create_scene_collision_cfg(
    scene_model_dict: Optional[Dict],
    collision_cache: Optional[Dict[str, int]],
    device_cfg: DeviceCfg,
):
    if scene_model_dict is None:
        return None
    if isinstance(scene_model_dict, SceneCollisionCfg):
        return scene_model_dict
    if isinstance(scene_model_dict, list):
        model = [item if isinstance(item, SceneCfg) else SceneCfg.create(item)
                 for item in scene_model_dict]
    elif isinstance(scene_model_dict, SceneCfg):
        model = scene_model_dict
    elif isinstance(scene_model_dict, dict):
        model = SceneCfg.create(scene_model_dict)
    else:
        raise TypeError("scene_model_dict must be SceneCfg, mapping, list, or None")
    return SceneCollisionCfg(device_cfg=device_cfg, scene_model=model, cache=collision_cache)


def create_rollout_configs(
    optimization_dicts: List[Dict], transition_model_dict: Dict,
    robot_config: RobotCfg, device_cfg: DeviceCfg,
    optimizer_collision_activation_distance: Optional[float],
    transition_model_config_instance_type: Type = object,
    cost_manager_config_instance_type: Type = object,
    self_collision_check: bool = True,
) -> List[Dict]:
    result = []
    for item in optimization_dicts:
        if not isinstance(item, dict):
            raise TypeError("each optimizer configuration must resolve to a mapping")
        rollout = dict(item.get("rollout", item.get("rollout_cfg", {})))
        transition = transition_model_dict.get("transition_model_cfg", transition_model_dict)
        rollout["transition_model_cfg"] = transition
        cfg = RobotRolloutCfg.create_with_component_types(
            rollout, robot_config, device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )
        # Preserve these values on the record even where an eager portable
        # rollout has no matching packed CUDA cost buffer.
        cfg.collision_activation_distance = optimizer_collision_activation_distance
        cfg.self_collision_check = bool(self_collision_check)
        result.append(cfg)
    return result


def create_metrics_rollout_config(
    metrics_rollout_dict: Dict, transition_model_dict: Dict,
    robot_config: RobotCfg, device_cfg: DeviceCfg,
    transition_model_config_instance_type: Type = object,
    cost_manager_config_instance_type: Type = object,
):
    if not isinstance(metrics_rollout_dict, dict):
        raise TypeError("metrics_rollout_dict must resolve to a mapping")
    rollout = dict(metrics_rollout_dict.get("rollout", metrics_rollout_dict))
    rollout["transition_model_cfg"] = transition_model_dict.get(
        "transition_model_cfg", transition_model_dict
    )
    return RobotRolloutCfg.create_with_component_types(
        rollout, robot_config, device_cfg,
        transition_model_config_instance_type=transition_model_config_instance_type,
        cost_manager_config_instance_type=cost_manager_config_instance_type,
    )


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
    optimizer_cfgs = []
    for item in optimizer_dicts:
        if not isinstance(item, dict):
            raise TypeError("optimizer_dicts must contain mappings")
        payload = item.get("optimizer", item)
        value = create_optimization_config(payload, device_cfg)
        if override_optimizer_num_iters is not None:
            override = override_optimizer_num_iters.get(value.solver_name)
            if override is not None:
                value.update_niters(override)
        value.store_debug = bool(store_debug)
        optimizer_cfgs.append(value)
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
        robot_config, device_cfg, optimizer_cfgs, rollout, metrics,
        create_scene_collision_cfg(scene_model_dict, collision_cache, device_cfg),
        use_cuda_graph=use_cuda_graph,
        random_seed=random_seed, store_debug=store_debug,
    )


__all__ = [
    "SolverCoreCfg", "resolve_yaml_configs", "create_scene_collision_cfg",
    "create_rollout_configs", "create_metrics_rollout_config",
    "create_solver_core_cfg",
]

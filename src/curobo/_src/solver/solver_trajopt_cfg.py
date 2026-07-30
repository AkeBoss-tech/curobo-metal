"""Trajectory optimizer configuration at the pinned import path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType


@dataclass
class TrajOptSolverCfg:
    core_cfg: Any
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.0
    self_collision_check: bool = True
    minimum_trajectory_dt: float = 0.002
    maximum_trajectory_dt: float = 0.2
    interpolation_dt: float = 0.025
    interpolation_type: TrajInterpolationType = TrajInterpolationType.BSPLINE_KNOTS_CUDA
    interpolation_buffer_size: int = 1000
    # Portable controls corresponding to the compiled optimizer configuration.
    action_horizon: int = 32
    max_iterations: int = 200
    optimizer_name: str = "adam"
    use_cuda_graph_value: bool = True
    random_seed_value: int = 123
    store_debug_value: bool = False

    @property
    def device_cfg(self): return self.robot_config.device_cfg
    @property
    def use_cuda_graph(self): return self.use_cuda_graph_value
    @property
    def random_seed(self): return self.random_seed_value
    @property
    def store_debug(self): return self.store_debug_value
    @property
    def scene_collision_cfg(self): return None
    @property
    def optimizer_configs(self): return self.core_cfg
    @property
    def optimizer_rollout_configs(self): return []
    @property
    def metrics_rollout_config(self): return None

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 4,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        minimum_trajectory_dt: float = 0.002,
        maximum_trajectory_dt: float = 0.2,
        load_collision_spheres: bool = True,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None},
        transition_model_config_instance_type: Type = object,
        cost_manager_config_instance_type: Type = object,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: int = 1000,
    ):
        del metrics_rollout, transition_model, scene_model, collision_cache
        del transition_model_config_instance_type, cost_manager_config_instance_type
        robot_cfg = robot if isinstance(robot, RobotCfg) else RobotCfg.create(
            robot, device_cfg, load_collision_spheres
        )
        iteration = override_optimizer_num_iters.get("lbfgs")
        return TrajOptSolverCfg(
            optimizer_configs, robot_cfg, max_batch_size, multi_env, max_goalset,
            num_seeds, position_tolerance, orientation_tolerance,
            optimizer_collision_activation_distance, 0.0, self_collision_check,
            minimum_trajectory_dt, maximum_trajectory_dt, interpolation_dt,
            TrajInterpolationType.BSPLINE_KNOTS_CUDA, interpolation_buffer_size,
            max_iterations=200 if iteration is None else iteration,
            optimizer_name="lbfgs" if any("lbfgs" in str(x).lower() for x in optimizer_configs) else "adam",
            use_cuda_graph_value=use_cuda_graph, random_seed_value=random_seed,
            store_debug_value=store_debug,
        )


__all__ = ["TrajOptSolverCfg"]

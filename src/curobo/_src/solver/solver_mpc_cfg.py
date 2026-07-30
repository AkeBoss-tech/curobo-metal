"""Portable pinned MPC configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg

from .solver_core_cfg import SolverCoreCfg


@dataclass
class MPCSolverCfg:
    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 1
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.001
    self_collision_check: bool = True
    interpolation_steps: int = 4
    optimization_dt: float = 0.02
    warm_start_optimization_num_iters: int = 200
    cold_start_optimization_num_iters: int = 300
    use_deceleration_on_failure: bool = True
    deceleration_time: Optional[float] = None
    deceleration_profile: str = "exponential"
    max_deceleration_time: float = 2.0

    device_cfg = property(lambda self: self.core_cfg.device_cfg)
    use_cuda_graph = property(lambda self: self.core_cfg.use_cuda_graph)
    random_seed = property(lambda self: self.core_cfg.random_seed)
    store_debug = property(lambda self: self.core_cfg.store_debug)
    scene_collision_cfg = property(lambda self: self.core_cfg.scene_collision_cfg)
    optimizer_configs = property(lambda self: self.core_cfg.optimizer_configs)
    optimizer_rollout_configs = property(lambda self: self.core_cfg.optimizer_rollout_configs)
    metrics_rollout_config = property(lambda self: self.core_cfg.metrics_rollout_config)

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["mpc/lbfgs_mpc.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "mpc/transition_bspline_mpc.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None},
        transition_model_config_instance_type: Type = object,
        cost_manager_config_instance_type: Type = object,
        optimization_dt: float = 0.02,
        interpolation_steps: int = 4,
        use_deceleration_on_failure: bool = True,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
        max_deceleration_time: float = 2.0,
        load_collision_spheres: bool = True,
        num_control_points: Optional[int] = None,
        squared_l2_regularization_weight: Optional[List[float]] = None,
        warm_start_optimization_num_iters: int = 200,
        cold_start_optimization_num_iters: int = 300,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        **kwargs,
    ):
        del (
            metrics_rollout, transition_model, collision_cache,
            override_optimizer_num_iters, transition_model_config_instance_type,
            cost_manager_config_instance_type, num_control_points,
            squared_l2_regularization_weight,
        )
        if kwargs:
            raise TypeError(f"unsupported MPC configuration fields: {sorted(kwargs)}")
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        else:
            kin = KinematicsCfg.from_robot_yaml_file(
                robot, device_cfg=device_cfg,
                load_collision_spheres=load_collision_spheres,
            )
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        # The user-facing upstream default remains accepted, but execution uses
        # a persistent portable cache rather than NVIDIA CUDA Graph capture.
        core = SolverCoreCfg(
            robot_cfg, device_cfg, list(optimizer_configs),
            scene_collision_cfg=scene_model, use_cuda_graph=False,
            random_seed=random_seed, store_debug=store_debug,
        )
        return MPCSolverCfg(
            core, robot_cfg, max_batch_size, multi_env, max_goalset,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            self_collision_check=self_collision_check,
            interpolation_steps=interpolation_steps,
            optimization_dt=optimization_dt,
            warm_start_optimization_num_iters=warm_start_optimization_num_iters,
            cold_start_optimization_num_iters=cold_start_optimization_num_iters,
            use_deceleration_on_failure=use_deceleration_on_failure,
            deceleration_time=deceleration_time,
            deceleration_profile=deceleration_profile,
            max_deceleration_time=max_deceleration_time,
        )


__all__ = ["MPCSolverCfg"]

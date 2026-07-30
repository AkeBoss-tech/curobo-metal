"""Pinned IK configuration with portable CPU/MPS defaults."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg

from .solver_core_cfg import SolverCoreCfg


@dataclass
class IKSolverCfg:
    core_cfg: SolverCoreCfg
    robot_config: RobotCfg
    max_batch_size: int = 1
    multi_env: bool = False
    max_goalset: int = 1
    num_seeds: int = 32
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    optimizer_collision_activation_distance: float = 0.01
    non_terminal_tool_pose_weight_factor: float = 0.0
    success_requires_convergence: bool = True
    override_iters_for_multi_link_ik: Optional[int] = None
    use_lm_seed: bool = True
    exit_early: bool = True
    exit_early_batch_success_threshold: float = 1.0
    optimization_dt: Optional[float] = None
    seed_position_weight: float = 1.0
    seed_orientation_weight: float = 1.0
    seed_velocity_weight: float = 0.0
    seed_acceleration_weight: float = 0.0
    seed_solver_num_seeds: int = 32
    self_collision_check: bool = True

    @property
    def device_cfg(self): return self.core_cfg.device_cfg
    @property
    def use_cuda_graph(self): return self.core_cfg.use_cuda_graph
    @property
    def random_seed(self): return self.core_cfg.random_seed
    @property
    def store_debug(self): return self.core_cfg.store_debug
    @property
    def scene_collision_cfg(self): return self.core_cfg.scene_collision_cfg
    @property
    def optimizer_configs(self): return self.core_cfg.optimizer_configs
    @property
    def optimizer_rollout_configs(self): return self.core_cfg.optimizer_rollout_configs
    @property
    def metrics_rollout_config(self): return self.core_cfg.metrics_rollout_config

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 32,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {"particle": None, "lbfgs": None},
        transition_model_config_instance_type: Type = object,
        cost_manager_config_instance_type: Type = object,
        override_iters_for_multi_link_ik: Optional[int] = None,
        optimization_dt: Optional[float] = None,
        load_collision_spheres: bool = True,
        velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None,
        success_requires_convergence: bool = True,
        seed_position_weight: float = 1.0,
        seed_orientation_weight: float = 1.0,
        seed_velocity_weight: float = 0.0,
        seed_acceleration_weight: float = 0.0,
        seed_solver_num_seeds: int = 32,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
    ):
        del metrics_rollout, transition_model, collision_cache, override_optimizer_num_iters
        del transition_model_config_instance_type, cost_manager_config_instance_type
        del velocity_regularization_weight, acceleration_regularization_weight
        if use_cuda_graph:
            use_cuda_graph = False  # dependency-name replacement keeps upstream default usable
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        else:
            kin = KinematicsCfg.from_robot_yaml_file(
                robot, device_cfg=device_cfg, load_collision_spheres=load_collision_spheres
            )
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        core = SolverCoreCfg(
            robot_cfg, device_cfg, list(optimizer_configs), scene_collision_cfg=scene_model,
            use_cuda_graph=use_cuda_graph, random_seed=random_seed, store_debug=store_debug,
        )
        return IKSolverCfg(
            core, robot_cfg, max_batch_size, multi_env, max_goalset, num_seeds,
            position_tolerance, orientation_tolerance,
            optimizer_collision_activation_distance,
            success_requires_convergence=success_requires_convergence,
            override_iters_for_multi_link_ik=override_iters_for_multi_link_ik,
            optimization_dt=optimization_dt,
            seed_position_weight=seed_position_weight,
            seed_orientation_weight=seed_orientation_weight,
            seed_velocity_weight=seed_velocity_weight,
            seed_acceleration_weight=seed_acceleration_weight,
            seed_solver_num_seeds=seed_solver_num_seeds,
            self_collision_check=self_collision_check,
        )


__all__ = ["IKSolverCfg"]

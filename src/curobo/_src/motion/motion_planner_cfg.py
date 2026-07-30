"""Pinned MotionPlanner configuration compiled to portable solver configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


@dataclass
class MotionPlannerCfg:
    ik_solver_config: IKSolverCfg
    trajopt_solver_config: TrajOptSolverCfg
    graph_planner_config: Optional[PRMGraphPlannerCfg] = None
    scene_collision_cfg: Optional[object] = None
    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"][1:],
        ik_transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        trajopt_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"],
        trajopt_transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        graph_planner_config: Union[str, Dict[str, Any], None] = "graph_planner/exact_graph_planner.yml",
        graph_planner_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        graph_planner_transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        transition_model_config_instance_type: Type = object,
        cost_manager_config_instance_type: Type = object,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: int = 1000,
    ) -> "MotionPlannerCfg":
        del (
            ik_transition_model, metrics_rollout, trajopt_transition_model,
            graph_planner_rollout, graph_planner_transition_model, collision_cache,
            transition_model_config_instance_type, cost_manager_config_instance_type,
        )
        if isinstance(robot, RobotCfg):
            robot_cfg = robot
        else:
            kin = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg)
            robot_cfg = RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)
        ik_cfg = IKSolverCfg.create(
            robot_cfg, optimizer_configs=ik_optimizer_configs,
            scene_model=scene_model, self_collision_check=self_collision_check,
            device_cfg=device_cfg, num_seeds=num_ik_seeds or 32,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=use_cuda_graph, random_seed=random_seed,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            store_debug=store_debug, max_batch_size=max_batch_size,
            multi_env=multi_env, max_goalset=max_goalset,
        )
        traj_cfg = TrajOptSolverCfg.create(
            robot_cfg, optimizer_configs=trajopt_optimizer_configs,
            scene_model=scene_model, self_collision_check=self_collision_check,
            device_cfg=device_cfg, num_seeds=num_trajopt_seeds or 4,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=False, random_seed=random_seed,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=interpolation_buffer_size,
            max_batch_size=max_batch_size, multi_env=multi_env,
            max_goalset=max_goalset,
        )
        # The portable PRM is built lazily by MotionPlanner. Preserve the
        # upstream field while avoiding assumptions about task YAML internals.
        graph_cfg = None if graph_planner_config is None else graph_planner_config
        return MotionPlannerCfg(
            ik_cfg, traj_cfg, graph_cfg, scene_model, device_cfg
        )


__all__ = ["MotionPlannerCfg"]

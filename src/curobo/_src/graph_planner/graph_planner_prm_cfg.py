"""Portable PRM configuration with the pinned cuRoboV2 field layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


@dataclass
class PRMGraphPlannerCfg:
    max_nodes: int = 2048
    feasibility_buffer_size: int = 4096
    steer_buffer_size: int = 4096
    exploration_radius: float = 2.0
    new_nodes_per_iteration: int = 128
    max_path_finding_iterations: int = 20
    min_finetune_iterations: int = 0
    use_default_position_heuristic: bool = True
    cspace_similarity_threshold: float = 1e-4
    sample_rejection_ratio: int = 10
    neighbors_per_node: int = 12
    rollout_config: Any = None
    sampler_seed: int = 0
    sampler_buffer_size: int = 4096
    use_cuda_graph_for_rollout: bool = True
    connect_terminal_nodes_with_nearest: bool = True
    exploration_radius_growth_factor: float = 1.5
    neighbors_per_node_growth_factor: float = 1.5
    new_nodes_per_iteration_growth_factor: float = 1.5
    ellipsoid_projection_method: str = "householder"
    scene_collision_cfg: Optional[Any] = None
    device_cfg: DeviceCfg = DeviceCfg()
    graph_path_finder_seed: int = 42
    # Portable execution data; additive fields do not alter upstream constructor compatibility.
    robot_config: Optional[RobotCfg] = None
    action_lower_bounds: Optional[torch.Tensor] = None
    action_upper_bounds: Optional[torch.Tensor] = None
    check_feasibility_fn: Any = None
    connection_radius: Optional[float] = None
    edge_step: float = 0.05

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        use_cuda_graph_for_rollout: bool = True,
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        graph_path_finder_seed: int = 42,
    ) -> "PRMGraphPlannerCfg":
        del rollout, transition_model, scene_model, collision_cache, self_collision_check
        del transition_model_config_instance_type, cost_manager_config_instance_type
        robot_cfg = robot if isinstance(robot, RobotCfg) else RobotCfg.create(robot, device_cfg)
        parameters: dict[str, Any] = {}
        if isinstance(graph_planner_config, dict):
            parameters.update(graph_planner_config.get("graph_planner", graph_planner_config))
        metal = robot_cfg.kinematics
        joints = [
            joint for joint in metal.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        lower = device_cfg.to_device([joint.limits.lower for joint in joints])
        upper = device_cfg.to_device([joint.limits.upper for joint in joints])
        return PRMGraphPlannerCfg(
            **parameters,
            robot_config=robot_cfg,
            action_lower_bounds=lower,
            action_upper_bounds=upper,
            device_cfg=device_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph_for_rollout,
            graph_path_finder_seed=graph_path_finder_seed,
        )


__all__ = ["PRMGraphPlannerCfg"]

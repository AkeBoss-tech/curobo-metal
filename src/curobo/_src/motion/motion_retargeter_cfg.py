from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class MotionRetargeterCfg:
    robot: Union[str, Dict[str, Any]]
    tool_pose_criteria: Dict[str, ToolPoseCriteria]
    num_envs: int = 1
    use_mpc: bool = False
    self_collision_check: bool = True
    scene_model: Optional[Union[str, Dict[str, Any]]] = None
    optimization_dt: float = 0.05
    num_seeds_global: int = 64
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    load_collision_spheres: bool = True
    ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["ik/lbfgs_retarget_ik.yml"]
    )
    mpc_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["mpc/lbfgs_retarget_mpc.yml"]
    )
    num_seeds_local: int = 1
    num_control_points: Optional[int] = None
    steps_per_target: int = 8
    velocity_regularization_weight: Optional[float] = None
    acceleration_regularization_weight: Optional[float] = None
    collision_activation_distance: float = 0.01
    global_ik_num_iters: Optional[int] = None
    local_ik_num_iters: Optional[int] = None
    mpc_warm_start_num_iters: int = 100
    mpc_cold_start_num_iters: int = 300

    @property
    def tool_frames(self): return list(self.tool_pose_criteria)

    @staticmethod
    def create(
        robot, tool_pose_criteria, num_envs=1, use_mpc=False,
        self_collision_check=True, scene_model=None, optimization_dt=0.05,
        num_seeds_global=64, load_collision_spheres=True,
        ik_optimizer_configs=None, mpc_optimizer_configs=None,
        num_seeds_local=1, num_control_points=12, steps_per_target=4,
        position_tolerance=0.005, orientation_tolerance=0.05,
        device_cfg=DeviceCfg(), velocity_regularization_weight=None,
        acceleration_regularization_weight=None,
        collision_activation_distance=0.01, global_ik_num_iters=None,
        local_ik_num_iters=None, mpc_warm_start_num_iters=100,
        mpc_cold_start_num_iters=300,
    ):
        return MotionRetargeterCfg(
            robot, tool_pose_criteria, num_envs, use_mpc,
            self_collision_check, scene_model, optimization_dt,
            num_seeds_global, position_tolerance, orientation_tolerance,
            device_cfg, load_collision_spheres,
            ik_optimizer_configs or ["ik/lbfgs_retarget_ik.yml"],
            mpc_optimizer_configs or ["mpc/lbfgs_retarget_mpc.yml"],
            num_seeds_local, num_control_points, steps_per_target,
            velocity_regularization_weight, acceleration_regularization_weight,
            collision_activation_distance, global_ik_num_iters,
            local_ik_num_iters, mpc_warm_start_num_iters,
            mpc_cold_start_num_iters,
        )

    def __post_init__(self):
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not self.tool_pose_criteria:
            raise ValueError("tool_pose_criteria cannot be empty")


__all__ = ["MotionRetargeterCfg"]

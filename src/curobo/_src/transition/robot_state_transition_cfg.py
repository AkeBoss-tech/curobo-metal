"""Configuration for portable robot state transitions."""

from copy import deepcopy
from dataclasses import dataclass
import torch
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.state_filter import FilterCfg


@dataclass
class TimeTrajCfg:
    base_dt: float
    base_ratio: float
    max_dt: float

    def get_dt_array(self, num_points: int):
        base_count = int(self.base_ratio * num_points)
        output = [self.base_dt] * base_count
        output += torch.linspace(self.base_dt, self.max_dt, steps=num_points-base_count).tolist()
        return output[:num_points]

    def update_dt(self, all_dt=None, base_dt=None, max_dt=None, base_ratio=None):
        if all_dt is not None:
            self.base_dt = self.max_dt = all_dt
            return
        if base_dt is not None: self.base_dt = base_dt
        if max_dt is not None: self.max_dt = max_dt
        if base_ratio is not None: self.base_ratio = base_ratio


@dataclass
class RobotStateTransitionCfg:
    robot_config: RobotCfg
    dt_traj_params: TimeTrajCfg
    device_cfg: DeviceCfg
    vel_scale: float = 1.0
    state_estimation_variance: float = 0.0
    batch_size: int = 1
    horizon: int = 5
    n_knots: int = 0
    control_space: ControlSpace = ControlSpace.ACCELERATION
    state_filter_cfg: FilterCfg | None = None
    teleport_mode: bool = False
    return_full_act_buffer: bool = False
    state_finite_difference_mode: str = "BACKWARD"
    filter_robot_command: bool = False
    interpolation_steps: int = 1
    class_type: type | None = None

    @staticmethod
    def create(data_dict_in, robot_cfg, device_cfg=DeviceCfg()):
        data = deepcopy(data_dict_in)
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotCfg.create(robot_cfg, device_cfg)
        data["robot_config"] = robot_cfg
        data["dt_traj_params"] = TimeTrajCfg(**data["dt_traj_params"])
        if isinstance(data.get("control_space"), str):
            data["control_space"] = ControlSpace[data["control_space"]]
        filter_data = data.get("state_filter_cfg")
        if isinstance(filter_data, dict):
            data["state_filter_cfg"] = FilterCfg.create(
                filter_data["filter_coeff"], filter_data.get("enable", True),
                data["dt_traj_params"].base_dt, data["control_space"], device_cfg,
                data.get("teleport_mode", False),
            )
        return RobotStateTransitionCfg(**data, device_cfg=device_cfg)

    def __post_init__(self):
        if self.control_space in ControlSpace.bspline_types():
            if not 1 <= self.interpolation_steps <= 32:
                raise ValueError("interpolation_steps needs to be between 1 and 32")
            if self.n_knots <= 5:
                raise ValueError("n_knots needs to be greater than 5")
            if self.teleport_mode:
                raise ValueError("Teleport mode is not supported for bspline control spaces")
            self.horizon = ControlSpace.spline_total_interpolation_steps(
                self.control_space, self.n_knots, self.interpolation_steps
            )

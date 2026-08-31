"""Configuration for portable robot state transitions."""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Type, Union
import torch
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.state_filter import FilterCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class TimeTrajCfg:
    base_dt: float
    base_ratio: float
    max_dt: float

    def get_dt_array(self, num_points: int):
        """Return a deterministic, positive trajectory-time schedule.

        The V2 yaml surface specifies a leading constant section followed by
        a linear blend.  Treating a one-point or all-base schedule as a
        special case matters: ``torch.linspace(..., steps=0)`` is invalid and
        used to make otherwise valid horizon-one rollouts fail before any
        transition arithmetic ran.
        """
        if not isinstance(num_points, int) or num_points < 0:
            raise ValueError("num_points must be a non-negative integer")
        if num_points == 0:
            return []
        base_count = min(num_points, int(self.base_ratio * num_points))
        output = [self.base_dt] * base_count
        blend_count = num_points - base_count
        if blend_count:
            output.extend(torch.linspace(self.base_dt, self.max_dt, steps=blend_count).tolist())
        return output

    def update_dt(
        self,
        all_dt: float = None,
        base_dt: float = None,
        max_dt: float = None,
        base_ratio: float = None,
    ):
        if all_dt is not None:
            self.base_dt = self.max_dt = all_dt
            self.__post_init__()
            return
        if base_dt is not None: self.base_dt = base_dt
        if max_dt is not None: self.max_dt = max_dt
        if base_ratio is not None: self.base_ratio = base_ratio
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in ("base_dt", "max_dt"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite, positive number")
            setattr(self, name, float(value))
        if (
            not isinstance(self.base_ratio, (int, float))
            or not math.isfinite(self.base_ratio)
            or not 0.0 <= self.base_ratio <= 1.0
        ):
            raise ValueError("base_ratio must be a finite value in [0, 1]")
        self.base_ratio = float(self.base_ratio)


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
    class_type: Optional[Type[Any]] = RobotStateTransition

    @staticmethod
    def create(
        data_dict_in, robot_cfg: Union[Dict, RobotCfg], device_cfg=DeviceCfg()
    ):
        if not isinstance(data_dict_in, Mapping):
            raise TypeError("transition configuration must be a mapping")
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        data = deepcopy(data_dict_in)
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotCfg.create(robot_cfg, device_cfg)
        data["robot_config"] = robot_cfg
        time_cfg = data.get("dt_traj_params")
        data["dt_traj_params"] = (
            time_cfg if isinstance(time_cfg, TimeTrajCfg) else TimeTrajCfg(**time_cfg)
        )
        control = data.get("control_space", ControlSpace.ACCELERATION)
        if isinstance(control, str):
            try:
                control = ControlSpace[control.upper()]
            except KeyError as error:
                raise ValueError(f"unknown control_space {control!r}") from error
        data["control_space"] = control
        filter_data = data.get("state_filter_cfg")
        if isinstance(filter_data, dict):
            data["state_filter_cfg"] = FilterCfg.create(
                filter_data["filter_coeff"], filter_data.get("enable", True),
                data["dt_traj_params"].base_dt, data["control_space"], device_cfg,
                data.get("teleport_mode", False),
            )
        elif filter_data is not None and not isinstance(filter_data, FilterCfg):
            raise TypeError("state_filter_cfg must be a FilterCfg or mapping")
        return RobotStateTransitionCfg(**data, device_cfg=device_cfg)

    def __post_init__(self):
        if not isinstance(self.robot_config, RobotCfg):
            raise TypeError("robot_config must be a RobotCfg")
        if not isinstance(self.dt_traj_params, TimeTrajCfg):
            raise TypeError("dt_traj_params must be a TimeTrajCfg")
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(self.control_space, ControlSpace):
            raise TypeError("control_space must be a ControlSpace")
        for name in ("batch_size", "horizon", "n_knots", "interpolation_steps"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.n_knots < 0:
            raise ValueError("n_knots must be non-negative")
        if not isinstance(self.vel_scale, (int, float)) or self.vel_scale <= 0:
            raise ValueError("vel_scale must be positive")
        if not isinstance(self.state_estimation_variance, (int, float)) or self.state_estimation_variance < 0:
            raise ValueError("state_estimation_variance must be non-negative")
        if self.state_filter_cfg is not None and not isinstance(self.state_filter_cfg, FilterCfg):
            raise TypeError("state_filter_cfg must be a FilterCfg")
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

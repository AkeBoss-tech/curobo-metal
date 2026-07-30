"""Trajectory interpolation helpers compatible with cuRoboV2."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class TrajInterpolationType(Enum):
    LINEAR = "linear"
    CUBIC = "cubic"
    QUARTIC = "quartic"
    QUINTIC = "quintic"
    LINEAR_CUDA = "linear_cuda"
    BSPLINE_KNOTS_CUDA = "bspline_knots_cuda"


def calculate_dt_no_clamp(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    epsilon: float = 1e-5,
):
    """Return the minimum physical timestep satisfying derivative limits."""
    values = (
        (vel.abs() / max_vel.clamp_min(epsilon)).amax(dim=-1),
        torch.sqrt((acc.abs() / max_acc.clamp_min(epsilon)).amax(dim=-1)),
        torch.pow((jerk.abs() / max_jerk.clamp_min(epsilon)).amax(dim=-1), 1.0 / 3.0),
    )
    return torch.stack(values).amax(dim=0).clamp_min(epsilon)


def calculate_traj_steps(
    opt_dt: torch.Tensor,
    interpolation_dt: torch.Tensor,
    horizon: int,
    nearest_int: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = torch.as_tensor(opt_dt)
    target = torch.as_tensor(interpolation_dt, device=dt.device, dtype=dt.dtype)
    count = (dt * (horizon - 1) / target).round() if nearest_int else torch.ceil(
        dt * (horizon - 1) / target
    )
    count = count.to(torch.int64) + 1
    return count, count.max()


def _linear_state(raw: JointState, steps: torch.Tensor, out: JointState) -> JointState:
    batch = raw.position.unsqueeze(0) if raw.position.ndim == 2 else raw.position
    for index in range(batch.shape[0]):
        count = int(steps[index].item())
        coordinate = torch.linspace(
            0, batch.shape[1] - 1, count, device=batch.device, dtype=batch.dtype
        )
        low = coordinate.floor().to(torch.int64).clamp_max(batch.shape[1] - 2)
        fraction = (coordinate - low).unsqueeze(-1)
        values = batch[index, low] * (1 - fraction) + batch[index, low + 1] * fraction
        out.position[index, :count] = values
        out.position[index, count:] = values[-1]
    return out.finite_difference(
        torch.as_tensor(1.0, device=out.position.device, dtype=out.position.dtype)
    )


def get_batch_interpolated_trajectory(
    raw_traj: JointState,
    interpolation_dt: torch.Tensor,
    kind: TrajInterpolationType = TrajInterpolationType.LINEAR_CUDA,
    out_traj_state: Optional[JointState] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
    current_state: Optional[JointState] = None,
    goal_state: Optional[JointState] = None,
    start_idx: Optional[torch.Tensor] = None,
    goal_idx: Optional[torch.Tensor] = None,
    use_implicit_goal_state: Optional[torch.Tensor] = None,
):
    del current_state, goal_state, start_idx, goal_idx, use_implicit_goal_state
    if kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
        raise NotImplementedError(
            "BSPLINE_KNOTS_CUDA requires the pinned CUDA spline kernel; use LINEAR_CUDA "
            "or a portable interpolation type"
        )
    raw = raw_traj.unsqueeze(0) if raw_traj.position.ndim == 2 else raw_traj
    raw_dt = raw.dt
    if raw_dt is None:
        raw_dt = torch.ones(raw.position.shape[0], device=raw.position.device, dtype=raw.position.dtype)
    steps, maximum = calculate_traj_steps(raw_dt, interpolation_dt, raw.position.shape[1])
    size = (raw.position.shape[0], int(maximum.item()), raw.position.shape[-1])
    if out_traj_state is None or out_traj_state.position.shape[1] < size[1]:
        out_traj_state = JointState.zeros(size, device_cfg, joint_names=raw.joint_names)
    return _linear_state(raw, steps, out_traj_state), steps


def get_cpu_linear_interpolation(
    raw_traj,
    traj_steps,
    out_traj_state,
    kind: TrajInterpolationType,
    interpolation_dt=None,
):
    del kind, interpolation_dt
    return _linear_state(raw_traj, traj_steps, out_traj_state)


def linear_smooth(
    x: np.array,
    y=None,
    n=10,
    kind=TrajInterpolationType.CUBIC,
    last_step=None,
    opt_dt=None,
    interpolation_dt=None,
):
    del opt_dt, interpolation_dt
    values = np.asarray(x)
    count = n if last_step is None else last_step
    source = np.arange(values.shape[0], dtype=np.float64) if y is None else np.asarray(y)
    target = np.linspace(source[0], source[-1], count)
    if kind == TrajInterpolationType.QUARTIC:
        raise NotImplementedError("cuRoboV2 QUARTIC interpolation is not implemented upstream")
    # Linear is deterministic and dependency-free; higher-order labels retain the same endpoints.
    return np.interp(target, source, values)


def get_interpolated_trajectory(
    trajectory: List[torch.Tensor],
    out_traj_state: JointState,
    des_horizon: Optional[int] = None,
    interpolation_dt: float = 0.02,
    kind=TrajInterpolationType.CUBIC,
    device_cfg: DeviceCfg = DeviceCfg(),
    max_joint_velocity: Optional[torch.Tensor] = None,
) -> JointState:
    del interpolation_dt, device_cfg, max_joint_velocity
    raw = JointState.from_position(torch.stack(trajectory), out_traj_state.joint_names)
    steps = torch.tensor(
        [des_horizon or out_traj_state.position.shape[-2]], device=raw.position.device
    )
    return get_cpu_linear_interpolation(raw.unsqueeze(0), steps, out_traj_state, kind)


__all__ = [
    "TrajInterpolationType", "calculate_dt_no_clamp", "calculate_traj_steps",
    "get_batch_interpolated_trajectory", "get_cpu_linear_interpolation",
    "get_interpolated_trajectory", "linear_smooth",
]

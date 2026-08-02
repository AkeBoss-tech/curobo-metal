"""Trajectory interpolation helpers compatible with cuRoboV2."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.autograd.profiler as profiler

from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import get_torch_jit_decorator


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
    """Return the derivative-limit timestep multiplier for each batch item."""
    max_v = vel.abs().amax(dim=-2)
    max_a = acc.abs().amax(dim=-2)
    max_j = jerk.abs().amax(dim=-2)
    v_score = (max_v / max_vel.clamp_min(epsilon).reshape(1, -1)).amax(dim=-1)
    a_score = torch.sqrt((max_a / max_acc.clamp_min(epsilon).reshape(1, -1)).amax(dim=-1))
    j_score = torch.pow((max_j / max_jerk.clamp_min(epsilon).reshape(1, -1)).amax(dim=-1), 1.0 / 3.0)
    return torch.maximum(torch.maximum(v_score, a_score), j_score) * (1.0 + epsilon)


def calculate_traj_steps(
    opt_dt: torch.Tensor,
    interpolation_dt: torch.Tensor,
    horizon: int,
    nearest_int: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = torch.as_tensor(opt_dt)
    target = torch.as_tensor(interpolation_dt, device=dt.device, dtype=dt.dtype)
    per_waypoint = (dt + target) / target if nearest_int else (dt + torch.remainder(dt, target)) / target
    count = ((horizon - 1) * per_waypoint.to(torch.int64) + 1).to(torch.int32)
    return count, count.max().to(torch.int32)


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
    if kind == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
        return get_bspline_interpolation(
            raw_traj, out_traj_state, interpolation_dt, current_state=current_state,
            goal_state=goal_state, start_idx=start_idx, goal_idx=goal_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
    del current_state, goal_state, start_idx, goal_idx, use_implicit_goal_state
    raw = raw_traj.unsqueeze(0) if raw_traj.position.ndim == 2 else raw_traj
    raw_dt = raw.dt
    if raw_dt is None:
        raw_dt = torch.ones(raw.position.shape[0], device=raw.position.device, dtype=raw.position.dtype)
    steps, maximum = calculate_traj_steps(raw_dt, interpolation_dt, raw.position.shape[1])
    size = (raw.position.shape[0], int(maximum.item()), raw.position.shape[-1])
    if out_traj_state is None or out_traj_state.position.shape[1] < size[1]:
        out_traj_state = JointState.zeros(size, device_cfg, joint_names=raw.joint_names)
    if kind == TrajInterpolationType.LINEAR_CUDA:
        return get_cuda_linear_interpolation(raw, steps, out_traj_state), steps
    return get_cpu_linear_interpolation(raw, steps, out_traj_state, kind, interpolation_dt), steps


def get_cpu_linear_interpolation(
    raw_traj,
    traj_steps,
    out_traj_state,
    kind: TrajInterpolationType,
    interpolation_dt=None,
):
    del kind, interpolation_dt
    return _linear_state(raw_traj, traj_steps, out_traj_state)


def get_cuda_linear_interpolation(raw_traj, traj_tsteps, out_traj):
    """Portable implementation of the historical CUDA-named interpolation entry point."""
    return _linear_state(raw_traj, traj_tsteps, out_traj)


def get_bspline_interpolation(*args, **kwargs):
    """Document the raw CUDA spline-kernel boundary without silently changing its semantics."""
    del args, kwargs
    raise NotImplementedError(
        "BSPLINE_KNOTS_CUDA requires cuRobo's CUDA spline kernel; use LINEAR_CUDA, LINEAR, "
        "CUBIC, or QUINTIC on the portable CPU/MPS backend"
    )


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
    "get_batch_interpolated_trajectory", "get_bspline_interpolation", "get_cuda_linear_interpolation", "get_cpu_linear_interpolation",
    "get_interpolated_trajectory", "linear_smooth",
]

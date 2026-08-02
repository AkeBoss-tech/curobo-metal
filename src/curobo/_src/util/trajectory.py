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
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    dt = torch.as_tensor(opt_dt)
    target = torch.as_tensor(interpolation_dt, device=dt.device, dtype=dt.dtype)
    if target.numel() != 1 or bool((dt <= 0).any().item()) or bool((target <= 0).any().item()):
        raise ValueError("opt_dt and interpolation_dt must be positive")
    per_waypoint = (
        (dt + target) / target
        if nearest_int
        else (dt + torch.remainder(dt, target)) / target
    )
    count = ((horizon - 1) * per_waypoint.to(torch.int64) + 1).to(torch.int32)
    return count, count.max().to(torch.int32)


def _interpolate_values(values: torch.Tensor, count: int, kind: TrajInterpolationType) -> torch.Tensor:
    """Retiming kernel shared by CPU/MPS paths, preserving endpoint values.

    CUBIC uses a Catmull--Rom Hermite stencil and QUINTIC uses a smoothstep
    blend.  Both are composed Torch operations, so their output remains on the
    caller's device and differentiable with respect to waypoint positions.
    """
    horizon = values.shape[-2]
    coordinate = torch.linspace(0, horizon - 1, count, device=values.device, dtype=values.dtype)
    low = coordinate.floor().to(torch.int64).clamp_max(horizon - 2)
    fraction = (coordinate - low).unsqueeze(-1)
    p1, p2 = values[low], values[low + 1]
    if kind in (TrajInterpolationType.LINEAR, TrajInterpolationType.LINEAR_CUDA):
        return p1 * (1 - fraction) + p2 * fraction
    if kind == TrajInterpolationType.CUBIC:
        p0 = values[(low - 1).clamp_min(0)]
        p3 = values[(low + 2).clamp_max(horizon - 1)]
        u, u2, u3 = fraction, fraction.square(), fraction.pow(3)
        return 0.5 * (
            2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
        )
    if kind == TrajInterpolationType.QUINTIC:
        smooth = fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction))
        return p1 * (1 - smooth) + p2 * smooth
    if kind == TrajInterpolationType.QUARTIC:
        raise NotImplementedError("cuRoboV2 QUARTIC interpolation is not implemented upstream")
    raise ValueError(f"Unsupported interpolation type: {kind}")


def _interpolate_state(
    raw: JointState,
    steps: torch.Tensor,
    out: JointState,
    kind: TrajInterpolationType,
    interpolation_dt: torch.Tensor,
) -> JointState:
    batch = raw.position.unsqueeze(0) if raw.position.ndim == 2 else raw.position
    if out.position.ndim == 2:
        out = out.unsqueeze(0)
    # ``calculate_traj_steps`` returns a scalar for a shared scalar dt and a
    # per-batch vector otherwise.  Normalize both contracts here so a batched
    # trajectory with one common dt does not try to index a 0-D tensor.
    step_values = steps.reshape(-1)
    if step_values.numel() not in (1, batch.shape[0]):
        raise ValueError("trajectory step counts must be scalar or per-batch")
    for index in range(batch.shape[0]):
        count = int(step_values[0 if step_values.numel() == 1 else index].item())
        values = _interpolate_values(batch[index], count, kind)
        out.position[index, :count] = values
        out.position[index, count:] = values[-1]
    output_dt = torch.as_tensor(interpolation_dt, device=out.position.device, dtype=out.position.dtype)
    return out.finite_difference(output_dt)


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
    if int(maximum.item()) > 10000:
        raise ValueError("interpolated trajectory exceeds the portable 10000-step limit")
    size = (raw.position.shape[0], int(maximum.item()), raw.position.shape[-1])
    if out_traj_state is None or out_traj_state.position.shape[1] < size[1]:
        output_cfg = DeviceCfg(raw.position.device, raw.position.dtype)
        out_traj_state = JointState.zeros(size, output_cfg, joint_names=raw.joint_names)
    if kind == TrajInterpolationType.LINEAR_CUDA:
        return get_cuda_linear_interpolation(raw, steps, out_traj_state, interpolation_dt), steps
    return get_cpu_linear_interpolation(raw, steps, out_traj_state, kind, interpolation_dt), steps


def get_cpu_linear_interpolation(
    raw_traj,
    traj_steps,
    out_traj_state,
    kind: TrajInterpolationType,
    interpolation_dt=None,
):
    if interpolation_dt is None:
        interpolation_dt = torch.ones((), device=raw_traj.position.device, dtype=raw_traj.position.dtype)
    return _interpolate_state(raw_traj, traj_steps, out_traj_state, kind, interpolation_dt)


def get_cuda_linear_interpolation(raw_traj, traj_tsteps, out_traj, interpolation_dt=None):
    """Portable implementation of the historical CUDA-named interpolation entry point."""
    if interpolation_dt is None:
        interpolation_dt = torch.ones((), device=raw_traj.position.device, dtype=raw_traj.position.dtype)
    return _interpolate_state(
        raw_traj, traj_tsteps, out_traj, TrajInterpolationType.LINEAR_CUDA, interpolation_dt
    )


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
    values = np.asarray(x)
    count = n if last_step is None else last_step
    source = np.arange(values.shape[0], dtype=np.float64) if y is None else np.asarray(y)
    if values.ndim != 1 or source.ndim != 1 or values.shape != source.shape:
        raise ValueError("x and y must be matching one-dimensional arrays")
    if count < 1:
        raise ValueError("n must be positive")
    if opt_dt is not None and interpolation_dt is not None:
        target = np.arange(count, dtype=np.float64) * float(interpolation_dt)
    else:
        target = np.linspace(source[0], source[-1], count)
    if kind == TrajInterpolationType.QUARTIC:
        raise NotImplementedError("cuRoboV2 QUARTIC interpolation is not implemented upstream")
    # This NumPy-facing helper has no differentiable input.  Use the exact same
    # portable Torch interpolation stencil as the batched CPU/MPS routine.
    tensor = torch.as_tensor(values, dtype=torch.float64).unsqueeze(-1)
    # Nonuniform source coordinates are represented by first mapping target to
    # the segment coordinate; trajectory execution normally uses uniform time.
    coordinate = np.interp(target, source, np.arange(source.size, dtype=np.float64))
    coordinate_t = torch.as_tensor(coordinate, dtype=tensor.dtype)
    low = coordinate_t.floor().to(torch.int64).clamp(0, tensor.shape[0] - 2)
    fraction = (coordinate_t - low).unsqueeze(-1)
    p1, p2 = tensor[low], tensor[low + 1]
    if kind in (TrajInterpolationType.LINEAR, TrajInterpolationType.LINEAR_CUDA):
        result = p1 * (1 - fraction) + p2 * fraction
    elif kind == TrajInterpolationType.CUBIC:
        p0 = tensor[(low - 1).clamp_min(0)]
        p3 = tensor[(low + 2).clamp_max(tensor.shape[0] - 1)]
        u, u2, u3 = fraction, fraction.square(), fraction.pow(3)
        result = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
    else:
        smooth = fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction))
        result = p1 * (1 - smooth) + p2 * smooth
    return result.squeeze(-1)


def get_interpolated_trajectory(
    trajectory: List[torch.Tensor],
    out_traj_state: JointState,
    des_horizon: Optional[int] = None,
    interpolation_dt: float = 0.02,
    kind=TrajInterpolationType.CUBIC,
    device_cfg: DeviceCfg = DeviceCfg(),
    max_joint_velocity: Optional[torch.Tensor] = None,
) -> JointState:
    del max_joint_velocity
    if not trajectory:
        raise ValueError("trajectory must contain at least one batch item")
    horizon = des_horizon or out_traj_state.position.shape[-2]
    if horizon < 2:
        raise ValueError("des_horizon must be at least two")
    if kind not in (
        TrajInterpolationType.LINEAR, TrajInterpolationType.CUBIC,
        TrajInterpolationType.QUARTIC, TrajInterpolationType.QUINTIC,
    ):
        raise ValueError(f"Unsupported interpolation type: {kind}")
    if kind == TrajInterpolationType.QUARTIC:
        raise NotImplementedError("cuRoboV2 QUARTIC interpolation is not implemented upstream")
    if len(trajectory) != out_traj_state.position.shape[0]:
        raise ValueError("trajectory batch must match out_traj_state")
    for batch_index, item in enumerate(trajectory):
        values = item.reshape(-1, item.shape[-1]).to(
            device=out_traj_state.position.device, dtype=out_traj_state.position.dtype
        )
        actual_kind = TrajInterpolationType.LINEAR if values.shape[0] < 4 and kind == TrajInterpolationType.CUBIC else kind
        retimed = _interpolate_values(values, horizon, actual_kind)
        out_traj_state.position[batch_index, :horizon] = retimed
        out_traj_state.position[batch_index, horizon:] = retimed[-1]
    out_traj_state = out_traj_state.finite_difference(interpolation_dt)
    last_steps = [horizon] * len(trajectory)
    dt = device_cfg.to_device([interpolation_dt] * len(trajectory))
    return out_traj_state, last_steps, dt


__all__ = [
    "TrajInterpolationType", "calculate_dt_no_clamp", "calculate_traj_steps",
    "get_batch_interpolated_trajectory", "get_bspline_interpolation", "get_cuda_linear_interpolation", "get_cpu_linear_interpolation",
    "get_interpolated_trajectory", "linear_smooth",
]

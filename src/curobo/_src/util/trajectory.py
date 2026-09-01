"""Trajectory interpolation helpers compatible with cuRoboV2."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.autograd.profiler as profiler
from scipy import interpolate

from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.util._trajectory_aliases import (
    get_bspline_interpolation,
    get_cuda_linear_interpolation,
)


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
        raw = raw_traj.unsqueeze(0) if raw_traj.position.ndim == 2 else raw_traj
        if raw.knot is None or raw.knot_dt is None:
            raise ValueError(
                "BSPLINE_KNOTS_CUDA requires JointState.knot [batch, knot, dof] and knot_dt"
            )
        if raw.control_space not in ControlSpace.bspline_types():
            raise ValueError("BSPLINE_KNOTS_CUDA requires a BSPLINE_3, BSPLINE_4, or BSPLINE_5 control_space")
        knot = raw.knot.unsqueeze(0) if raw.knot.ndim == 2 else raw.knot
        if knot.ndim != 3 or knot.shape[0] != raw.position.shape[0] or knot.shape[-1] != raw.position.shape[-1]:
            raise ValueError("JointState.knot must have shape [batch, knot, dof] matching position")
        steps, maximum = calculate_traj_steps(
            raw.knot_dt, interpolation_dt,
            ControlSpace.spline_total_knots(raw.control_space, knot.shape[1]) + 1,
            nearest_int=True,
        )
        if int(maximum.item()) > 10000:
            raise ValueError("interpolated trajectory exceeds the portable 10000-step limit")
        size = (knot.shape[0], int(maximum.item()), knot.shape[-1])
        if out_traj_state is None or out_traj_state.position.ndim != 3 or out_traj_state.position.shape[0] != size[0] or out_traj_state.position.shape[1] < size[1] or out_traj_state.position.shape[-1] != size[-1]:
            out_traj_state = JointState.zeros(size, DeviceCfg(knot.device, knot.dtype), joint_names=raw.joint_names)
        return get_bspline_interpolation(
            raw, out_traj_state, interpolation_dt, current_state=current_state,
            goal_state=goal_state, start_idx=start_idx, goal_idx=goal_idx,
            use_implicit_goal_state=use_implicit_goal_state, interpolated_horizon=steps,
            bspline_degree=ControlSpace.spline_degree(raw.control_space),
        ), steps
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


def _get_cuda_linear_interpolation_portable(
    raw_traj, traj_tsteps, out_traj, interpolation_dt=None
):
    """Portable implementation of the historical CUDA-named interpolation entry point."""
    if interpolation_dt is None:
        interpolation_dt = torch.ones((), device=raw_traj.position.device, dtype=raw_traj.position.dtype)
    return _interpolate_state(
        raw_traj, traj_tsteps, out_traj, TrajInterpolationType.LINEAR_CUDA, interpolation_dt
    )


def _select_bspline_boundary(
    state: Optional[JointState], indices: Optional[torch.Tensor], batch: int,
    reference: torch.Tensor, name: str,
) -> Optional[torch.Tensor]:
    """Return one position row per spline batch without CPU staging.

    The CUDA entry point accepts a table of current/goal states and integer
    lookup buffers.  This portable variant retains the useful table semantics
    using regular Torch indexing.  It purposely has no packed-buffer or CUDA
    graph dependency.
    """
    if state is None:
        return None
    value = state.position
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != reference.shape[-1]:
        raise ValueError(f"{name}_state.position must be [table, dof]")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name}_state must share spline knot device and dtype")
    if indices is None:
        if value.shape[0] == 1:
            return value.expand(batch, -1)
        if value.shape[0] == batch:
            return value
        raise ValueError(f"{name}_idx is required when {name}_state has a table size other than one or batch")
    if value.shape[0] == batch:
        # The pinned call surface also passes dense-horizon indices alongside
        # an already batch-aligned boundary state. In that form there is no
        # state table to gather; each row is already the selected boundary.
        return value
    indices = torch.as_tensor(indices, device=reference.device, dtype=torch.long).reshape(-1)
    if indices.numel() == 1:
        indices = indices.expand(batch)
    if indices.numel() != batch:
        raise ValueError(f"{name}_idx must be scalar or [batch]")
    if bool(((indices < 0) | (indices >= value.shape[0])).any().item()):
        raise IndexError(f"{name}_idx contains an out-of-range state index")
    return value.index_select(0, indices)


def _clamped_uniform_bspline_basis(
    count: int, controls: int, degree: int, reference: torch.Tensor
) -> torch.Tensor:
    """Evaluate a clamped uniform B-spline basis with ordinary Torch ops.

    This is a portable mathematical implementation, not a substitute claim
    for cuRobo's CUDA launch layout.  It is deterministic, autograd-safe, and
    uses the conventional endpoint interpolation of clamped B-splines.
    """
    if count < 2:
        raise ValueError("B-spline output requires at least two samples")
    if degree < 1 or controls <= degree:
        raise ValueError("B-spline needs at least degree + 1 control knots")
    # p+1 repeated endpoint knots and uniformly spaced internal knots.
    interior_count = controls - degree - 1
    if interior_count > 0:
        interior = torch.arange(1, interior_count + 1, device=reference.device, dtype=reference.dtype)
        interior = interior / (interior_count + 1)
        knots = torch.cat((torch.zeros(degree + 1, device=reference.device, dtype=reference.dtype), interior, torch.ones(degree + 1, device=reference.device, dtype=reference.dtype)))
    else:
        knots = torch.cat((torch.zeros(degree + 1, device=reference.device, dtype=reference.dtype), torch.ones(degree + 1, device=reference.device, dtype=reference.dtype)))
    parameter = torch.linspace(0, 1, count, device=reference.device, dtype=reference.dtype)
    # Degree-zero basis first.  The right endpoint belongs to the final basis
    # function by convention, preserving exact endpoint values.
    basis = ((parameter[:, None] >= knots[:-1]) & (parameter[:, None] < knots[1:])).to(reference.dtype)
    basis[-1].zero_()
    basis[-1, -1] = 1
    for order in range(1, degree + 1):
        width = knots.numel() - order - 1
        left_denominator = knots[order:order + width] - knots[:width]
        right_denominator = knots[order + 1:order + 1 + width] - knots[1:width + 1]
        left = torch.where(
            left_denominator.abs() > 0,
            (parameter[:, None] - knots[:width]) / left_denominator,
            torch.zeros((count, width), device=reference.device, dtype=reference.dtype),
        ) * basis[:, :width]
        right = torch.where(
            right_denominator.abs() > 0,
            (knots[order + 1:order + 1 + width] - parameter[:, None]) / right_denominator,
            torch.zeros((count, width), device=reference.device, dtype=reference.dtype),
        ) * basis[:, 1:width + 1]
        basis = left + right
    # Recursive half-open intervals exclude t=1.  Restore the conventional
    # clamped endpoint after the recurrence so the final control point is
    # represented exactly rather than only in the degree-zero stencil.
    basis[-1].zero_()
    basis[-1, -1] = 1
    return basis[:, :controls]


def _cubic_boundary_spline(
    action: torch.Tensor,
    start: JointState,
    goal: JointState,
    interpolation_dt: torch.Tensor,
    horizon: int,
    start_idx: Optional[torch.Tensor] = None,
    goal_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the pinned CUDA cubic B-spline boundary kernel.

    cuRobo's action knots are surrounded by four fixed knots reconstructed
    from endpoint position, velocity, and acceleration.  The implicit-goal
    kernel omits the final action knot and appends four goal knots, yielding
    ``action_knots + 4`` cardinal spline segments.
    """
    if action.ndim != 3 or action.shape[1] < 1:
        raise ValueError("cubic spline actions must be [batch, knot, dof]")
    segments = action.shape[1] + 4
    if (horizon - 1) % segments != 0:
        raise ValueError("cubic spline horizon minus one must be divisible by action knots plus four")
    interpolation_steps = (horizon - 1) // segments
    if interpolation_steps < 1:
        raise ValueError("cubic spline horizon is too short for the action knot count")
    step_dt = torch.as_tensor(interpolation_dt, device=action.device, dtype=action.dtype)
    if step_dt.numel() != 1 or bool((step_dt <= 0).any().item()):
        raise ValueError("interpolation_dt must be one positive scalar")
    knot_dt = step_dt.reshape(()) * interpolation_steps

    def fixed(state: JointState, indices: Optional[torch.Tensor]) -> torch.Tensor:
        position = state.position.reshape(-1, action.shape[-1])
        velocity = state.velocity.reshape_as(position)
        acceleration = state.acceleration.reshape_as(position)
        if indices is not None:
            lookup = torch.as_tensor(indices, device=action.device, dtype=torch.int64).reshape(-1)
            if lookup.numel() == 1:
                lookup = lookup.expand(action.shape[0])
            if lookup.numel() != action.shape[0] or bool((lookup < 0).any().item()) or bool((lookup >= position.shape[0]).any().item()):
                raise ValueError("boundary indices must select one state per spline batch")
            position, velocity, acceleration = position[lookup], velocity[lookup], acceleration[lookup]
        elif position.shape[0] == 1:
            position = position.expand(action.shape[0], -1)
            velocity = velocity.expand(action.shape[0], -1)
            acceleration = acceleration.expand(action.shape[0], -1)
        elif position.shape[0] != action.shape[0]:
            raise ValueError("boundary state batch must be one or match spline actions")
        vel_coeff = action.new_tensor([-1.0, 0.0, 1.0, 2.0])
        acc_coeff = action.new_tensor([1.0 / 3.0, -1.0 / 6.0, 1.0 / 3.0, 11.0 / 6.0])
        return (
            position[:, None, :]
            + velocity[:, None, :] * knot_dt * vel_coeff[None, :, None]
            + acceleration[:, None, :] * knot_dt.square() * acc_coeff[None, :, None]
        )

    controls = torch.cat((fixed(start, start_idx), action[:, :-1, :], fixed(goal, goal_idx)), dim=1)
    sample = torch.arange(horizon, device=action.device)
    segment = torch.div(sample, interpolation_steps, rounding_mode="floor").clamp_max(segments - 1)
    local = (sample.to(action.dtype) / interpolation_steps) - torch.div(
        sample, interpolation_steps, rounding_mode="floor"
    ).to(action.dtype)
    local[-1] = 1.0
    window = torch.stack([controls[:, segment + offset, :] for offset in range(4)], dim=-2)
    one_minus = 1.0 - local
    position_basis = torch.stack((
        one_minus.pow(3) / 6.0,
        (4.0 - 6.0 * local.square() + 3.0 * local.pow(3)) / 6.0,
        (1.0 + 3.0 * local + 3.0 * local.square() - 3.0 * local.pow(3)) / 6.0,
        local.pow(3) / 6.0,
    ), dim=-1)
    velocity_basis = torch.stack((
        -0.5 * one_minus.square(),
        0.5 * (3.0 * local.square() - 4.0 * local),
        0.5 * (1.0 + 2.0 * local - 3.0 * local.square()),
        0.5 * local.square(),
    ), dim=-1) / knot_dt
    acceleration_basis = torch.stack((
        one_minus, 3.0 * local - 2.0, 1.0 - 3.0 * local, local,
    ), dim=-1) / knot_dt.square()
    jerk_basis = action.new_tensor([-1.0, 3.0, -3.0, 1.0]).expand(horizon, -1) / knot_dt.pow(3)

    def evaluate(basis: torch.Tensor) -> torch.Tensor:
        return (window * basis[None, :, :, None]).sum(dim=-2)

    return evaluate(position_basis), evaluate(velocity_basis), evaluate(acceleration_basis), evaluate(jerk_basis)


def _get_bspline_interpolation_portable(
    input_trajectory: Optional[JointState] = None,
    output_trajectory: Optional[JointState] = None,
    interpolation_dt: Optional[torch.Tensor] = None,
    current_state: Optional[JointState] = None,
    goal_state: Optional[JointState] = None,
    start_idx: Optional[torch.Tensor] = None,
    goal_idx: Optional[torch.Tensor] = None,
    use_implicit_goal_state: Optional[torch.Tensor] = None,
    interpolated_horizon: Optional[torch.Tensor] = None,
    bspline_degree: int = 4,
) -> JointState:
    """Materialise a clamped-uniform B-spline trajectory on CPU or MPS.

    The function keeps the pinned CUDA-named signature so normal cuRobo
    callers can exercise B-spline TrajOpt results on Metal. Cubic implicit-goal
    trajectories with a uniform full horizon reproduce the pinned CUDA fixed-
    knot boundary contract. Other degree, mixed-goal, and variable-horizon
    combinations retain the portable clamped-uniform implementation.
    """
    # Retain the old zero-argument CUDA-kernel boundary used by callers that
    # probe for the raw packed ABI.  Real portable callers provide the typed
    # state/output/dt contract below and use composed PyTorch interpolation.
    if input_trajectory is None or output_trajectory is None or interpolation_dt is None:
        raise NotImplementedError("CUDA spline kernel requires explicit packed buffers")
    if input_trajectory.knot is None:
        raise ValueError("input_trajectory.knot is required for B-spline interpolation")
    knots = input_trajectory.knot
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    if knots.ndim != 3:
        raise ValueError("input_trajectory.knot must be [batch, knot, dof]")
    output = output_trajectory
    if output.position.ndim == 2:
        output = output.unsqueeze(0)
    if output.position.ndim != 3 or output.position.shape[0] != knots.shape[0] or output.position.shape[-1] != knots.shape[-1]:
        raise ValueError("output_trajectory.position must be [batch, horizon, dof] matching knots")
    if knots.device != output.position.device or knots.dtype != output.position.dtype:
        raise ValueError("input spline knots and output trajectory must share device and dtype")
    batch, _, dof = knots.shape
    if interpolated_horizon is None:
        steps = torch.full((batch,), output.position.shape[1], device=knots.device, dtype=torch.int32)
    else:
        steps = torch.as_tensor(interpolated_horizon, device=knots.device, dtype=torch.int32).reshape(-1)
        if steps.numel() == 1:
            steps = steps.expand(batch)
        if steps.numel() != batch:
            raise ValueError("interpolated_horizon must be scalar or [batch]")
    if bool((steps < 2).any().item()) or bool((steps > output.position.shape[1]).any().item()):
        raise ValueError("interpolated_horizon must be between two and output horizon")
    controls = knots
    start = _select_bspline_boundary(current_state, start_idx, batch, knots, "start")
    goal = _select_bspline_boundary(goal_state, goal_idx, batch, knots, "goal")
    if start is not None:
        controls = torch.cat((start[:, None, :], controls[:, 1:, :]), dim=1)
    if goal is not None:
        if use_implicit_goal_state is None:
            implicit_goal = torch.ones(batch, device=knots.device, dtype=torch.bool)
        else:
            implicit_goal = torch.as_tensor(use_implicit_goal_state, device=knots.device, dtype=torch.bool).reshape(-1)
            if implicit_goal.numel() == 1:
                implicit_goal = implicit_goal.expand(batch)
            if implicit_goal.numel() != batch:
                raise ValueError("use_implicit_goal_state must be scalar or [batch]")
        terminal = torch.where(implicit_goal[:, None], goal, controls[:, -1, :])
        controls = torch.cat((controls[:, :-1, :], terminal[:, None, :]), dim=1)

    implicit_cuda_mode = use_implicit_goal_state is None or bool(
        torch.as_tensor(use_implicit_goal_state, device=knots.device, dtype=torch.bool).all().item()
    )
    uniform_full_horizon = bool((steps == output.position.shape[1]).all().item())
    if (
        bspline_degree == 3 and start is not None and goal is not None
        and implicit_cuda_mode and uniform_full_horizon
        and (output.position.shape[1] - 1) % (knots.shape[1] + 4) == 0
    ):
        position, velocity, acceleration, jerk = _cubic_boundary_spline(
            knots, current_state, goal_state, interpolation_dt, output.position.shape[1],
            start_idx=start_idx, goal_idx=goal_idx,
        )
        output.position = position
        output.velocity = velocity
        output.acceleration = acceleration
        output.jerk = jerk
        output.dt = torch.as_tensor(interpolation_dt, device=knots.device, dtype=knots.dtype).expand(batch)
        output.knot = input_trajectory.knot
        output.knot_dt = input_trajectory.knot_dt
        output.control_space = input_trajectory.control_space
        return output

    # Keep output-buffer semantics: fill every batch row, then extend the
    # final valid knot through any shared maximum-horizon tail.
    positions = []
    for item in range(batch):
        count = int(steps[item].item())
        basis = _clamped_uniform_bspline_basis(count, controls.shape[1], bspline_degree, controls)
        value = basis @ controls[item]
        if count < output.position.shape[1]:
            value = torch.cat((value, value[-1:].expand(output.position.shape[1] - count, dof)), dim=0)
        positions.append(value)
    output.position = torch.stack(positions, dim=0)
    result = output.finite_difference(interpolation_dt)
    result.knot = input_trajectory.knot
    result.knot_dt = input_trajectory.knot_dt
    result.control_space = input_trajectory.control_space
    return result


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

import torch
from .trajectory_config import BSplineLaunchCfg, LegacyTrajectoryLaunchCfg, TrajectoryKernelCfg

get_runtime = launch_kernel = None


def _dt(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def launch_integration_acceleration_kernel(out_position: torch.Tensor, out_velocity: torch.Tensor, out_acceleration: torch.Tensor, out_jerk: torch.Tensor, u_acc: torch.Tensor, start_position: torch.Tensor, start_velocity: torch.Tensor, start_acceleration: torch.Tensor, start_idx: torch.Tensor, traj_dt: torch.Tensor, batch_size: int, horizon: int, dof: int, use_rk2: bool = True):
    dt = _dt(traj_dt, u_acc).reshape(-1)[0]
    acceleration = u_acc
    velocity = torch.cat((start_velocity[:, None], start_velocity[:, None] + torch.cumsum(acceleration[:, :-1] * dt, dim=1)), dim=1)
    if use_rk2:
        position_delta = velocity * dt + 0.5 * acceleration * dt.square()
    else:
        position_delta = velocity * dt
    position = start_position[:, None] + torch.cumsum(position_delta, dim=1)
    jerk = torch.diff(acceleration, dim=1, prepend=start_acceleration[:, None]) / dt
    out_position.copy_(position)
    out_velocity.copy_(velocity)
    out_acceleration.copy_(acceleration)
    out_jerk.copy_(jerk)
    return [out_position, out_velocity, out_acceleration, out_jerk]


def launch_differentiation_position_forward_kernel(out_position: torch.Tensor, out_velocity: torch.Tensor, out_acceleration: torch.Tensor, out_jerk: torch.Tensor, out_dt: torch.Tensor, u_position: torch.Tensor, start_position: torch.Tensor, start_velocity: torch.Tensor, start_acceleration: torch.Tensor, goal_position: torch.Tensor, goal_velocity: torch.Tensor, goal_acceleration: torch.Tensor, start_idx: torch.Tensor, goal_idx: torch.Tensor, traj_dt: torch.Tensor, use_implicit_goal_state: torch.Tensor, batch_size: int, horizon: int, dof: int):
    dt = _dt(traj_dt, u_position).reshape(-1)[0]
    position = out_position.clone()
    position[:, 2:-2].copy_(u_position)
    position[:, :2].copy_(start_position[:, None])
    position[:, -2:].copy_(goal_position[:, None])
    velocity = torch.gradient(position, spacing=(dt,), dim=(1,))[0]
    acceleration = torch.gradient(velocity, spacing=(dt,), dim=(1,))[0]
    jerk = torch.gradient(acceleration, spacing=(dt,), dim=(1,))[0]
    out_position.copy_(position); out_velocity.copy_(velocity); out_acceleration.copy_(acceleration); out_jerk.copy_(jerk)
    if out_dt.numel(): out_dt.fill_(float(dt))
    return [out_position, out_velocity, out_acceleration, out_jerk, out_dt]


def launch_differentiation_position_backward_kernel(out_grad_position: torch.Tensor, grad_position: torch.Tensor, grad_velocity: torch.Tensor, grad_acceleration: torch.Tensor, grad_jerk: torch.Tensor, traj_dt: torch.Tensor, dt_idx: torch.Tensor, use_implicit_goal_state: torch.Tensor, batch_size: int, horizon: int, dof: int):
    out_grad_position.copy_(grad_position[:, 2:-2])
    return out_grad_position


def _unsupported(name):
    raise NotImplementedError(f"{name} raw CUDA B-spline kernel is unavailable; use curobo._src.util.trajectory")

def launch_bspline_interpolation_forward_kernel(
    out_position: torch.Tensor, out_velocity: torch.Tensor, out_acceleration: torch.Tensor, out_jerk: torch.Tensor, out_dt: torch.Tensor, u_position: torch.Tensor,
    start_position: torch.Tensor, start_velocity: torch.Tensor, start_acceleration: torch.Tensor, start_jerk: torch.Tensor,
    goal_position: torch.Tensor, goal_velocity: torch.Tensor, goal_acceleration: torch.Tensor, goal_jerk: torch.Tensor, start_idx: torch.Tensor,
    goal_idx: torch.Tensor, traj_dt: torch.Tensor, use_implicit_goal_state: torch.Tensor, batch_size: int, horizon: int, dof: int,
    n_knots: int, bspline_degree: int,
):
    del (out_position, out_velocity, out_acceleration, out_jerk, out_dt,
         u_position, start_position, start_velocity, start_acceleration,
         start_jerk, goal_position, goal_velocity, goal_acceleration, goal_jerk,
         start_idx, goal_idx, traj_dt, use_implicit_goal_state, batch_size,
         horizon, dof, n_knots, bspline_degree)
    return _unsupported("launch_bspline_interpolation_forward_kernel")


def launch_bspline_interpolation_backward_kernel(
    out_grad_position: torch.Tensor, grad_position: torch.Tensor, grad_velocity: torch.Tensor, grad_acceleration: torch.Tensor,
    grad_jerk: torch.Tensor, traj_dt: torch.Tensor, dt_idx: torch.Tensor, use_implicit_goal_state: torch.Tensor, batch_size: int,
    padded_horizon: int, dof: int, n_knots: int, bspline_degree: int, use_direct_polynomial: bool,
):
    del (out_grad_position, grad_position, grad_velocity, grad_acceleration,
         grad_jerk, traj_dt, dt_idx, use_implicit_goal_state, batch_size,
         padded_horizon, dof, n_knots, bspline_degree, use_direct_polynomial)
    return _unsupported("launch_bspline_interpolation_backward_kernel")


def launch_bspline_interpolation_single_dt_kernel(
    out_position: torch.Tensor, out_velocity: torch.Tensor, out_acceleration: torch.Tensor, out_jerk: torch.Tensor, out_dt: torch.Tensor, knots: torch.Tensor,
    knot_dt: torch.Tensor, start_position: torch.Tensor, start_velocity: torch.Tensor, start_acceleration: torch.Tensor, start_jerk: torch.Tensor,
    goal_position: torch.Tensor, goal_velocity: torch.Tensor, goal_acceleration: torch.Tensor, goal_jerk: torch.Tensor, start_idx: torch.Tensor,
    goal_idx: torch.Tensor, interpolation_dt: torch.Tensor, use_implicit_goal_state: torch.Tensor, interpolation_horizon: torch.Tensor,
    batch_size: int, max_out_tsteps: int, dof: int, n_knots: int, bspline_degree: int,
):
    del (out_position, out_velocity, out_acceleration, out_jerk, out_dt, knots,
         knot_dt, start_position, start_velocity, start_acceleration, start_jerk,
         goal_position, goal_velocity, goal_acceleration, goal_jerk, start_idx,
         goal_idx, interpolation_dt, use_implicit_goal_state, interpolation_horizon,
         batch_size, max_out_tsteps, dof, n_knots, bspline_degree)
    return _unsupported("launch_bspline_interpolation_single_dt_kernel")

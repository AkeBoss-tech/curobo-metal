"""Upstream-compatible trajectory kernel facades for CPU and MPS."""

import torch

from curobo._src.curobolib.backends import trajectory as trajectory_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


def get_bspline_interpolation(
    input_trajectory: JointState,
    output_trajectory: JointState,
    interpolation_dt: torch.Tensor,
    current_state: JointState,
    goal_state: JointState,
    start_idx: torch.Tensor,
    goal_idx: torch.Tensor,
    use_implicit_goal_state: torch.Tensor,
    interpolated_horizon: torch.Tensor,
    bspline_degree=4,
):
    raise NotImplementedError(
        "The raw CUDA B-spline boundary kernel is unavailable on Metal; use "
        "curobo._src.util.trajectory interpolation or MotionGen interpolation"
    )


class CliqueTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, u_act, start_position, start_velocity, start_acceleration,
        goal_position, goal_velocity, goal_acceleration, start_idx, goal_idx,
        out_position, out_velocity, out_acceleration, out_jerk, out_dt, traj_dt,
        use_implicit_goal_state, out_grad_position,
    ):
        trajectory_cu.launch_differentiation_position_forward_kernel(
            out_position, out_velocity, out_acceleration, out_jerk, out_dt,
            u_act, start_position, start_velocity, start_acceleration,
            goal_position, goal_velocity, goal_acceleration, start_idx, goal_idx,
            traj_dt, use_implicit_goal_state, out_position.shape[0],
            out_position.shape[1], out_position.shape[-1],
        )
        ctx.save_for_backward(traj_dt, out_grad_position, goal_idx, use_implicit_goal_state)
        return out_position, out_velocity, out_acceleration, out_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        traj_dt, out_grad, goal_idx, implicit = ctx.saved_tensors
        trajectory_cu.launch_differentiation_position_backward_kernel(
            out_grad, grad_out_p, grad_out_v, grad_out_a, grad_out_j, traj_dt,
            goal_idx, implicit, grad_out_p.shape[0], grad_out_p.shape[1],
            grad_out_p.shape[2],
        )
        return (out_grad,) + (None,) * 16


class AccelerationTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, u_act, start_position, start_velocity, start_acceleration,
        start_idx, out_position, out_velocity, out_acceleration, out_jerk,
        traj_dt, out_grad_position,
    ):
        trajectory_cu.launch_integration_acceleration_kernel(
            out_position, out_velocity, out_acceleration, out_jerk, u_act,
            start_position, start_velocity, start_acceleration, start_idx,
            traj_dt, out_position.shape[0], out_position.shape[1],
            out_position.shape[-1],
        )
        return out_position, out_velocity, out_acceleration, out_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        del ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j
        raise NotImplementedError(
            "The CUDA acceleration-kernel custom VJP is unavailable; use the "
            "production trajectory optimizer's differentiable PyTorch rollout"
        )


class BSplineIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_jerk,
        goal_position,
        goal_velocity,
        goal_acceleration,
        goal_jerk,
        start_idx,
        goal_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        out_dt,
        traj_dt,
        use_implicit_goal_state,
        out_grad_position,
        bspline_degree,
        use_flat_gradient=False,
    ):
        del (
            ctx,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            goal_velocity,
            goal_acceleration,
            goal_jerk,
            start_idx,
            goal_idx,
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            out_dt,
            traj_dt,
            use_implicit_goal_state,
            out_grad_position,
            bspline_degree,
            use_flat_gradient,
        )
        raise NotImplementedError(
            "BSplineIdxKernel is a CUDA packed-buffer ABI; use portable "
            "trajectory interpolation instead"
        )

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        del ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j
        raise NotImplementedError(
            "BSplineIdxKernel's packed CUDA VJP is unavailable on Metal; use "
            "portable trajectory interpolation with ordinary PyTorch autograd"
        )

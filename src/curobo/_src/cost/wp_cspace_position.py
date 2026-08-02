"""Portable tensor implementation of cuRobo's position c-space cost bridge.

The pinned V2 module exposes both a ``torch.autograd.Function`` and a raw
Warp kernel.  The former is useful to Python callers independently of Warp;
it now runs using ordinary PyTorch CPU/MPS tensors while retaining the
caller-owned diagnostic buffers and first-order backward convention.  The
raw kernel name remains an explicit CUDA/Warp boundary.
"""

from __future__ import annotations

import torch


def _require_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        received = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"{name}.shape: {received} != {shape}")


def _bound_cost_and_grad(
    value: torch.Tensor, bounds: torch.Tensor, activation: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V2's elementwise activated squared-hinge cost and gradient."""
    lower = bounds[0] + activation * (bounds[1] - bounds[0])
    upper = bounds[1] - activation * (bounds[1] - bounds[0])
    delta = torch.where(value < lower, value - lower, torch.where(value > upper, value - upper, torch.zeros_like(value)))
    return 0.5 * weight * delta.square(), weight * delta


class PositionCSpaceFunction(torch.autograd.Function):
    """Differentiable portable equivalent of the V2 packed position-cost call.

    It accepts the pinned positional argument layout, writes ``out_cost``,
    ``out_gp`` and ``out_gtau`` in-place, and returns ``out_cost``.  Like the
    upstream custom bridge, gradients are only exposed for position and
    torque.  The buffer ABI is ordinary tensors rather than CUDA/Warp memory.
    """

    @classmethod
    def apply(cls, *args, **kwargs):
        # Keep the prior explicit portable boundary for a legacy/raw call
        # while allowing the complete pinned tensor signature below.
        if len(args) != 20 or kwargs:
            raise RuntimeError(
                "PositionCSpaceFunction raw Warp/CUDA ABI requires the complete "
                "20-argument tensor signature; use PositionCSpaceCost for high-level calls"
            )
        return super().apply(*args)

    @staticmethod
    def forward(
        ctx,
        pos: torch.Tensor,
        joint_torque: torch.Tensor,
        target_joint_position: torch.Tensor,
        idxs_target_joint_position: torch.Tensor,
        p_l: torch.Tensor,
        effort_limit: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        cspace_target_weight: torch.Tensor,
        cspace_target_dof_weight: torch.Tensor,
        squared_l2_regularization_weight: torch.Tensor,
        current_position: torch.Tensor,
        current_velocity: torch.Tensor,
        idxs_current_state: torch.Tensor,
        v_b: torch.Tensor,
        state_dt: torch.Tensor,
        out_cost: torch.Tensor,
        out_gp: torch.Tensor,
        out_gtau: torch.Tensor,
        use_grad_input: bool,
    ) -> torch.Tensor:
        if pos.ndim != 3:
            raise ValueError("pos must have shape [batch, horizon, dof]")
        batch, horizon, dof = pos.shape
        _require_shape("joint_torque", joint_torque, (batch, horizon, dof))
        _require_shape("p_l", p_l, (2, dof))
        _require_shape("effort_limit", effort_limit, (2, dof))
        _require_shape("v_b", v_b, (2, dof))
        _require_shape("weight", weight, (2,))
        _require_shape("activation_distance", activation_distance, (2,))
        _require_shape("cspace_target_weight", cspace_target_weight, (1,))
        _require_shape("cspace_target_dof_weight", cspace_target_dof_weight, (dof,))
        _require_shape("squared_l2_regularization_weight", squared_l2_regularization_weight, (2,))
        _require_shape("out_cost", out_cost, (batch, horizon, dof))
        _require_shape("out_gp", out_gp, (batch, horizon, dof))
        _require_shape("out_gtau", out_gtau, (batch, horizon, dof))
        if target_joint_position.ndim == 1:
            target_joint_position = target_joint_position.unsqueeze(0)
        if target_joint_position.ndim != 2 or target_joint_position.shape[1] != dof:
            raise ValueError("target_joint_position must have shape [goals, dof]")
        if idxs_target_joint_position.ndim == 2 and idxs_target_joint_position.shape[1] == 1:
            idxs_target_joint_position = idxs_target_joint_position[:, 0]
        if idxs_target_joint_position.ndim != 1 or idxs_target_joint_position.shape[0] != batch:
            raise ValueError("idxs_target_joint_position must have shape [batch]")
        if idxs_target_joint_position.dtype not in (torch.int32, torch.int64):
            raise TypeError("idxs_target_joint_position must be int32 or int64")
        if current_position.ndim == 1:
            current_position = current_position.unsqueeze(0)
        if current_velocity.ndim == 1:
            current_velocity = current_velocity.unsqueeze(0)
        if current_position.ndim != 2 or tuple(current_position.shape[1:]) != (dof,):
            raise ValueError("current_position must have shape [states, dof]")
        if current_velocity.shape != current_position.shape:
            raise ValueError("current_velocity must match current_position")
        if idxs_current_state.ndim == 2 and idxs_current_state.shape[1] == 1:
            idxs_current_state = idxs_current_state[:, 0]
        if idxs_current_state.ndim != 1 or idxs_current_state.shape[0] != batch:
            raise ValueError("idxs_current_state must have shape [batch]")
        if idxs_current_state.dtype not in (torch.int32, torch.int64):
            raise TypeError("idxs_current_state must be int32 or int64")
        if state_dt.ndim != 1:
            raise ValueError("state_dt must be rank 1")
        if any(value.device != pos.device for value in (joint_torque, target_joint_position, idxs_target_joint_position, p_l, effort_limit, weight, activation_distance, cspace_target_weight, cspace_target_dof_weight, squared_l2_regularization_weight, current_position, current_velocity, idxs_current_state, v_b, state_dt, out_cost, out_gp, out_gtau)):
            raise ValueError("all c-space position tensors must share pos.device")
        if bool(((idxs_target_joint_position < 0) | (idxs_target_joint_position >= target_joint_position.shape[0])).any()):
            raise ValueError("idxs_target_joint_position contains an out-of-range goal index")
        if bool(((idxs_current_state < 0) | (idxs_current_state >= current_position.shape[0])).any()):
            raise ValueError("idxs_current_state contains an out-of-range state index")
        if idxs_current_state.numel() and state_dt.numel() <= int(idxs_current_state.max()):
            raise ValueError("state_dt must provide one value for each indexed current state")

        # The dynamic position interval is tightened only for positive dt,
        # exactly like V2's Warp kernel.  Broadcast it over the horizon.
        current_idx = idxs_current_state.to(dtype=torch.long)
        current_q = current_position.index_select(0, current_idx).unsqueeze(1)
        current_v = current_velocity.index_select(0, current_idx).unsqueeze(1)
        dt = state_dt.index_select(0, current_idx).reshape(batch, 1, 1).to(dtype=pos.dtype)
        position_bounds = p_l.to(dtype=pos.dtype).clone()
        position_bounds = position_bounds.unsqueeze(0).expand(batch, -1, -1).clone()
        dynamic_lower = current_q[:, 0] + v_b[0].to(pos) * dt[:, 0]
        dynamic_upper = current_q[:, 0] + v_b[1].to(pos) * dt[:, 0]
        positive_dt = dt[:, 0] > 0
        position_bounds[:, 0] = torch.where(positive_dt, torch.maximum(position_bounds[:, 0], dynamic_lower), position_bounds[:, 0])
        position_bounds[:, 1] = torch.where(positive_dt, torch.minimum(position_bounds[:, 1], dynamic_upper), position_bounds[:, 1])
        activated_lower = position_bounds[:, 0] + activation_distance[0].to(pos) * (position_bounds[:, 1] - position_bounds[:, 0])
        activated_upper = position_bounds[:, 1] - activation_distance[0].to(pos) * (position_bounds[:, 1] - position_bounds[:, 0])
        p_delta = torch.where(pos < activated_lower.unsqueeze(1), pos - activated_lower.unsqueeze(1), torch.where(pos > activated_upper.unsqueeze(1), pos - activated_upper.unsqueeze(1), torch.zeros_like(pos)))
        cost = 0.5 * weight[0].to(pos) * p_delta.square()
        grad_position = weight[0].to(pos) * p_delta

        torque_cost, grad_torque = _bound_cost_and_grad(
            joint_torque, effort_limit.to(pos), activation_distance[1].to(pos), weight[1].to(pos)
        )
        cost = cost + torque_cost

        target = target_joint_position.index_select(0, idxs_target_joint_position.to(dtype=torch.long)).unsqueeze(1).to(pos)
        target_weight = cspace_target_weight.to(pos) * cspace_target_dof_weight.to(pos)
        target_error = pos - target
        cost = cost + target_weight * target_error.square()
        grad_position = grad_position + 2.0 * target_weight * target_error

        # V2 retimes these two regularizers so their physical-time scaling is
        # stable when a rollout is sampled at a different dt.
        safe_dt = torch.where(dt > 0, dt, torch.ones_like(dt))
        implied_v = (pos - current_q) / safe_dt
        implied_a = (implied_v - current_v) / safe_dt
        vel_weight = squared_l2_regularization_weight[0].to(pos) * dt.clamp_min(0)
        acc_weight = squared_l2_regularization_weight[1].to(pos) * dt.clamp_min(0).square()
        cost = cost + 0.5 * vel_weight * implied_v.square() + 0.5 * acc_weight * implied_a.square()
        grad_position = grad_position + vel_weight * implied_v / safe_dt + acc_weight * implied_a / safe_dt.square()

        out_cost.copy_(cost)
        out_gp.copy_(grad_position)
        out_gtau.copy_(grad_torque)
        ctx.use_grad_input = bool(use_grad_input)
        ctx.save_for_backward(grad_position, grad_torque)
        return out_cost

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_cost: torch.Tensor | None):
        grad_position, grad_torque = ctx.saved_tensors
        if grad_out_cost is not None and ctx.use_grad_input:
            grad_position = grad_position * grad_out_cost
            grad_torque = grad_torque * grad_out_cost
        return grad_position, grad_torque, *(None for _ in range(18))


def forward_cspace_position_warp(*args, **kwargs):
    """Raw Warp kernel ABI, intentionally unavailable on CPU/MPS."""
    del args, kwargs
    raise NotImplementedError(
        "forward_cspace_position_warp requires cuRobo's CUDA/Warp kernel ABI; "
        "use PositionCSpaceFunction or PositionCSpaceCost on CPU/MPS"
    )


__all__ = ["PositionCSpaceFunction", "forward_cspace_position_warp"]

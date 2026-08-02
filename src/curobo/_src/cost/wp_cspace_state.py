"""Portable tensor implementation of cuRobo's state c-space cost bridge."""

from __future__ import annotations

import torch

from .wp_cspace_position import _bound_cost_and_grad, _require_shape


class StateCSpaceFunction(torch.autograd.Function):
    """CPU/MPS tensor implementation of V2's packed state-cost function.

    The positional signature and caller-owned diagnostics match the public V2
    custom autograd bridge.  It supports V2's optional retiming flags here,
    even though the higher-level portable ``CSpaceCostCfg`` deliberately
    rejects CUDA-specific rollout retiming configuration.
    """

    @classmethod
    def apply(cls, *args, **kwargs):
        if len(args) != 28 or kwargs:
            raise RuntimeError(
                "StateCSpaceFunction raw Warp/CUDA ABI requires the complete "
                "28-argument tensor signature; use StateCSpaceCost for high-level calls"
            )
        return super().apply(*args)

    @staticmethod
    def forward(
        ctx,
        pos: torch.Tensor,
        vel: torch.Tensor,
        acc: torch.Tensor,
        jerk: torch.Tensor,
        joint_torque: torch.Tensor,
        state_dt: torch.Tensor,
        target_joint_position: torch.Tensor,
        idxs_target_joint_position: torch.Tensor,
        p_b: torch.Tensor,
        v_b: torch.Tensor,
        a_b: torch.Tensor,
        j_b: torch.Tensor,
        effort_limit: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        squared_l2_regularization_weights: torch.Tensor,
        cspace_target_weight: torch.Tensor,
        cspace_non_terminal_weight_factor: torch.Tensor,
        cspace_target_dof_weight: torch.Tensor,
        out_cost: torch.Tensor,
        out_gp: torch.Tensor,
        out_gv: torch.Tensor,
        out_ga: torch.Tensor,
        out_gj: torch.Tensor,
        out_gtau: torch.Tensor,
        retime_weights: bool,
        retime_regularization_weights: bool,
        use_grad_input: bool,
    ) -> torch.Tensor:
        if pos.ndim != 3:
            raise ValueError("pos must have shape [batch, horizon, dof]")
        batch, horizon, dof = pos.shape
        for name, value in (("vel", vel), ("acc", acc), ("jerk", jerk), ("joint_torque", joint_torque)):
            _require_shape(name, value, (batch, horizon, dof))
        _require_shape("state_dt", state_dt, (batch,))
        for name, value in (("p_b", p_b), ("v_b", v_b), ("a_b", a_b), ("j_b", j_b), ("effort_limit", effort_limit)):
            _require_shape(name, value, (2, dof))
        for name, value, shape in (
            ("weight", weight, (5,)),
            ("activation_distance", activation_distance, (5,)),
            ("squared_l2_regularization_weights", squared_l2_regularization_weights, (5,)),
            ("cspace_target_weight", cspace_target_weight, (1,)),
            ("cspace_non_terminal_weight_factor", cspace_non_terminal_weight_factor, (1,)),
            ("cspace_target_dof_weight", cspace_target_dof_weight, (dof,)),
        ):
            _require_shape(name, value, shape)
        for name, value in (("out_cost", out_cost), ("out_gp", out_gp), ("out_gv", out_gv), ("out_ga", out_ga), ("out_gj", out_gj), ("out_gtau", out_gtau)):
            _require_shape(name, value, (batch, horizon, dof))
        if target_joint_position.ndim == 1:
            target_joint_position = target_joint_position.unsqueeze(0)
        if target_joint_position.ndim != 2 or target_joint_position.shape[1] != dof:
            raise ValueError("target_joint_position must have shape [goals, dof]")
        if idxs_target_joint_position.ndim != 1 or idxs_target_joint_position.shape != (batch,):
            raise ValueError("idxs_target_joint_position must have shape [batch]")
        if idxs_target_joint_position.dtype not in (torch.int32, torch.int64):
            raise TypeError("idxs_target_joint_position must be int32 or int64")
        values = (
            vel, acc, jerk, joint_torque, state_dt, target_joint_position, idxs_target_joint_position, p_b, v_b, a_b,
            j_b, effort_limit, weight, activation_distance, squared_l2_regularization_weights,
            cspace_target_weight, cspace_non_terminal_weight_factor, cspace_target_dof_weight,
            out_cost, out_gp, out_gv, out_ga, out_gj, out_gtau,
        )
        if any(value.device != pos.device for value in values):
            raise ValueError("all c-space state tensors must share pos.device")
        if bool(((idxs_target_joint_position < 0) | (idxs_target_joint_position >= target_joint_position.shape[0])).any()):
            raise ValueError("idxs_target_joint_position contains an out-of-range goal index")

        dt = state_dt.reshape(batch, 1, 1).to(pos)
        # V2 weights the derivative bound terms by dt^n when asked to retime.
        bound_weight = weight.to(pos).clone()
        if retime_weights:
            scale = torch.stack((torch.ones_like(state_dt), state_dt, state_dt.square(), state_dt.pow(3), torch.ones_like(state_dt)), dim=-1)
            # These values vary per batch, so retain them as a [B,1,5] tensor.
            bound_weight = bound_weight.reshape(1, 1, 5) * scale.reshape(batch, 1, 5).to(pos)
        else:
            bound_weight = bound_weight.reshape(1, 1, 5)
        channels = (pos, vel, acc, jerk, joint_torque)
        limits = (p_b, v_b, a_b, j_b, effort_limit)
        cost = torch.zeros_like(pos)
        gradients = []
        for term, (channel, bounds) in enumerate(zip(channels, limits)):
            component_cost, component_grad = _bound_cost_and_grad(
                channel, bounds.to(pos), activation_distance[term].to(pos), bound_weight[..., term].unsqueeze(-1)
            )
            cost = cost + component_cost
            gradients.append(component_grad)

        target = target_joint_position.index_select(0, idxs_target_joint_position.to(dtype=torch.long)).unsqueeze(1).to(pos)
        target_weight = cspace_target_weight.to(pos) * cspace_target_dof_weight.to(pos)
        if horizon > 1:
            target_weight = target_weight.reshape(1, 1, dof).expand(batch, horizon, dof).clone()
            target_weight[:, :-1] *= cspace_non_terminal_weight_factor.to(pos)
        target_error = pos - target
        cost = cost + target_weight * target_error.square()
        gradients[0] = gradients[0] + 2.0 * target_weight * target_error

        regularizer = squared_l2_regularization_weights.to(pos).clone()
        if retime_regularization_weights:
            reg_scale = torch.stack((state_dt, state_dt.square(), state_dt.pow(3), torch.ones_like(state_dt), state_dt), dim=-1)
            regularizer = regularizer.reshape(1, 1, 5) * reg_scale.reshape(batch, 1, 5).to(pos)
        else:
            regularizer = regularizer.reshape(1, 1, 5)
        for term, channel in enumerate((vel, acc, jerk, joint_torque)):
            local_weight = regularizer[..., term].unsqueeze(-1)
            cost = cost + 0.5 * local_weight * channel.square()
            gradients[term + 1] = gradients[term + 1] + local_weight * channel
        energy = joint_torque * vel * dt
        energy_weight = regularizer[..., 4].unsqueeze(-1)
        cost = cost + energy_weight * energy.square()
        gradients[4] = gradients[4] + 2.0 * energy_weight * energy * vel * dt
        gradients[1] = gradients[1] + 2.0 * energy_weight * energy * joint_torque * dt

        out_cost.copy_(cost)
        for output, gradient in zip((out_gp, out_gv, out_ga, out_gj, out_gtau), gradients):
            output.copy_(gradient)
        ctx.use_grad_input = bool(use_grad_input)
        ctx.save_for_backward(*gradients)
        return out_cost

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_cost: torch.Tensor | None):
        gradients = ctx.saved_tensors
        if grad_out_cost is not None and ctx.use_grad_input:
            gradients = tuple(gradient * grad_out_cost for gradient in gradients)
        return *gradients, *(None for _ in range(23))


def forward_cspace_state_warp(*args, **kwargs):
    """Raw Warp kernel ABI, intentionally unavailable on CPU/MPS."""
    del args, kwargs
    raise NotImplementedError(
        "forward_cspace_state_warp requires cuRobo's CUDA/Warp kernel ABI; "
        "use StateCSpaceFunction or StateCSpaceCost on CPU/MPS"
    )


__all__ = ["StateCSpaceFunction", "forward_cspace_state_warp"]

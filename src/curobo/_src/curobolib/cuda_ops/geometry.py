"""Portable autograd facade for the low-level self-collision operator."""

from typing import Optional

import torch

from curobo._src.curobolib.backends import geometry as geometry_cu


class SelfCollisionDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        robot_spheres: torch.Tensor,
        out_distance: torch.Tensor,
        out_vec: torch.Tensor,
        pair_distance: torch.Tensor,
        sparse_idx: torch.Tensor,
        weight: torch.Tensor,
        sphere_padding: torch.Tensor,
        pair_locations: torch.Tensor,
        block_batch_max_value: torch.Tensor,
        block_batch_max_index: torch.Tensor,
        num_blocks_per_batch: int,
        max_threads_per_block: int,
        store_pair_distance: bool,
        return_loss: bool,
    ):
        b, h, num_spheres, _ = robot_spheres.shape
        geometry_cu.self_collision_distance(
            out_distance, out_vec, pair_distance, sparse_idx, robot_spheres,
            sphere_padding, weight, pair_locations, block_batch_max_value,
            block_batch_max_index, num_blocks_per_batch, max_threads_per_block,
            b, h, num_spheres, pair_locations.shape[0], store_pair_distance,
            robot_spheres.requires_grad,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_vec)
        return out_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_distance: Optional[torch.Tensor]):
        sphere_grad = None
        if grad_out_distance is not None and ctx.needs_input_grad[0]:
            (sphere_grad,) = ctx.saved_tensors
            if ctx.return_loss:
                sphere_grad = sphere_grad * grad_out_distance[..., None, None]
        return (sphere_grad,) + (None,) * 13

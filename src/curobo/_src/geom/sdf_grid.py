"""Deprecated integer-grid SDF lookup and finite-difference gradient."""

import torch


def lookup_distance(pt, dist_matrix_flat, num_voxels):
    ind_pt = (
        pt[..., 0] * (num_voxels[1] * num_voxels[2])
        + pt[..., 1] * num_voxels[2]
        + pt[..., 2]
    )
    return dist_matrix_flat[ind_pt]


def compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist):
    gradients = []
    for axis in range(3):
        point_before = pt.clone()
        point_after = pt.clone()
        point_before[..., axis] -= 1
        point_after[..., axis] += 1
        point_before[point_before < 0] = 0
        distance_before = lookup_distance(point_before, dist_matrix_flat, num_voxels)
        distance_after = lookup_distance(point_after, dist_matrix_flat, num_voxels)
        choose_before = distance_before < distance_after
        direction = torch.where(choose_before, -1, 1)
        delta = torch.where(
            choose_before, distance_before - dist, distance_after - dist
        )
        gradients.append(delta / direction)
    return torch.stack(gradients, dim=-1)


class SDFGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pt, dist_matrix_flat, num_voxels):
        pt = pt.to(dtype=torch.int64)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)
        ctx.save_for_backward(pt, dist_matrix_flat, num_voxels, dist)
        return dist.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        pt, dist_matrix_flat, num_voxels, dist = ctx.saved_tensors
        grad_pt = grad_matrix_flat = grad_voxels = None
        if ctx.needs_input_grad[0]:
            grad_pt = grad_output * compute_sdf_gradient(
                pt.to(dtype=torch.int64), dist_matrix_flat, num_voxels, dist
            )
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("SDFGrid: Can't get gradient w.r.t. dist_matrix")
        if ctx.needs_input_grad[2]:
            raise NotImplementedError("SDFGrid: Can't get gradient w.r.t. num_voxels")
        return grad_pt, grad_matrix_flat, grad_voxels


__all__ = ["SDFGrid", "compute_sdf_gradient", "lookup_distance"]

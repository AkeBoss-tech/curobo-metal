"""Differentiable trilinear dense SDF sampling."""

import torch
import torch.nn.functional as F


def lookup_distance(pt, dist_matrix_flat, num_voxels):
    volume = dist_matrix_flat.reshape(tuple(int(x) for x in num_voxels))
    index = tuple(torch.round(torch.as_tensor(pt)).long().unbind(-1))
    return volume[index]


def compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist=None):
    point = torch.as_tensor(pt, dtype=dist_matrix_flat.dtype, device=dist_matrix_flat.device).requires_grad_(True)
    value = SDFGrid.apply(point, dist_matrix_flat, torch.as_tensor(num_voxels, device=point.device))
    return torch.autograd.grad(value.sum(), point, create_graph=True)[0]


class SDFGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, dist_matrix_flat, num_voxels):
        shape = tuple(int(x) for x in num_voxels.tolist())
        volume = dist_matrix_flat.reshape(1, 1, *shape)
        scale = points.new_tensor([max(shape[2]-1,1), max(shape[1]-1,1), max(shape[0]-1,1)])
        grid = (points / scale * 2 - 1).reshape(1, 1, 1, -1, 3)
        ctx.save_for_backward(points, dist_matrix_flat, num_voxels)
        return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(points.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        points, dist, shape = ctx.saved_tensors
        with torch.enable_grad():
            p = points.detach().requires_grad_(True)
            value = SDFGrid.forward(type("C", (), {"save_for_backward": lambda *args: None})(), p, dist, shape)
        return torch.autograd.grad(value, p, grad_output)[0], None, None


__all__ = ["SDFGrid", "compute_sdf_gradient", "lookup_distance"]

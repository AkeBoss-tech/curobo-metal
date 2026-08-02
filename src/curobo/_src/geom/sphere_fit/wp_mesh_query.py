"""Torch replacement for V2's Warp mesh-query wrapper.

The query contract is preserved (negative SDF inside a declared-watertight
mesh), but mesh ids and Warp BVH ownership are intentionally absent.  Queries
use the production differentiable triangle-distance operator instead.
"""
from __future__ import annotations

import torch

from curobo_metal.ops.world_collision import Mesh as TorchMesh, mesh_distance


class WarpMeshQuery:
    def __init__(self, mesh, device):
        self.device = torch.device(device)
        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device)
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError("mesh.vertices must have shape [vertices, 3]")
        if faces.ndim != 2 or faces.shape[-1] != 3:
            raise ValueError("mesh.faces must have shape [triangles, 3]")
        self.mesh = TorchMesh(vertices, faces, bool(getattr(mesh, "is_watertight", False)))
        bounds = torch.stack((vertices.amin(0), vertices.amax(0)))
        self._max_distance = float(torch.linalg.vector_norm(bounds[1] - bounds[0]).item()) + 0.01

    def _points(self, points: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [points, 3]")
        if points.device != self.device:
            raise ValueError("points must be on the query device")
        return points.contiguous()

    def query_sdf(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = self._points(points)
        eye = torch.eye(3, dtype=points.dtype, device=points.device).reshape(1, 1, 3, 3)
        zero = torch.zeros((1, 1, 3), dtype=points.dtype, device=points.device)
        result = mesh_distance(points, [self.mesh], zero, eye, signed=self.mesh.watertight)
        distance = result.reduced_distance.reshape(-1)
        gradient = result.reduced_gradient.reshape(-1, 3)
        # Warp returns the configured maximum for failed distant queries.  The
        # vectorized backend has no failure state, so clamp only unsigned scans
        # to retain a finite and useful portable result.
        if not self.mesh.watertight:
            distance = distance.clamp_max(self._max_distance)
        return distance, gradient

    def query_outside_mask(self, points: torch.Tensor) -> torch.Tensor:
        return self.query_sdf(points)[0] > 0

    def query_closest_point(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = self._points(points)
        distance, gradient = self.query_sdf(points)
        return points - distance.unsqueeze(-1) * gradient, distance


class WarpSphereSDFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, mesh_query: WarpMeshQuery) -> torch.Tensor:
        distance, gradient = mesh_query.query_sdf(points.detach())
        ctx.save_for_backward(gradient)
        return distance

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gradient,) = ctx.saved_tensors
        return grad_output.unsqueeze(-1) * gradient, None


__all__ = ["WarpMeshQuery", "WarpSphereSDFFunction"]

"""Portable batched 2-D convex hull utilities."""
from __future__ import annotations
from typing import Optional
import torch
from curobo._src.types.device_cfg import DeviceCfg

class ConvexPolygon2DHelper:
    def __init__(self, device_cfg: DeviceCfg = DeviceCfg()):
        self._cached_convex_hulls = None
        self._tensor_args = device_cfg

    def build_convex_hull(self, vertices: torch.Tensor, padding: Optional[float] = None):
        if vertices.ndim != 3 or vertices.shape[-1] != 2:
            raise ValueError("vertices must have shape [batch, vertices, 2]")
        hulls = [self._hull(v.detach(), padding) for v in vertices]
        width = max((len(h) for h in hulls), default=0)
        out = vertices.new_zeros((len(hulls), width, 2))
        for i, hull in enumerate(hulls):
            out[i, :len(hull)] = hull
            if len(hull) and len(hull) < width:
                out[i, len(hull):] = hull[-1]
        self._cached_convex_hulls = out

    @staticmethod
    def _hull(points: torch.Tensor, padding: Optional[float]) -> torch.Tensor:
        if len(points) < 3:
            return points
        order = sorted(range(len(points)), key=lambda i: (float(points[i, 0]), float(points[i, 1])))
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        lower, upper = [], []
        for i in order:
            while len(lower) >= 2 and float(cross(points[lower[-2]], points[lower[-1]], points[i])) <= 0:
                lower.pop()
            lower.append(i)
        for i in reversed(order):
            while len(upper) >= 2 and float(cross(points[upper[-2]], points[upper[-1]], points[i])) <= 0:
                upper.pop()
            upper.append(i)
        hull = points[(lower[:-1] + upper[:-1])]
        if padding and len(hull):
            direction = hull - hull.mean(0)
            hull = hull + float(padding) * direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return hull

    def compute_point_hull_distance(self, points: torch.Tensor, idxs_batch: torch.Tensor) -> torch.Tensor:
        if self._cached_convex_hulls is None:
            return torch.zeros_like(points[..., 0])
        hull = self._cached_convex_hulls[idxs_batch]
        a, b = hull, torch.roll(hull, -1, 1)
        edge = b-a
        rel = points.unsqueeze(-2)-a[:, None, None]
        t = (rel*edge[:, None, None]).sum(-1)/(edge.square().sum(-1)[:, None, None].clamp_min(1e-12))
        closest = a[:, None, None]+t.clamp(0, 1).unsqueeze(-1)*edge[:, None, None]
        distance = (points.unsqueeze(-2)-closest).norm(dim=-1).amin(-1)
        cross = edge[:, None, None, :, 0]*rel[..., 1]-edge[:, None, None, :, 1]*rel[..., 0]
        inside = (cross >= -1e-8).all(-1) | (cross <= 1e-8).all(-1)
        return torch.where(inside, -distance, distance)

__all__ = ["ConvexPolygon2DHelper"]

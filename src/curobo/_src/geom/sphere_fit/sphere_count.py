from __future__ import annotations

import numpy as np
import torch as _torch
from types import SimpleNamespace as trimesh


_BASE_DENSITY = 1.0 / 15.0
_MIN_SPHERES = 3
_MAX_SPHERES = 100


def _vertices(mesh):
    value = getattr(mesh, "vertices", None)
    if value is None:
        raise ValueError("mesh must expose vertices")
    if isinstance(value, _torch.Tensor):
        if value.ndim != 2 or value.shape[1] != 3 or len(value) < 1:
            raise ValueError("mesh vertices must have shape [V,3]")
        return value
    tensor = np.asarray(value, dtype=np.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or len(tensor) < 1:
        raise ValueError("mesh vertices must have shape [V,3]")
    return tensor


def estimate_sphere_count(
    mesh: trimesh.Trimesh,
    sphere_density: float = 1.0,
) -> int:
    bounds = getattr(mesh, "bounds", None)
    if bounds is None:
        vertices = _vertices(mesh)
        if isinstance(vertices, _torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    bbox_vol_cm3 = float(np.prod((np.asarray(bounds)[1] - np.asarray(bounds)[0]) * 100))
    n = int(sphere_density * _BASE_DENSITY * bbox_vol_cm3)
    max_spheres = max(int(_MAX_SPHERES * sphere_density), _MIN_SPHERES)
    return max(min(max_spheres, n), _MIN_SPHERES)

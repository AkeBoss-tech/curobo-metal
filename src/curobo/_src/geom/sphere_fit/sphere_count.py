import math
import torch


def _vertices(mesh):
    value = getattr(mesh, "vertices", None)
    if value is None:
        raise ValueError("mesh must expose vertices")
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3 or len(tensor) < 1:
        raise ValueError("mesh vertices must have shape [V,3]")
    return tensor


def estimate_sphere_count(mesh, sphere_density: float = 1.0):
    if sphere_density <= 0:
        raise ValueError("sphere_density must be positive")
    vertices = _vertices(mesh)
    extent = vertices.max(0).values - vertices.min(0).values
    aspect = extent.max() / extent.clamp_min(1e-9).min()
    return max(1, int(math.ceil(float(aspect) * sphere_density * 4)))

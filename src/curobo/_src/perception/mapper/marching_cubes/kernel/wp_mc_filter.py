"""Triangle filtering with a portable host implementation.

The two named kernels are retained as explicit Warp ABI boundaries.  The
public host helper has the same observable result on CPU and MPS without
requiring a CUDA stream or Warp buffer conversion.
"""

from __future__ import annotations

import torch

from curobo._src.util.warp import wp

try:
    import warp as _raw_wp
except ImportError:
    _raw_wp = None

if _raw_wp is not None:
    wp = _raw_wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_int32_tensors
from curobo._src.util.warp import get_warp_device_stream

from curobo._src.perception.mapper._portable import unsupported_kernel

if not hasattr(wp, "kernel"):
    class _PortableWarp:
        @staticmethod
        def kernel(function):
            return function

    wp = _PortableWarp()


@wp.kernel
def count_valid_triangles_kernel(
    triangles: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    n_triangles: wp.int32,
    min_area_sq: wp.float32,
    valid_count: wp.array(dtype=wp.int32),
):
    return unsupported_kernel(triangles, vertices, n_triangles, min_area_sq, valid_count)


@wp.kernel
def compact_valid_triangles_kernel(
    triangles_in: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    n_triangles: wp.int32,
    min_area_sq: wp.float32,
    flip_winding: wp.bool,
    output_counter: wp.array(dtype=wp.int32),
    triangles_out: wp.array(dtype=wp.int32),
):
    return unsupported_kernel(
        triangles_in,
        vertices,
        n_triangles,
        min_area_sq,
        flip_winding,
        output_counter,
        triangles_out,
    )


def filter_triangles(
    triangles: torch.Tensor,
    vertices: torch.Tensor,
    voxel_size: float,
    flip_winding: bool = True,
) -> torch.Tensor:
    """Discard invalid and degenerate triangle rows on CPU or MPS.

    This reproduces the upstream predicate and winding behavior.  It is a
    host-side replacement for the upstream two-pass Warp compaction, not an
    attempt to expose the raw kernel ABI.
    """
    if triangles.ndim != 2 or triangles.shape[-1] != 3:
        raise ValueError("triangles must have shape (N, 3)")
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (M, 3)")
    if triangles.device != vertices.device:
        raise ValueError("triangles and vertices must be on the same device")
    check_int32_tensors(triangles.device, triangles=triangles)

    if triangles.shape[0] == 0:
        return triangles

    nonnegative = (triangles >= 0).all(dim=-1)
    in_range = (triangles < vertices.shape[0]).all(dim=-1)
    distinct = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    candidates = triangles[nonnegative & in_range & distinct]
    if candidates.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=triangles.device)

    points = vertices[candidates.long()]
    cross = torch.linalg.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area_sq = (cross * cross).sum(dim=-1)
    min_area_sq = (float(voxel_size) * 1.0e-6) ** 2
    result = candidates[area_sq > min_area_sq]
    return result[:, [0, 2, 1]] if flip_winding else result

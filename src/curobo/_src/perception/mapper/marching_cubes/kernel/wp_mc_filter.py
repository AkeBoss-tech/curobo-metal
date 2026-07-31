"""Portable triangle filtering."""

import torch

from curobo._src.perception.mapper._portable import unsupported_kernel

count_valid_triangles_kernel = unsupported_kernel
compact_valid_triangles_kernel = unsupported_kernel


def filter_triangles(triangles, vertices, voxel_size, flip_winding=True):
    tri = vertices[triangles.long()]
    area2 = torch.linalg.vector_norm(
        torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), dim=-1
    )
    result = triangles[area2 > max(float(voxel_size) ** 2 * 1e-8, 0.0)]
    return result[:, [0, 2, 1]] if flip_winding else result

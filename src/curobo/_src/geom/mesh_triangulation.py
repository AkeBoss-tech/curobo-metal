"""Triangle/quad face conversion without requiring Warp."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp, wp
from curobo.logging import log_and_raise


def _normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    value = np.cross(b - a, c - a)
    length = np.linalg.norm(value)
    return value / length if length else value


def triangulate_quads_warp(
    vertices: List[List[float]],
    quads: List[List[int]],
    device_cfg: DeviceCfg,
) -> List[List[List[int]]]:
    """Portable equivalent of the pinned Warp quad-diagonal selection."""
    del device_cfg
    verts = np.asarray(vertices, dtype=np.float32)
    result: List[List[List[int]]] = []
    for q0, q1, q2, q3 in quads:
        n_a1 = _normal(verts[q0], verts[q1], verts[q2])
        n_a2 = _normal(verts[q0], verts[q2], verts[q3])
        n_b1 = _normal(verts[q1], verts[q3], verts[q0])
        n_b2 = _normal(verts[q1], verts[q2], verts[q3])
        if float(np.dot(n_a1, n_a2)) > float(np.dot(n_b1, n_b2)):
            result.append([[q0, q1, q2], [q0, q2, q3]])
        else:
            result.append([[q1, q3, q0], [q1, q2, q3]])
    return result


def triangulate_mesh_faces(
    vertices: List[List[float]],
    faces: List[int],
    face_counts: List[int],
    device_cfg: DeviceCfg = DeviceCfg(),
) -> List[List[int]]:
    if not faces:
        return []
    if sum(face_counts) != len(faces):
        raise ValueError("Face index buffer length does not match face counts")

    output_parts: List[Tuple[str, int]] = []
    triangles: List[List[int]] = []
    quads: List[List[int]] = []
    offset = 0
    for count in face_counts:
        face = list(faces[offset : offset + count])
        offset += count
        if count not in (3, 4):
            raise ValueError("only triangles and quads are supported")
        if any(index < 0 or index >= len(vertices) for index in face):
            raise ValueError("Mesh face index is out of bounds")
        target = triangles if count == 3 else quads
        output_parts.append(("triangle" if count == 3 else "quad", len(target)))
        target.append(face)

    quad_triangles = triangulate_quads_warp(vertices, quads, device_cfg) if quads else []
    result: List[List[int]] = []
    for kind, index in output_parts:
        if kind == "triangle":
            result.append(triangles[index])
        else:
            result.extend(quad_triangles[index])
    return result


def triangulate_quads_kernel(
    verts: wp.array(dtype=wp.vec3),
    quads: wp.array(dtype=wp.vec4i),
    tris_out: wp.array(dtype=wp.vec3i),
):
    """Pinned Warp-kernel declaration; raw Warp dispatch is unavailable here.

    :func:`triangulate_quads_warp` remains the portable NumPy implementation,
    preserving the useful face-conversion behavior without claiming a Warp
    kernel ABI on Metal.
    """
    del verts, quads, tris_out
    raise NotImplementedError("raw Warp triangulation kernels are unavailable")


__all__ = ["triangulate_mesh_faces", "triangulate_quads_kernel", "triangulate_quads_warp"]

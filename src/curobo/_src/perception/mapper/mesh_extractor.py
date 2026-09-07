"""Portable extraction of a mesh from the block-pool TSDF.

The Warp marching-cubes implementation owns the CUDA path.  This module is
the CPU/MPS implementation used by ``PortableSparseTSDF``.  A single sparse
integer-coordinate lattice is used instead of extracting each block in
isolation: this preserves cells which straddle a block boundary and gives us
deterministic edge-based vertex welding.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


_CORNERS = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)
_TETRAHEDRA = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)
_TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _empty(device: torch.device):
    value = torch.empty((0, 3), device=device, dtype=torch.float32)
    return value, value.clone(), value.clone()


def _checked_iterations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("refine_iterations must be a nonnegative integer")
    return value


def _trilinear_refine(
    point: torch.Tensor,
    cell: torch.Tensor,
    cube_values: torch.Tensor,
    world_origin: torch.Tensor,
    voxel_size: float,
    level: float,
    iterations: int,
) -> torch.Tensor:
    """Apply Newton steps to an interpolated vertex in the local cell."""
    if iterations == 0:
        return point
    # Voxel centres are world_origin + (index + .5) * voxel_size.
    local = (point - (world_origin + (cell + 0.5) * voxel_size)) / voxel_size
    local = local.clamp(0.0, 1.0)
    # ``_CORNERS`` follows the marching-cubes convention, whose order is not
    # the row-major order of a [x, y, z] tensor.  Re-index explicitly before
    # evaluating the trilinear polynomial.
    values = torch.empty((2, 2, 2), dtype=cube_values.dtype)
    for index, (x_index, y_index, z_index) in enumerate(_CORNERS):
        values[x_index, y_index, z_index] = cube_values[index]
    for _ in range(iterations):
        x, y, z = local
        wx = torch.stack((1.0 - x, x))
        wy = torch.stack((1.0 - y, y))
        wz = torch.stack((1.0 - z, z))
        value = (values * wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).sum()
        dx = (values[1] * wy[:, None] * wz[None, :] - values[0] * wy[:, None] * wz[None, :]).sum() / voxel_size
        dy = (values[:, 1] * wx[:, None] * wz[None, :] - values[:, 0] * wx[:, None] * wz[None, :]).sum() / voxel_size
        dz = (values[:, :, 1] * wx[:, None] * wy[None, :] - values[:, :, 0] * wx[:, None] * wy[None, :]).sum() / voxel_size
        gradient = torch.stack((dx, dy, dz))
        denominator = torch.dot(gradient, gradient)
        if float(denominator) <= 1.0e-20:
            break
        local = (local - ((value - level) / denominator) * gradient * voxel_size).clamp(0.0, 1.0)
    return world_origin + (cell + 0.5 + local) * voxel_size


def _extract_indexed(
    tsdf,
    level: float,
    surface_only: bool,
    refine_iterations: int,
    minimum_tsdf_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return welded vertices, indexed faces, normals, and colours."""
    # A marching cell emits geometry only when it crosses the requested level;
    # that is precisely the source API's surface_only behaviour.
    del surface_only
    iterations = _checked_iterations(refine_iterations)
    data = getattr(tsdf, "data", None)
    if data is None or not hasattr(data, "num_allocated"):
        raise NotImplementedError("indexed portable extraction requires a block-sparse TSDF")
    high_water = int(data.num_allocated.item())
    device = data.block_data.device
    empty_faces = torch.empty((0, 3), device=device, dtype=torch.int32)
    if high_water == 0:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        return empty, empty_faces, empty.clone(), empty.clone()
    block_size = int(data.block_size)
    if block_size < 2:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        return empty, empty_faces, empty.clone(), empty.clone()

    # Host-side arithmetic is intentional: MPS has incomplete support for
    # several advanced indexing/linalg operations and this keeps CPU/MPS
    # output identical.  Pool order is not stable, so sort by block coords.
    # ``num_allocated`` is a high-water mark in the pool.  Recycled slots can
    # therefore occur below it; use the source-visible hash-slot marker when
    # it carries active entries, while retaining compatibility with minimal
    # test doubles that do not provide that marker.
    pool_indices = torch.arange(high_water, dtype=torch.long)
    slots = getattr(data, "block_to_hash_slot", None)
    if slots is not None:
        slots_cpu = slots[:high_water].detach().cpu()
        active_slots = torch.nonzero(slots_cpu >= 0, as_tuple=False).flatten()
        if active_slots.numel():
            pool_indices = active_slots
    count = int(pool_indices.numel())
    coords = data.block_coords.view(-1, 3)[:high_water].detach().float().cpu()[pool_indices]
    sums = data.block_data[:high_water, :, 0].detach().float().cpu()[pool_indices]
    weights = data.block_data[:high_water, :, 1].detach().float().cpu()[pool_indices]
    block_valid = weights >= float(minimum_tsdf_weight)
    fields = sums / weights.clamp_min(1.0e-12)
    rgbw = data.block_grid_rgb[:high_water, 0].detach().float().cpu()[pool_indices]
    block_colours = rgbw[:, :3] / rgbw[:, 3:4].clamp_min(1.0e-6)

    order = sorted(range(count), key=lambda index: tuple(int(v) for v in coords[index].tolist()))
    field_at: Dict[Tuple[int, int, int], float] = {}
    valid_at: Dict[Tuple[int, int, int], bool] = {}
    colour_at: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for block_index in order:
        base = tuple(int(v) * block_size for v in coords[block_index].tolist())
        for linear in range(block_size ** 3):
            ix = linear // (block_size * block_size)
            iy = (linear // block_size) % block_size
            iz = linear % block_size
            key = (base[0] + ix, base[1] + iy, base[2] + iz)
            # Invalid pools should not overwrite a valid duplicate.  Valid
            # duplicate coordinates are malformed state; first sorted block is
            # deterministic and mirrors the hash table's first-owner policy.
            if key in field_at:
                continue
            field_at[key] = float(fields[block_index, linear])
            valid_at[key] = bool(block_valid[block_index, linear])
            colour_at[key] = block_colours[block_index]

    cells: set[Tuple[int, int, int]] = set()
    for block_index in order:
        base = tuple(int(v) * block_size for v in coords[block_index].tolist())
        # Include the final local index.  Those cells straddle into the next
        # block; they are retained only when all eight lattice corners are
        # present below, so an isolated block still emits no out-of-map cell.
        for ix in range(block_size):
            for iy in range(block_size):
                for iz in range(block_size):
                    cells.add((base[0] + ix, base[1] + iy, base[2] + iz))

    world_origin = data.origin.detach().float().cpu()
    voxel_size = float(data.voxel_size)
    vertex_by_edge: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    vertices: List[torch.Tensor] = []
    normal_sums: List[torch.Tensor] = []
    colour_sums: List[torch.Tensor] = []
    colour_counts: List[int] = []
    faces: List[Tuple[int, int, int]] = []

    for cell in sorted(cells):
        corner_keys = tuple(
            (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            for offset in _CORNERS
        )
        if not all(key in field_at and valid_at[key] for key in corner_keys):
            continue
        cube_values = torch.tensor([field_at[key] for key in corner_keys], dtype=torch.float32)
        below = cube_values < float(level)
        if bool(below.all()) or not bool(below.any()):
            continue
        cube_positions = torch.stack(
            [world_origin + (torch.tensor(key, dtype=torch.float32) + 0.5) * voxel_size for key in corner_keys]
        )
        matrix = cube_positions[1:] - cube_positions[0]
        rhs = cube_values[1:] - cube_values[0]
        gradient = torch.linalg.lstsq(matrix, rhs[:, None]).solution[:, 0]
        gradient = gradient / torch.linalg.vector_norm(gradient).clamp_min(1.0e-12)
        cube_colour = torch.stack([colour_at[key] for key in corner_keys]).mean(0)
        for tetrahedron in _TETRAHEDRA:
            ids = torch.tensor(tetrahedron, dtype=torch.long)
            tetra_below = below[ids]
            if bool(tetra_below.all()) or not bool(tetra_below.any()):
                continue
            intersections: List[Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], torch.Tensor]] = []
            for start, end in _TETRA_EDGES:
                if bool(tetra_below[start]) == bool(tetra_below[end]):
                    continue
                start_id, end_id = tetrahedron[start], tetrahedron[end]
                start_key, end_key = corner_keys[start_id], corner_keys[end_id]
                edge_key = tuple(sorted((start_key, end_key)))
                denominator = cube_values[end_id] - cube_values[start_id]
                fraction = (float(level) - cube_values[start_id]) / denominator
                point = cube_positions[start_id] + fraction * (cube_positions[end_id] - cube_positions[start_id])
                point = _trilinear_refine(
                    point, torch.tensor(cell, dtype=torch.float32), cube_values,
                    world_origin, voxel_size, float(level), iterations,
                )
                intersections.append((edge_key, point))
            if len(intersections) not in (3, 4):
                continue
            local_faces = ((0, 1, 2),) if len(intersections) == 3 else ((0, 1, 2), (0, 2, 3))
            for local_face in local_faces:
                ids_for_face: List[int] = []
                for local_index in local_face:
                    edge_key, point = intersections[local_index]
                    vertex_index = vertex_by_edge.get(edge_key)
                    if vertex_index is None:
                        vertex_index = len(vertices)
                        vertex_by_edge[edge_key] = vertex_index
                        vertices.append(point)
                        normal_sums.append(torch.zeros(3))
                        colour_sums.append(torch.zeros(3))
                        colour_counts.append(0)
                    ids_for_face.append(vertex_index)
                triangle = torch.stack([vertices[index] for index in ids_for_face])
                face_normal = torch.linalg.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                if torch.dot(face_normal, gradient) < 0:
                    ids_for_face[1], ids_for_face[2] = ids_for_face[2], ids_for_face[1]
                faces.append(tuple(ids_for_face))
                for index in ids_for_face:
                    normal_sums[index] += gradient
                    colour_sums[index] += cube_colour
                    colour_counts[index] += 1

    if not vertices:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        return empty, empty_faces, empty.clone(), empty.clone()
    result_vertices = torch.stack(vertices).to(device=device)
    result_normals = torch.stack([
        normal_sums[index] / torch.linalg.vector_norm(normal_sums[index]).clamp_min(1.0e-12)
        for index in range(len(vertices))
    ]).to(device=device)
    result_colours = torch.stack([
        colour_sums[index] / max(colour_counts[index], 1) for index in range(len(vertices))
    ]).to(device=device)
    result_faces = torch.tensor(faces, dtype=torch.int32, device=device)
    return result_vertices, result_faces, result_normals, result_colours


def extract_mesh_block_sparse(
    tsdf,
    level: float = 0.0,
    surface_only: bool = False,
    refine_iterations: int = 0,
    minimum_tsdf_weight: float = 0.1,
    *,
    return_faces: bool = False,
):
    """Extract a portable mesh, preserving the pinned triangle-soup default.

    ``return_faces=True`` exposes the welded indexed representation for the
    integrator's ``Mesh`` path.  The historical three-tensor return remains a
    triangle soup, as required by the upstream helper tests.
    """
    if hasattr(tsdf, "extract_mesh") and not getattr(tsdf, "_portable_sparse", False):
        mesh = tsdf.extract_mesh()
        vertices = torch.as_tensor(mesh.vertices)
        normals = torch.zeros_like(vertices)
        colours = torch.zeros_like(vertices)
        if return_faces:
            return vertices, torch.as_tensor(mesh.faces, device=vertices.device, dtype=torch.int32), normals, colours
        return vertices, normals, colours
    vertices, faces, normals, colours = _extract_indexed(
        tsdf, level, surface_only, refine_iterations, minimum_tsdf_weight,
    )
    if return_faces:
        return vertices, faces, normals, colours
    if faces.numel() == 0:
        return _empty(vertices.device)
    soup_indices = faces.reshape(-1).to(torch.long)
    return vertices[soup_indices], normals[soup_indices], colours[soup_indices]


__all__ = ["extract_mesh_block_sparse"]

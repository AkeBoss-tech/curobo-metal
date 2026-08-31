"""Deterministic, portable dispatcher for mesh-to-sphere fitting.

The pinned implementation chooses between surface sampling, a Warp signed-SDF
voxel fit, and MorphIt.  The Metal port keeps that public lifecycle while
using the vectorised PyTorch mesh query that backs :mod:`fit_voxel`.  Meshes
without triangle topology (for example the primitive vertex clouds used by
the attachment manager) still have a useful deterministic surface fit rather
than accidentally taking a CUDA/Warp-only path.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import numpy
import torch

from curobo._src.types.device_cfg import DeviceCfg

from .fit_morphit import MorphItLossWeights, morphit_sphere_fit
from .fit_voxel import sample_even_fit_mesh, voxel_fit_mesh
from .metrics import populate_metrics
from .sphere_count import _vertices, estimate_sphere_count
from .types import SphereFitMetrics, SphereFitResult, SphereFitType
from . import _trimesh_compat as trimesh
from curobo._src.util.logging import log_info, log_warn


class _TopologyMesh:
    """Small CPU topology view used only to construct a deterministic grid.

    The actual SDF evaluation remains on ``device_cfg.device`` inside
    ``voxel_fit_mesh``.  A CPU view is necessary because the portable grid
    builder deliberately uses NumPy, and it avoids calling ``.numpy()`` on an
    MPS tensor directly.
    """

    def __init__(self, mesh: Any, vertices: torch.Tensor) -> None:
        self.vertices = vertices.detach().cpu().numpy()
        self.faces = torch.as_tensor(mesh.faces, dtype=torch.int64).detach().cpu().numpy()
        self.bounds = np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))
        self.is_watertight = bool(getattr(mesh, "is_watertight", False))
        self.volume = float(getattr(mesh, "volume", 0.0))


def _coerce_fit_type(value: SphereFitType | str) -> SphereFitType:
    try:
        return value if isinstance(value, SphereFitType) else SphereFitType(value)
    except (TypeError, ValueError) as error:
        options = ", ".join(item.value for item in SphereFitType)
        raise ValueError(f"fit_type must be one of: {options}") from error


def _topology_mesh(mesh: Any, vertices: torch.Tensor) -> Optional[_TopologyMesh]:
    """Return a validated triangle view, or ``None`` for a vertex cloud."""

    faces = getattr(mesh, "faces", None)
    if faces is None:
        return None
    faces = torch.as_tensor(faces)
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.numel() == 0:
        return None
    if faces.dtype.is_floating_point:
        if not bool(torch.equal(faces, faces.round())):
            raise ValueError("mesh faces must contain integer vertex indices")
    faces = faces.to(dtype=torch.int64)
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("mesh faces contain an out-of-range vertex index")
    # Keep the source object's visible attributes intact (notably a trimesh
    # convex hull) but use a simple view for the portable SDF backend.
    return _TopologyMesh(mesh, vertices)


def _farthest_surface_fit(
    vertices: torch.Tensor,
    count: int,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input-order-stable farthest-point surface seeds for vertex-only meshes."""

    # ``argmin`` resolves a coordinate tie by the first input index, which is
    # deterministic and preserves callers' serialized mesh ordering.
    chosen = [int(torch.argmin(vertices[:, 0]).item())]
    nearest = torch.full((len(vertices),), torch.inf, dtype=vertices.dtype, device=vertices.device)
    while len(chosen) < min(count, len(vertices)):
        point = vertices[chosen[-1]].unsqueeze(0)
        nearest = torch.minimum(nearest, torch.linalg.vector_norm(vertices - point, dim=-1))
        chosen.append(int(torch.argmax(nearest).item()))
    index = torch.tensor(chosen, dtype=torch.long, device=vertices.device)
    centers = vertices.index_select(0, index)
    if len(centers) < count:
        centers = torch.cat((centers, centers[-1:].expand(count - len(centers), -1)), dim=0)
    radii = torch.full((count,), float(radius), dtype=vertices.dtype, device=vertices.device)
    return centers, radii


def _clip_result(result: SphereFitResult, clip_plane: tuple, buffer: float = 0.02) -> None:
    """Apply the pinned half-plane policy directly on the output device."""

    if len(clip_plane) != 2:
        raise ValueError("clip_plane must be ((nx, ny, nz), offset)")
    normal, offset = clip_plane
    normal_t = torch.as_tensor(normal, dtype=result.centers.dtype, device=result.centers.device).reshape(-1)
    if normal_t.numel() != 3:
        raise ValueError("clip_plane normal must contain three values")
    norm = torch.linalg.vector_norm(normal_t)
    if not bool(torch.isfinite(norm)) or float(norm) <= torch.finfo(result.centers.dtype).eps:
        raise ValueError("clip_plane normal must be finite and nonzero")
    offset = float(offset)
    if not np.isfinite(offset):
        raise ValueError("clip_plane offset must be finite")
    normal_t = normal_t / norm
    signed = result.centers @ normal_t - offset
    keep = signed > float(buffer)
    result.centers = result.centers[keep]
    result.radii = result.radii[keep]
    signed = signed[keep]
    if len(signed):
        result.radii = torch.minimum(result.radii, (signed - float(buffer)).clamp_min(1e-4))
    result.num_spheres = int(result.centers.shape[0])


def _to_result_tensor(value: Any, device_cfg: DeviceCfg) -> torch.Tensor:
    return torch.as_tensor(value, dtype=device_cfg.dtype, device=device_cfg.device).reshape(-1)


def _query_device(device: torch.device) -> torch.device:
    """Normalise Metal's implicit device index for strict query-device checks."""

    if device.type == "mps" and device.index is None:
        return torch.device("mps", 0)
    return device


def _vertex_surface_metrics(vertices: torch.Tensor, result: SphereFitResult) -> SphereFitMetrics:
    """Deterministic surface-only metric proxy for triangle-free callers."""

    if result.num_spheres == 0:
        return SphereFitMetrics(num_spheres=0, surface_gap_mean=float("inf"), surface_gap_p95=float("inf"), max_uncovered_gap=float("inf"))
    gaps = (torch.cdist(vertices, result.centers) - result.radii.unsqueeze(0)).amin(dim=-1).clamp_min(0)
    return SphereFitMetrics(
        num_spheres=result.num_spheres,
        coverage=float((gaps <= 1e-6).float().mean()),
        surface_gap_mean=float(gaps.mean()),
        surface_gap_p95=float(torch.quantile(gaps, 0.95)),
        max_uncovered_gap=float(gaps.max()),
    )


def fit_spheres_to_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: Optional[int] = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    fit_type: SphereFitType = SphereFitType.MORPHIT,
    iterations: int = 200,
    compute_metrics: bool = False,
    coverage_weight: Optional[float] = None,
    protrusion_weight: Optional[float] = None,
    clip_plane: Optional[tuple] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> SphereFitResult:
    """Approximate a triangle mesh or vertex cloud with deterministic spheres.

    ``SURFACE`` performs deterministic farthest-point sampling, ``VOXEL``
    uses the differentiable portable signed-SDF query for triangle meshes,
    and ``MORPHIT`` retains V2's voxel-seeded optimisation lifecycle.  The
    latter is a portable approximation, not a claim of Warp/MorphIt numerical
    equivalence.  When no triangle faces are available, all modes use the
    deterministic surface fallback and identify that boundary in
    ``debug_info``.
    """

    if sphere_density <= 0 or not np.isfinite(sphere_density):
        raise ValueError("sphere_density must be finite and positive")
    if surface_radius < 0 or not np.isfinite(surface_radius):
        raise ValueError("surface_radius must be finite and nonnegative")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    fit_type = _coerce_fit_type(fit_type)
    vertices = device_cfg.to_device(_vertices(mesh)).contiguous()
    if not bool(torch.isfinite(vertices).all()):
        raise ValueError("mesh vertices must be finite")
    requested_count = None if num_spheres is None else int(num_spheres)
    if requested_count is not None and requested_count < 1:
        raise ValueError("num_spheres must be positive")
    count = estimate_sphere_count(mesh, sphere_density) if requested_count is None else requested_count
    topology = _topology_mesh(mesh, vertices)
    query_device = _query_device(device_cfg.device)
    start = time.perf_counter()
    history: list[tuple[Any, Any]] = []
    fallback_used = False
    backend = "surface_farthest_point"

    centers: Optional[torch.Tensor] = None
    radii: Optional[torch.Tensor] = None
    if fit_type is SphereFitType.SURFACE or topology is None:
        centers, radii = _farthest_surface_fit(vertices, count, surface_radius)
        fallback_used = fit_type is not SphereFitType.SURFACE
    elif fit_type is SphereFitType.VOXEL:
        voxel_centers, voxel_radii = voxel_fit_mesh(topology, count, device=query_device)
        if voxel_centers is not None and voxel_radii is not None and len(voxel_centers):
            centers = device_cfg.to_device(voxel_centers).reshape(-1, 3)
            radii = _to_result_tensor(voxel_radii, device_cfg)
            backend = "portable_signed_sdf_voxel"
    else:
        voxel_centers, voxel_radii = voxel_fit_mesh(topology, count, device=query_device)
        if voxel_centers is not None and voxel_radii is not None and len(voxel_centers):
            weights = None
            if coverage_weight is not None or protrusion_weight is not None:
                weights = MorphItLossWeights(
                    coverage=1000.0 if coverage_weight is None else float(coverage_weight),
                    protrusion=10.0 if protrusion_weight is None else float(protrusion_weight),
                )
            fitted_centers, fitted_radii, raw_history = morphit_sphere_fit(
                topology,
                count,
                iterations=iterations,
                init_centers=voxel_centers,
                init_radii=voxel_radii,
                loss_weights=weights,
                # The common output clipping below gives all modes identical
                # postconditions, including the portable fallback.
                clip_plane=None,
                max_spheres=count,
                device=query_device,
            )
            if fitted_centers is not None and fitted_radii is not None and len(fitted_centers):
                centers = device_cfg.to_device(fitted_centers).reshape(-1, 3)
                radii = _to_result_tensor(fitted_radii, device_cfg)
                history = [
                    (device_cfg.to_device(item_centers).reshape(-1, 3), _to_result_tensor(item_radii, device_cfg))
                    for item_centers, item_radii in raw_history
                ]
                backend = "portable_morphit_voxel_seed"

    if centers is None or radii is None or len(centers) == 0:
        centers, radii = _farthest_surface_fit(vertices, count, surface_radius)
        fallback_used = True
        backend = "surface_farthest_point_fallback"

    # V2 keeps the largest spheres when a backend returns more than requested.
    if len(centers) > count:
        order = torch.argsort(radii, descending=True, stable=True)[:count]
        centers, radii = centers.index_select(0, order), radii.index_select(0, order)
    result = SphereFitResult(
        centers=centers.contiguous(),
        radii=radii.contiguous(),
        num_spheres=int(centers.shape[0]),
        fit_time_s=time.perf_counter() - start,
        used_mesh=mesh,
        history=history,
        debug_info={
            "fit_type": fit_type.value,
            "portable": True,
            "backend": backend,
            "has_triangle_topology": topology is not None,
            "auto_n_spheres": requested_count is None,
            "requested_n_spheres": requested_count,
            "resolved_n_spheres": count,
            "fallback_used": fallback_used,
            "morphit_equivalence": "not_claimed" if fit_type is SphereFitType.MORPHIT else "not_applicable",
        },
    )
    if clip_plane is not None and result.num_spheres:
        _clip_result(result, clip_plane)
    if compute_metrics:
        if topology is None:
            # The historical public helper accepts primitive vertex clouds.
            # Preserve its useful count/gap output, while making clear that
            # signed-volume coverage and protrusion are unavailable.
            result.metrics = _vertex_surface_metrics(vertices, result)
            result.debug_info["metrics_boundary"] = "vertex-surface proxy; triangle topology required for signed metrics"
        else:
            populate_metrics(result, topology, device=query_device)
    return result


__all__ = ["fit_spheres_to_mesh"]

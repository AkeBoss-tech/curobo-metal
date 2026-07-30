import time
import torch

from curobo._src.types.device_cfg import DeviceCfg
from .sphere_count import _vertices, estimate_sphere_count
from .types import SphereFitMetrics, SphereFitResult, SphereFitType


def fit_spheres_to_mesh(
    mesh, num_spheres=None, sphere_density=1.0, surface_radius=0.005,
    fit_type=SphereFitType.MORPHIT, iterations=200, compute_metrics=False,
    coverage_weight=None, protrusion_weight=None, clip_plane=None,
    device_cfg=DeviceCfg(),
):
    start = time.perf_counter()
    vertices = device_cfg.to_device(_vertices(mesh))
    count = estimate_sphere_count(mesh, sphere_density) if num_spheres is None else int(num_spheres)
    if count < 1 or surface_radius <= 0:
        raise ValueError("num_spheres and surface_radius must be positive")
    # Deterministic farthest-point seeds give a useful portable approximation.
    chosen = [int(torch.argmin(vertices[:, 0]).item())]
    nearest = torch.full((len(vertices),), torch.inf, device=vertices.device)
    for _ in range(1, min(count, len(vertices))):
        nearest = torch.minimum(nearest, torch.linalg.vector_norm(vertices-vertices[chosen[-1]], dim=-1))
        chosen.append(int(torch.argmax(nearest).item()))
    centers = vertices[torch.tensor(chosen, device=vertices.device)]
    if len(centers) < count:
        centers = torch.cat((centers, centers[-1:].expand(count-len(centers), -1)))
    distance = torch.cdist(vertices, centers)
    assignment = distance.argmin(-1)
    radii = torch.full((count,), float(surface_radius), device=vertices.device)
    radii.scatter_reduce_(0, assignment, distance.min(-1).values, reduce="amax", include_self=True)
    metrics = None
    if compute_metrics:
        gaps = (distance-radii[None]).clamp_min(0).min(-1).values
        metrics = SphereFitMetrics(
            num_spheres=count,
            coverage=float((gaps <= 1e-6).float().mean()),
            surface_gap_mean=float(gaps.mean()),
            surface_gap_p95=float(torch.quantile(gaps, 0.95)),
            max_uncovered_gap=float(gaps.max()),
        )
    return SphereFitResult(
        centers, radii, count, metrics, time.perf_counter()-start, mesh,
        debug_info={"fit_type": SphereFitType(fit_type).value, "portable": True},
    )

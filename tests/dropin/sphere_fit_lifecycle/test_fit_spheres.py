from types import SimpleNamespace

import numpy as np
import pytest
import torch

from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.types.device_cfg import DeviceCfg


def _cube_mesh(device: str = "cpu"):
    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ], device=device,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ], device=device,
    )
    return SimpleNamespace(
        vertices=vertices,
        faces=faces,
        bounds=np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
        is_watertight=True,
        volume=1.0,
    )


@pytest.mark.parametrize("fit_type", list(SphereFitType))
def test_fit_dispatch_is_deterministic_and_reports_portable_boundary(fit_type):
    mesh = _cube_mesh()
    kwargs = dict(num_spheres=4, fit_type=fit_type, surface_radius=0.03, device_cfg=DeviceCfg())
    first = fit_spheres_to_mesh(mesh, **kwargs)
    second = fit_spheres_to_mesh(mesh, **kwargs)
    torch.testing.assert_close(first.centers, second.centers)
    torch.testing.assert_close(first.radii, second.radii)
    assert first.centers.shape == (4, 3)
    assert first.radii.shape == (4,)
    assert first.debug_info["fit_type"] == fit_type.value
    assert first.debug_info["has_triangle_topology"]
    if fit_type is SphereFitType.SURFACE:
        torch.testing.assert_close(first.radii, torch.full((4,), 0.03))
    if fit_type is SphereFitType.MORPHIT:
        assert first.debug_info["morphit_equivalence"] == "not_claimed"
        assert len(first.history) == 1


def test_vertex_cloud_fallback_clip_plane_and_metric_boundary():
    mesh = SimpleNamespace(vertices=_cube_mesh().vertices)
    result = fit_spheres_to_mesh(
        mesh,
        num_spheres=5,
        fit_type="voxel",
        surface_radius=0.1,
        clip_plane=((1.0, 0.0, 0.0), 0.0),
        compute_metrics=True,
    )
    assert result.debug_info["fallback_used"]
    assert result.debug_info["metrics_boundary"].startswith("vertex-surface proxy")
    assert result.metrics is not None and result.metrics.num_spheres == result.num_spheres
    assert torch.all(result.centers[:, 0] > 0.02)
    assert torch.all(result.radii <= result.centers[:, 0] - 0.02 + 1e-6)


def test_auto_count_and_invalid_arguments():
    result = fit_spheres_to_mesh(_cube_mesh(), sphere_density=0.5, fit_type="surface")
    assert result.debug_info["auto_n_spheres"] and result.num_spheres >= 1
    with pytest.raises(ValueError, match="fit_type"):
        fit_spheres_to_mesh(_cube_mesh(), fit_type="warp")
    with pytest.raises(ValueError, match="num_spheres"):
        fit_spheres_to_mesh(_cube_mesh(), num_spheres=0)
    with pytest.raises(ValueError, match="clip_plane normal"):
        fit_spheres_to_mesh(_cube_mesh(), num_spheres=1, clip_plane=((0, 0, 0), 0))


def test_mps_fit_stays_on_mps_without_fallback(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = fit_spheres_to_mesh(
        _cube_mesh("mps"),
        num_spheres=3,
        fit_type=SphereFitType.VOXEL,
        compute_metrics=True,
        device_cfg=DeviceCfg(device="mps"),
    )
    assert result.centers.device.type == "mps"
    assert result.radii.device.type == "mps"
    assert result.num_spheres >= 1
    assert result.metrics is not None

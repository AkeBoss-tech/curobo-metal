"""Production-query lifecycle coverage for the portable MeshData cache."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.data.data_mesh import MeshData
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


def _tetra(name: str = "tetra", *, pose=None) -> Mesh:
    return Mesh(
        name=name,
        pose=pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    )


def test_shared_geometry_pose_lifecycle_and_scene_interop() -> None:
    cfg = DeviceCfg()
    left, right = _tetra(pose=[0, 0, 0, 1, 0, 0, 0]), _tetra(pose=[2, 0, 0, 1, 0, 0, 0])
    data = MeshData.from_batch_scene_cfg([SceneCfg(mesh=[left]), SceneCfg(mesh=[right])], cfg)

    assert data.get_cached_mesh_names() == ["tetra"]
    assert data.mesh_ids[:, 0].tolist() == [1, 1]
    torch.testing.assert_close(data.dims[0, 0, :3], torch.ones(3))
    torch.testing.assert_close(data.get_world_pose("tetra", 1), torch.tensor([2.0, 0, 0, 1, 0, 0, 0]))

    data.update_pose("tetra", w_obj_pose=[3, 0, 0, 1, 0, 0, 0], env_idx=1)
    reconstructed = data.get_meshes(1)
    assert len(reconstructed) == 1 and reconstructed[0].name == "tetra"
    assert reconstructed[0].pose[:3] == [3.0, 0.0, 0.0]
    assert len(data.as_scene_cfg(1).mesh) == 1


def test_query_points_uses_current_environment_enable_and_pose() -> None:
    data = MeshData.from_batch_scene_cfg(
        [SceneCfg(mesh=[_tetra(pose=[0, 0, 0, 1, 0, 0, 0])]), SceneCfg(mesh=[_tetra(pose=[2, 0, 0, 1, 0, 0, 0])])],
        DeviceCfg(),
    )
    points = torch.tensor([[[1.5, 0.1, 0.1]], [[1.5, 0.1, 0.1]]], requires_grad=True)
    result = data.query_points(points, env_indices=torch.tensor([0, 1]))
    assert result.reduced_distance.shape == (2, 1)
    assert result.reduced_distance[1, 0] < result.reduced_distance[0, 0]
    result.reduced_distance.sum().backward()
    assert torch.isfinite(points.grad).all()

    data.set_enabled("tetra", False, env_idx=1)
    disabled = data.query_points(points.detach(), env_indices=torch.tensor([1, 1]))
    assert torch.isinf(disabled.reduced_distance).all()
    assert (disabled.reduced_gradient == 0).all()


def test_cache_rejects_ambiguous_geometry_replacement_and_exposes_warp_boundary() -> None:
    data = MeshData.from_scene_cfg(SceneCfg(mesh=[_tetra()]), DeviceCfg())
    changed = _tetra()
    changed.vertices = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="different geometry"):
        data.update_mesh(changed)
    with pytest.raises(NotImplementedError, match="raw Warp mesh IDs"):
        data.update_from_warp_id(7, "external", w_obj_pose=[0, 0, 0, 1, 0, 0, 0])

    data.clear(clear_warp_cache=True)
    assert data.get_cached_mesh_names() == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
def test_mps_query_retains_device_and_backpropagates() -> None:
    cfg = DeviceCfg(device="mps")
    data = MeshData.from_scene_cfg(SceneCfg(mesh=[_tetra()]), cfg)
    point = torch.tensor([[1.2, 0.2, 0.2]], device="mps", requires_grad=True)
    result = data.query_points(point)
    assert result.reduced_distance.device.type == "mps"
    result.reduced_distance.sum().backward()
    assert point.grad is not None and point.grad.device.type == "mps"

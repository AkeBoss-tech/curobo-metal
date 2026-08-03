"""Lifecycle coverage for portable SceneData cuboid and ESDF caches."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.data.data_cuboid import CuboidData
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.data.data_voxel import (
    VoxelData,
    sample_voxel_sdf,
    sample_voxel_sdf_with_grad,
    voxel_idx_to_flat,
)
from curobo._src.geom.types import Cuboid, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _grid(name: str, *, dims=(2.0, 2.0, 2.0), offset: float = 0.0) -> VoxelGrid:
    shape = tuple(round(value) for value in dims)
    return VoxelGrid(
        name=name,
        pose=[offset, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dims=list(dims),
        voxel_size=1.0,
        feature_tensor=torch.arange(float(shape[0] * shape[1] * shape[2])).reshape(shape),
    )


def test_cuboid_cache_is_atomic_and_rejects_duplicate_names() -> None:
    cfg = DeviceCfg()
    cache = CuboidData.create_cache(2, 1, cfg)
    first = Cuboid("first", [0, 0, 0, 1, 0, 0, 0], dims=[1, 2, 3])
    cache.load_batch([first], 0)
    with pytest.raises(ValueError, match="unique"):
        cache.load_batch([first, first], 0)
    assert cache.get_names() == ["first"]
    with pytest.raises(ValueError, match="already exists"):
        cache.add(first)
    with pytest.raises(ValueError, match="positive"):
        cache.update_dims("first", [1, 0, 1])
    torch.testing.assert_close(cache.dims[0, 0, :3], torch.tensor([1.0, 2.0, 3.0]))


def test_scene_rejects_cross_layer_duplicates_and_tracks_environments() -> None:
    cfg = DeviceCfg()
    duplicate = SceneCfg(
        cuboid=[Cuboid("same", [0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])],
        voxel=[_grid("same")],
    )
    with pytest.raises(ValueError, match="unique"):
        SceneData.from_scene_cfg(duplicate, cfg)

    scene = SceneData.create_cache(2, cfg, cuboid_cache=1, voxel_cache={"layers": 1, "dims": [2, 2, 2], "voxel_size": 1})
    scene.add_obstacle(Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]), env_idx=1)
    scene.add_obstacle(_grid("grid", offset=2.0), env_idx=1)
    assert scene.get_obstacle_names(1) == ["box", "grid"]
    scene.update_obstacle_pose("box", Pose.from_list([4, 0, 0, 1, 0, 0, 0], cfg), 1)
    torch.testing.assert_close(scene.cuboids.inv_pose[1, 0, :3], torch.tensor([-4.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="already exists"):
        scene.add_obstacle(_grid("grid"), env_idx=1)


def test_voxel_metadata_updates_are_exact_and_reconstruct_world_pose() -> None:
    cfg = DeviceCfg()
    source = _grid("grid", dims=(2.0, 3.0, 2.0), offset=3.0)
    data = VoxelData.from_voxel_grid(source, cfg)
    assert data.get_grid_shape(name="grid") == torch.Size([2, 3, 2])
    torch.testing.assert_close(data.params[0, 0], torch.tensor([2.0, 3.0, 2.0, 1.0]))
    data.update_features(torch.ones(12), "grid")
    restored = data.get_voxel_grid("grid")
    assert restored.get_grid_shape()[0] == [2, 3, 2]
    torch.testing.assert_close(torch.tensor(restored.pose[:3]), torch.tensor([3.0, 0.0, 0.0]))
    torch.testing.assert_close(restored.feature_tensor, torch.ones(12))
    with pytest.raises(ValueError, match="12 finite"):
        data.update_features(torch.ones(11), "grid")
    replacement = _grid("replacement", dims=(2.0, 3.0, 2.0), offset=-2.0)
    data.update_data(replacement, name="grid")
    assert data.get_names() == ["grid"]
    torch.testing.assert_close(torch.tensor(data.get_voxel_grid("grid").pose[:3]), torch.tensor([-2.0, 0.0, 0.0]))


def test_voxel_helpers_cover_offsets_boundaries_and_gradients() -> None:
    values = torch.arange(8.0)
    indices = torch.tensor([[0, 0, 0], [1, 1, 1], [-1, 0, 0]])
    torch.testing.assert_close(voxel_idx_to_flat(indices[:2], [2, 2, 2]), torch.tensor([0, 7]))
    sampled = sample_voxel_sdf(values, 0, indices, [2, 2, 2], 99.0)
    torch.testing.assert_close(sampled, torch.tensor([[0.0, 1.0], [7.0, 1.0], [99.0, 0.0]]))
    query = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    sdf_grad = sample_voxel_sdf_with_grad(values, 0, query, [2, 2, 2], 1.0, 99.0)
    torch.testing.assert_close(sdf_grad[0, :1], torch.tensor([3.5]))
    sdf_grad[..., 0].sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_scene_data_and_voxel_sampling_stay_on_mps_without_fallback() -> None:
    cfg = DeviceCfg(device=torch.device("mps"))
    grid = _grid("grid")
    grid.feature_tensor = grid.feature_tensor.to("mps")
    scene = SceneData.from_scene_cfg(SceneCfg(voxel=[grid]), cfg)
    assert scene.voxels.features.device.type == "mps"
    point = torch.zeros((1, 3), device="mps", requires_grad=True)
    out = sample_voxel_sdf_with_grad(scene.voxels.features[0, 0, :8], 0, point, [2, 2, 2], 1.0, 99.0)
    assert out.device.type == "mps"
    out[..., 0].sum().backward()
    assert point.grad is not None and point.grad.device.type == "mps"

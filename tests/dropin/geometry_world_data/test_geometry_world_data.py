"""Executable portable lifecycle coverage for V2 geometry data records."""

from __future__ import annotations

import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Capsule, Cuboid, Mesh, SceneCfg, Sphere, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _voxel(name: str) -> VoxelGrid:
    return VoxelGrid(
        name=name,
        pose=[0, 0, 0, 1, 0, 0, 0],
        dims=[2, 2, 2],
        voxel_size=1.0,
        feature_tensor=torch.arange(8.0).reshape(2, 2, 2),
    )


def test_scene_data_routes_per_environment_lifecycle() -> None:
    left = SceneCfg(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])], voxel=[_voxel("left")])
    right = SceneCfg(cuboid=[Cuboid("far", [2, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])], voxel=[_voxel("right")])
    data = SceneData.from_batch_scene_cfg([left, right], DeviceCfg())
    assert data.get_obstacle_names(0) == ["box", "left"]
    assert data.get_obstacle_names(1) == ["far", "right"]
    data.enable_obstacle("box", False)
    assert data.cuboids.enable[0, 0].item() == 0
    data.update_obstacle_pose("far", Pose.from_list([3, 0, 0, 1, 0, 0, 0]), env_idx=1)
    torch.testing.assert_close(data.cuboids.inv_pose[1, 0, :3], torch.tensor([-3.0, 0.0, 0.0]))
    data.clear(1)
    assert data.get_obstacle_names(1) == []
    assert data.get_obstacle_names(0) == ["box", "left"]


def test_voxel_grid_storage_validates_updates_and_reconstructs_metadata() -> None:
    grid = _voxel("grid")
    data = SceneData.from_scene_cfg(SceneCfg(voxel=[grid]), DeviceCfg())
    assert data.voxels is not None
    assert data.voxels.get_grid_shape(name="grid") == torch.Size([2, 2, 2])
    data.voxels.update_features(torch.ones(8), "grid")
    torch.testing.assert_close(data.voxels.features[0, 0, :8], torch.ones(8))
    xyzr = grid.create_xyzr_tensor()
    assert xyzr.shape == (8, 4)
    grid.xyzr_tensor = xyzr
    assert grid.get_occupied_voxels(feature_threshold=3.5).shape == (4, 4)


def test_scene_conversion_preserves_pose_and_exposes_collision_layers() -> None:
    mesh = Mesh("tri", pose=[1, 0, 0, 1, 0, 0, 0], vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    world = SceneCfg(
        cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[1, 2, 3])],
        sphere=[Sphere("ball", pose=[2, 0, 0, 1, 0, 0, 0], radius=.5)],
        capsule=[Capsule("cap", pose=[1, 0, 0, 1, 0, 0, 0], base=[0, 0, 0], tip=[0, 0, 2], radius=.1)],
        mesh=[mesh],
    )
    obb = world.get_obb_world()
    assert len(obb.cuboid) == 4
    triangle_obb = obb.get_obstacle("tri")
    assert triangle_obb is not None
    torch.testing.assert_close(torch.tensor(triangle_obb.pose[:3]), torch.tensor([1.5, .5, 0.0]))
    collision = world.get_collision_check_world()
    assert [entry.name for entry in collision.cuboid] == ["box"]
    assert {entry.name for entry in collision.mesh} == {"tri", "ball", "cap"}
    merged = world.get_mesh_world(merge_meshes=True)
    assert len(merged.mesh) == 1 and len(merged.mesh[0].faces) > 0

"""Closure coverage for portable geometry value-model semantics."""

import json

import pytest
import torch

from curobo._src.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    SceneCfg,
    Sphere,
    VoxelGrid,
    batch_tensor_cube,
    tensor_capsule,
    tensor_cube,
    tensor_sphere,
)
from curobo._src.types.device_cfg import DeviceCfg


def test_tensor_geometry_broadcasts_radii_and_honours_caller_output_buffers():
    cfg = DeviceCfg()
    centers = torch.tensor(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    sphere = tensor_sphere(centers, torch.tensor((0.1, 0.2)), device_cfg=cfg)
    assert sphere.shape == (2, 4)
    torch.testing.assert_close(sphere[:, 3], torch.tensor((0.1, 0.2)))

    output = torch.empty_like(sphere)
    assert tensor_sphere(centers, 0.3, output, cfg) is output
    torch.testing.assert_close(output[:, 3], torch.full((2,), 0.3))

    capsule = tensor_capsule(
        torch.zeros(3), torch.tensor(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
        torch.tensor(((0.2,), (0.4,))), device_cfg=cfg,
    )
    assert capsule.shape == (2, 7)
    torch.testing.assert_close(capsule[:, 6], torch.tensor((0.2, 0.4)))
    with pytest.raises(ValueError, match="nonnegative"):
        tensor_sphere(torch.zeros(3), -0.1)


def test_cube_helpers_validate_transforms_and_normalize_equivalent_quaternions():
    dims, inverse = tensor_cube([1, 2, 3, 2, 0, 0, 0], [1, 2, 3])
    torch.testing.assert_close(dims, torch.tensor((1.0, 2.0, 3.0)))
    torch.testing.assert_close(inverse, torch.tensor((-1.0, -2.0, -3.0, 1.0, 0.0, 0.0, 0.0)))
    batch_dims, batch_inverse = batch_tensor_cube(
        [[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]], [[1, 1, 1], [2, 2, 2]],
    )
    assert batch_dims.shape == (2, 3) and batch_inverse.shape == (2, 7)
    with pytest.raises(ValueError, match="nonzero"):
        tensor_cube([0, 0, 0, 0, 0, 0, 0], [1, 1, 1])


def test_scene_json_round_trip_preserves_collision_geometry_and_independent_clone():
    scene = SceneCfg(
        cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], [1, 2, 3], material=Material(0.2, 0.6))],
        sphere=[Sphere("ball", position=[1, 0, 0], radius=0.25)],
        capsule=[Capsule("cap", base=[0, 0, 0], tip=[0, 0, 1], radius=0.1)],
        cylinder=[Cylinder("cyl", pose=[0, 1, 0, 1, 0, 0, 0], radius=0.2, height=1.0)],
        mesh=[Mesh("tri", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])],
        voxel=[VoxelGrid("grid", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2], voxel_size=1.0,
                         feature_tensor=torch.arange(8.0))],
    )
    encoded = json.loads(json.dumps(scene.to_dict()))
    restored = SceneCfg.create(encoded)
    assert [item.name for item in restored] == [item.name for item in scene]
    assert restored.cuboid[0].material == Material(0.2, 0.6)
    assert restored.voxel[0].feature_tensor.shape == (8,)

    copied = restored.clone()
    copied.voxel[0].feature_tensor.add_(100)
    copied.cuboid[0].dims[0] = 99
    assert not torch.equal(copied.voxel[0].feature_tensor, restored.voxel[0].feature_tensor)
    assert restored.cuboid[0].dims[0] == 1
    assert len(restored.create_collision_support_world(restored).mesh) == 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_geometry_tensor_and_voxel_helpers_remain_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = DeviceCfg(device="mps")
    sphere = tensor_sphere(torch.zeros((2, 3), device="mps"), torch.tensor((0.1, 0.2), device="mps"), device_cfg=cfg)
    dims, inverse = tensor_cube(torch.tensor([0, 0, 0, 1, 0, 0, 0], device="mps"),
                                 torch.ones(3, device="mps"), cfg)
    grid = VoxelGrid("grid", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2], voxel_size=1.0,
                     feature_tensor=torch.arange(8.0, device="mps"), device_cfg=cfg)
    assert sphere.device.type == dims.device.type == inverse.device.type == "mps"
    assert grid.get_occupied_voxels(feature_threshold=2.0).device.type == "mps"

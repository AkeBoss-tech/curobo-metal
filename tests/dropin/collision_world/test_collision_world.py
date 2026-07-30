from __future__ import annotations

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg, create_scene_collision
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg


def _query(scene: SceneCfg, spheres: torch.Tensor):
    checker = create_scene_collision(SceneCollisionCfg(scene_model=scene))
    buffer = CollisionBuffer.from_shape(spheres.shape, DeviceCfg(device=spheres.device))
    return checker, checker.get_sphere_distance_raw(
        spheres, buffer, spheres.new_tensor(1), spheres.new_tensor(0)
    )


def test_cuboid_discrete_buffers_mutation_and_environments() -> None:
    scenes = [
        SceneCfg(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2])]),
        SceneCfg(cuboid=[Cuboid("far", [10, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2])]),
    ]
    checker = create_scene_collision(SceneCollisionCfg(scene_model=scenes))
    spheres = torch.tensor([[[[1.1, 0, 0, .2]]], [[[1.1, 0, 0, .2]]]])
    buffer = CollisionBuffer.from_shape(spheres.shape, DeviceCfg())
    result = checker.get_sphere_distance_raw(
        spheres, buffer, torch.tensor(2.), torch.tensor(0.),
        env_query_idx=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([.2, 0.]))
    assert buffer.gradient[0, 0, 0, 0] == -2
    checker.enable_obstacle("box", False)
    assert checker.get_obstacle_names() == ["box"]
    assert checker.check_obstacle_exists("box")
    disabled = checker.get_sphere_distance_raw(
        spheres[:1], buffer, torch.tensor(1.), torch.tensor(0.)
    )
    assert disabled.item() == 0


def test_mesh_and_voxel_esdf_paths() -> None:
    mesh = Mesh(
        "triangle", pose=[0, 0, 0, 1, 0, 0, 0],
        vertices=[[0, -1, -1], [0, 1, -1], [0, 0, 1]], faces=[[0, 1, 2]],
    )
    _, mesh_result = _query(
        SceneCfg(mesh=[mesh]), torch.tensor([[[[.1, 0, 0, .2]]]])
    )
    assert mesh_result.item() == pytest.approx(.1)

    values = torch.ones((3, 3, 3)) * .5
    voxel = VoxelGrid(
        "esdf", pose=[0, 0, 0, 1, 0, 0, 0], dims=[3, 3, 3],
        voxel_size=1, feature_tensor=values,
    )
    _, voxel_result = _query(
        SceneCfg(voxel=[voxel]), torch.tensor([[[[0, 0, 0, .6]]]])
    )
    assert voxel_result.item() == pytest.approx(.1)


def test_swept_detects_between_knots_and_rejects_noop_option() -> None:
    scene = SceneCfg(cuboid=[Cuboid("thin", [0, 0, 0, 1, 0, 0, 0], dims=[.2, 2, 2])])
    checker = create_scene_collision(SceneCollisionCfg(scene_model=scene))
    spheres = torch.tensor([[[[-1, 0, 0, .2]], [[1, 0, 0, .2]]]])
    buffer = CollisionBuffer.from_shape(spheres.shape, DeviceCfg())
    swept = checker.get_swept_sphere_distance_raw(
        spheres, buffer, torch.tensor(1.), torch.tensor(0.), torch.tensor(1.)
    )
    assert swept.max() > 0
    with pytest.raises(NotImplementedError, match="speed metric"):
        checker.get_swept_sphere_distance_raw(
            spheres, buffer, torch.tensor(1.), torch.tensor(0.), torch.tensor(1.),
            enable_speed_metric=True,
        )


def test_public_aliases_and_unsupported_obstacle_error() -> None:
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

    assert RobotCollisionChecker.__name__ == "RobotSceneCollision"
    assert RobotCollisionCheckerCfg.__name__ == "RobotSceneCollisionCfg"
    sphere = __import__("curobo._src.geom.types", fromlist=["Sphere"]).Sphere("s", radius=1)
    with pytest.raises(NotImplementedError, match="sphere"):
        create_scene_collision(SceneCollisionCfg(scene_model=SceneCfg(sphere=[sphere])))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_fallback_disabled_cuboid() -> None:
    device = torch.device("mps")
    cfg = DeviceCfg(device=device)
    scene = SceneCfg(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2], device_cfg=cfg)])
    checker = create_scene_collision(SceneCollisionCfg(device_cfg=cfg, scene_model=scene))
    spheres = torch.tensor([[[[1.1, 0, 0, .2]]]], device=device)
    buffer = CollisionBuffer.from_shape(spheres.shape, cfg)
    result = checker.get_sphere_distance_raw(
        spheres, buffer, torch.tensor(1., device=device), torch.tensor(0., device=device)
    )
    assert result.item() == pytest.approx(.1, abs=1e-5)

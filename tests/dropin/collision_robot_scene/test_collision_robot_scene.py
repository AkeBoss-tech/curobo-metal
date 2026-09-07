"""Pinned behavioral coverage for portable :class:`RobotSceneCollision`."""

from __future__ import annotations

import pytest
import torch

from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.types.device_cfg import DeviceCfg


def _checker(*, device: str = "cpu", with_scene: bool = True) -> RobotSceneCollision:
    scene = None
    if with_scene:
        scene = SceneCfg(
            sphere=[Sphere("world", pose=[10, 0, 0, 1, 0, 0, 0], radius=0.25)]
        )
    return RobotSceneCollision(
        RobotSceneCollisionCfg.load_from_config(
            scene_model=scene, device_cfg=DeviceCfg(device=device)
        )
    )


def test_tool_frames_and_batched_validation_preserve_batch_horizon_axes() -> None:
    checker = _checker(with_scene=False)
    assert checker.tool_frames == checker.kinematics.tool_frames

    # B != H exposes accidental [B,H] -> [B,H,S] broadcasting.  The public
    # self-collision result remains [B,H], but validation must be [B,H].
    q = checker.kinematics.default_joint_position.repeat(2, 3, 1)
    valid = checker.validate(q)
    assert valid.shape == (2, 3)
    assert valid.dtype is torch.bool


def test_empty_world_resets_reusable_collision_gradient() -> None:
    checker = _checker(with_scene=False)
    spheres = checker.get_kinematics(
        checker.kinematics.default_joint_position.unsqueeze(0).unsqueeze(1)
    ).robot_spheres
    checker.setup_batch_tensors(1, 1)
    checker.collision_buffer.gradient.fill_(3.0)

    distance, gradient = checker.get_collision_vector(spheres)
    assert distance.shape == spheres.shape[:-1]
    torch.testing.assert_close(distance, torch.zeros_like(distance))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_point_robot_distance_handles_single_and_batched_clouds() -> None:
    checker = _checker(with_scene=False)
    q = checker.kinematics.default_joint_position.unsqueeze(0)
    spheres = checker.get_kinematics(q.unsqueeze(1)).robot_spheres.squeeze(1)
    center = spheres[0, 0, :3]
    radius = spheres[0, 0, 3]
    points = torch.stack((center, center + torch.tensor([2.0, 0.0, 0.0])))

    values = checker.get_point_robot_distance(points, q)
    assert values.shape == (2,)
    assert values[0] >= radius - 1e-5
    assert values[1] < 0

    q_batch = q.repeat(2, 1)
    cloud_batch = points.unsqueeze(0).repeat(2, 1, 1)
    batched = checker.get_point_robot_distance(cloud_batch, q_batch)
    assert batched.shape == (2, 2)
    torch.testing.assert_close(batched[0], values)
    torch.testing.assert_close(batched[1], values)

    with pytest.raises(ValueError, match="robot batch"):
        checker.get_point_robot_distance(cloud_batch, q.repeat(3, 1))
    with pytest.raises(ValueError, match="q must have shape"):
        checker.get_point_robot_distance(points, q.squeeze(0))


def test_query_input_and_environment_validation_are_explicit() -> None:
    checker = _checker()
    spheres = checker.get_kinematics(
        checker.kinematics.default_joint_position.unsqueeze(0).unsqueeze(1)
    ).robot_spheres
    with pytest.raises(ValueError, match="env_query_idx"):
        checker.get_collision_distance(spheres, torch.tensor([[0]]))
    with pytest.raises(TypeError, match="floating-point"):
        checker.get_collision_distance(spheres.to(torch.int64))
    with pytest.raises(ValueError, match="joint_position dof"):
        checker.validate(torch.zeros(1, checker.kinematics.dof + 1))
    with pytest.raises(ValueError, match="\[batch, horizon, dof\]"):
        checker.get_kinematics(torch.zeros(1, checker.kinematics.dof))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_robot_scene_point_and_empty_world_queries_stay_on_mps(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    checker = _checker(device="mps", with_scene=False)
    q = checker.kinematics.default_joint_position.unsqueeze(0)
    points = torch.zeros((4, 3), device="mps")
    value = checker.get_point_robot_distance(points, q)
    assert value.device.type == "mps"
    state = checker.get_kinematics(q.unsqueeze(1))
    distance, gradient = checker.get_collision_vector(state)
    assert distance.device.type == "mps"
    assert gradient.device.type == "mps"


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize("geometry", ["cuboid", "sphere", "capsule", "cylinder", "mixed"])
def test_collision_constraint_retains_penetration_after_world_update(device, geometry):
    from curobo.types import DeviceCfg
    from curobo.scene import Capsule, Cuboid, Cylinder, Scene, Sphere

    def scene(x):
        return Scene(
            cuboid=[Cuboid(name="box", pose=[x, 0, 0, 1, 0, 0, 0], dims=[0.4, 0.4, 0.4])] if geometry in {"cuboid", "mixed"} else [],
            sphere=[Sphere(name="ball", pose=[x, 0, 0, 1, 0, 0, 0], radius=0.2)] if geometry in {"sphere", "mixed"} else [],
            capsule=[Capsule(name="capsule", pose=[x, 0, 0, 1, 0, 0, 0], radius=0.2, base=[0, 0, -1], tip=[0, 0, 1])] if geometry == "capsule" else [],
            cylinder=[Cylinder(name="cylinder", pose=[x, 0, 0, 1, 0, 0, 0], radius=0.2, height=0.4)] if geometry == "cylinder" else [],
        )

    checker = RobotSceneCollision(RobotSceneCollisionCfg.load_from_config(
        scene_model=scene(3.0), device_cfg=DeviceCfg(device)))
    sphere = torch.tensor([[[[0.05, 0, 0, 0.05]]]], device=device, requires_grad=True)
    far = checker.get_collision_constraint(sphere).clone()
    checker.update_world(scene(0.0))
    near = checker.get_collision_constraint(sphere)
    torch.testing.assert_close(far, torch.zeros_like(far))
    torch.testing.assert_close(near, torch.full_like(near, 0.2))
    near.sum().backward()
    assert sphere.grad[..., 0].abs().min().item() > 0.5
    checker.update_world(scene(3.0))
    torch.testing.assert_close(checker.get_collision_constraint(sphere), far)

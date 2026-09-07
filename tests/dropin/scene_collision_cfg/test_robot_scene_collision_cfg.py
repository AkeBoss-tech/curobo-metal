"""Lifecycle coverage for the portable robot/scene collision configuration."""

from __future__ import annotations

import pytest
import torch

from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
from curobo._src.types.device_cfg import DeviceCfg


def _scene(name: str, center: float) -> SceneCfg:
    return SceneCfg(
        cuboid=[Cuboid(name, [center, 0, 0, 1, 0, 0, 0], [0.2, 0.2, 0.2], device_cfg=DeviceCfg("cpu"))],
        sphere=[Sphere(f"{name}_sphere", [center + 1, 0, 0, 1, 0, 0, 0], 0.1, device_cfg=DeviceCfg("cpu"))],
    )


def test_clone_owns_kinematics_costs_sampler_and_builtin_world() -> None:
    config = RobotSceneCollisionCfg.load_from_config(scene_model=_scene("source", 1.0), device_cfg=DeviceCfg("cpu"))
    config.sampler.reset()
    cloned = config.clone()

    assert cloned is not config
    assert cloned.kinematics is not config.kinematics
    assert cloned.sampler is not config.sampler
    assert cloned.cspace_cost is not config.cspace_cost
    assert cloned.scene_model is not config.scene_model
    assert cloned.collision_cost.config.scene_collision_checker is cloned.scene_model
    torch.testing.assert_close(config.sampler.get_samples(5), cloned.sampler.get_samples(5))

    cloned.update_collision_parameters(0.4)
    assert cloned.contact_distance == pytest.approx(0.2)
    assert config.collision_cost.config.activation_distance.item() == pytest.approx(0.2)
    assert cloned.collision_cost.config.activation_distance.item() == pytest.approx(0.4)

    cloned.update_scene_model(_scene("clone_only", 3.0))
    assert config.scene_model.get_obstacle_names() == ["source", "source_sphere"]
    assert cloned.scene_model.get_obstacle_names() == ["clone_only", "clone_only_sphere"]


def test_scene_replacement_preserves_cost_identity_for_matching_environments() -> None:
    config = RobotSceneCollisionCfg.load_from_config(
        scene_model=[_scene("left", 1.0), _scene("right", 2.0)], num_envs=2
    , device_cfg=DeviceCfg("cpu"))
    checker = config.scene_model
    cost = config.collision_cost
    config.update_scene_model([_scene("near", 3.0), _scene("far", 4.0)])

    assert config.scene_model is checker
    assert config.collision_cost is cost
    assert cost.config.scene_collision_checker is checker
    assert checker.get_obstacle_names(0) == ["near", "near_sphere"]
    assert checker.get_obstacle_names(1) == ["far", "far_sphere"]

    config.update_scene_model(_scene("single", 5.0))
    assert config.num_envs == 1
    assert config.scene_model is not checker
    assert cost.config.scene_collision_checker is config.scene_model


def test_scene_lifecycle_detaches_and_validates_capacity() -> None:
    config = RobotSceneCollisionCfg.load_from_config(scene_model=_scene("world", 2.0), device_cfg=DeviceCfg("cpu"))
    config.update_scene_model(None)
    assert config.scene_model is None
    assert config.collision_cost.config.scene_collision_checker is None
    with pytest.raises(ValueError, match="n_cuboids"):
        config.update_scene_model(_scene("world", 2.0), n_cuboids=0)
    with pytest.raises(ValueError, match="rejection_ratio"):
        RobotSceneCollisionCfg(
            config.kinematics, config.sampler, config.bound_scale, config.cspace_cost,
            rejection_ratio=0,
            device_cfg=DeviceCfg("cpu"),
        )


def test_scene_growth_rebuilds_checker_instead_of_overflowing_a_cache() -> None:
    config = RobotSceneCollisionCfg.load_from_config(
        scene_model=_scene("small", 1.0), n_cuboids=1
    , device_cfg=DeviceCfg("cpu"))
    old_checker = config.scene_model
    grown = SceneCfg(cuboid=[
        Cuboid("first", [1, 0, 0, 1, 0, 0, 0], [0.2, 0.2, 0.2], device_cfg=DeviceCfg("cpu")),
        Cuboid("second", [2, 0, 0, 1, 0, 0, 0], [0.2, 0.2, 0.2], device_cfg=DeviceCfg("cpu")),
    ])
    config.update_scene_model(grown)
    assert config.scene_model is not old_checker
    assert config.scene_model.get_obstacle_names() == ["first", "second"]
    assert config.collision_cost.config.scene_collision_checker is config.scene_model


def test_to_cpu_dtype_preserves_scene_query_and_result_ownership() -> None:
    config = RobotSceneCollisionCfg.load_from_config(scene_model=_scene("world", 10.0), device_cfg=DeviceCfg("cpu"))
    moved = config.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.device_cfg.dtype == torch.float64
    assert moved.bound_scale.dtype == torch.float64
    assert moved.scene_model is not config.scene_model
    state = moved.kinematics._forward(moved.kinematics.default_joint_position.view(1, 1, -1))
    distance = moved.scene_model.get_sphere_distance_raw(
        state.robot_spheres,
        CollisionBuffer.from_shape(state.robot_spheres.shape, moved.device_cfg),
        torch.ones(1, **moved.device_cfg.as_torch_dict()),
        torch.zeros(1, **moved.device_cfg.as_torch_dict()),
    )
    assert distance.dtype == torch.float64
    assert torch.isfinite(distance).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_to_mps_rebuilds_builtin_scene_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    config = RobotSceneCollisionCfg.load_from_config(scene_model=_scene("world", 10.0), device_cfg=DeviceCfg("cpu"))
    moved = config.to(DeviceCfg(torch.device("mps")))
    state = moved.kinematics._forward(moved.kinematics.default_joint_position.view(1, 1, -1))
    assert moved.bound_scale.device.type == "mps"
    assert state.robot_spheres.device.type == "mps"
    assert moved.scene_model.device_cfg.device.type == "mps"

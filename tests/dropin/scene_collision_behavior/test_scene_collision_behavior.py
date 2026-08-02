"""Behavioral coverage for the portable high-level robot-scene checker."""

from __future__ import annotations

import torch
import pytest

from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.types import SceneCfg, Sphere


def _scene(name: str, center: float) -> SceneCfg:
    return SceneCfg(
        sphere=[Sphere(name, pose=[center, 0, 0, 1, 0, 0, 0], radius=0.2)]
    )


def test_config_builds_real_costs_sampler_and_packaged_scene() -> None:
    config = RobotSceneCollisionCfg.load_from_config(scene_model="collision_test.yml")
    assert config.sampler is not None
    assert config.cspace_cost is not None
    assert config.self_collision_cost is not None
    assert config.collision_cost is not None
    assert config.collision_constraint is not None

    checker = RobotSceneCollision(config)
    checker.setup_batch_tensors(2, 3)
    assert checker.cspace_cost.batch_size == 2
    assert checker.self_collision_cost.horizon == 3
    assert checker.collision_buffer.distance.shape == (2, 3, checker.kinematics.total_spheres)

    # Sampling uses the configured deterministic Halton buffer rather than a
    # global RNG, which makes planner reset/replay behavior reproducible.
    config.sampler.reset()
    first = checker.sample(4, mask_valid=False)
    config.sampler.reset()
    second = checker.sample(4, mask_valid=False)
    torch.testing.assert_close(first, second)


def test_multi_environment_world_update_and_collision_constraint() -> None:
    config = RobotSceneCollisionCfg.load_from_config(
        scene_model=[_scene("left", 0.0), _scene("right", 20.0)], num_envs=2
    )
    checker = RobotSceneCollision(config)
    state = checker.get_kinematics(
        checker.kinematics.default_joint_position.repeat(2, 1)
    )
    env = torch.tensor([0, 1], device=state.robot_spheres.device)
    constraint = checker.get_collision_constraint(state, env)
    assert constraint.shape == state.robot_spheres.shape[:-1]
    assert torch.isfinite(constraint).all()

    checker.update_world([_scene("near", 3.0), _scene("far", 30.0)])
    assert checker.scene_model.get_obstacle_names(0) == ["near"]
    assert checker.scene_model.get_obstacle_names(1) == ["far"]

    q = checker.kinematics.default_joint_position.repeat(2, 1)
    assert checker.validate(q, env).shape == (2,)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_robot_scene_query_stays_on_mps_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    config = RobotSceneCollisionCfg.load_from_config(
        scene_model=_scene("world", 10.0), device_cfg=DeviceCfg(device="mps")
    )
    checker = RobotSceneCollision(config)
    state = checker.get_kinematics(checker.kinematics.default_joint_position.unsqueeze(0))
    distance, gradient = checker.get_collision_vector(state)
    assert distance.device.type == "mps"
    assert gradient.device.type == "mps"

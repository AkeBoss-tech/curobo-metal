"""High-level portable motion-planner world and retry lifecycle coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.motion.motion_planner import MotionPlanner, _axis_string_to_vector
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState


def _scene(name: str, x: float) -> SceneCfg:
    return SceneCfg(sphere=[Sphere(name, position=[x, 0.0, 0.0], radius=0.1)])


def _single() -> MotionPlanner:
    config = MotionPlannerCfg.create("franka.yml", use_cuda_graph=False)
    config.trajopt_solver_config.max_iterations = 2
    return MotionPlanner(config)


def _batched() -> BatchMotionPlanner:
    config = MotionPlannerCfg.create(
        "franka.yml", max_batch_size=2, multi_env=True, use_cuda_graph=False,
    )
    config.trajopt_solver_config.max_iterations = 2
    return BatchMotionPlanner(config)


def test_axis_helper_has_fresh_values_and_strict_validation() -> None:
    first = _axis_string_to_vector("x")
    first[0] = 0.0
    assert _axis_string_to_vector("x") == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="axis"):
        _axis_string_to_vector("bad")


def test_single_world_update_preserves_adapter_identity_and_syncs_all_stages() -> None:
    planner = _single()
    planner.update_world(_scene("first", 3.0))
    world = planner.scene_collision_checker
    assert planner.world_generation == 1
    assert planner.ik_solver.scene_collision_checker is world
    assert planner.trajopt_solver.scene_collision_checker is world
    assert planner.trajopt_solver._pose_ik.scene_collision_checker is world
    assert planner.attachment_manager._scene_collision is world

    planner.update_world(_scene("second", 4.0))
    assert planner.scene_collision_checker is world
    assert world.get_obstacle_names() == ["second"]
    assert planner.world_generation == 2
    assert planner.config.scene_collision_cfg.scene_model.sphere[0].name == "second"

    replacement = SceneCollisionCfg(planner.device_cfg, _scene("replacement", 5.0))
    planner.update_world(replacement)
    assert planner.scene_collision_checker is not world
    assert planner.scene_collision_checker.get_obstacle_names() == ["replacement"]
    assert planner.attachment_manager._scene_collision is planner.scene_collision_checker


def test_batch_world_update_requires_exact_environment_count_and_keeps_env_routing() -> None:
    planner = _batched()
    planner.update_world([_scene("left", 3.0), _scene("right", 30.0)])
    world = planner.scene_collision_checker
    assert world.num_envs == 2
    assert world.get_obstacle_names(0) == ["left"]
    assert world.get_obstacle_names(1) == ["right"]
    assert planner.ik_solver.scene_collision_checker is world
    assert planner.trajopt_solver._pose_ik.scene_collision_checker is world

    generation = planner.world_generation
    with pytest.raises(ValueError, match="environment count"):
        planner.update_world(_scene("wrong", 2.0))
    assert planner.scene_collision_checker is world
    assert planner.world_generation == generation


def test_destroyed_planners_reject_new_work_but_are_idempotent() -> None:
    planner = _single()
    planner.destroy()
    planner.destroy()
    assert planner.is_destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        planner.update_world(_scene("late", 2.0))
    with pytest.raises(RuntimeError, match="destroyed"):
        planner.compute_kinematics(planner.default_joint_state)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_multi_environment_world_update_stays_on_mps_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    config = MotionPlannerCfg.create(
        "franka.yml", max_batch_size=2, multi_env=True, use_cuda_graph=False,
        device_cfg=DeviceCfg(torch.device("mps"), torch.float32),
    )
    planner = BatchMotionPlanner(config)
    planner.update_world([_scene("left", 3.0), _scene("right", 30.0)])
    state = planner.default_joint_state.position.repeat(2, 1)
    result = planner.compute_kinematics(JointState.from_position(state, planner.joint_names))
    assert result.robot_spheres.device.type == "mps"

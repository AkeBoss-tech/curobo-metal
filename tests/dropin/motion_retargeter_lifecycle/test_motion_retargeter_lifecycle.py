"""Portable world, retry, and teardown lifecycle checks for retargeting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


def _criteria(weight: float = 1.0):
    return {
        "panda_hand": ToolPoseCriteria.track_position_and_orientation(
            xyz=[weight, weight, weight], rpy=[weight, weight, weight]
        )
    }


def _scene(name: str, x: float) -> SceneCfg:
    return SceneCfg(sphere=[Sphere(name, position=[x, 0.0, 0.0], radius=0.1)])


def _config(*, device_cfg=DeviceCfg(), scene_model=None, use_mpc=False, load_collision_spheres=True):
    return MotionRetargeterCfg.create(
        "franka.yml", _criteria(), device_cfg=device_cfg, scene_model=scene_model,
        self_collision_check=False, use_mpc=use_mpc, num_seeds_global=1,
        num_seeds_local=1, steps_per_target=1, mpc_warm_start_num_iters=1,
        mpc_cold_start_num_iters=1, load_collision_spheres=load_collision_spheres,
    )


def _goal(retargeter: MotionRetargeter) -> GoalToolPose:
    poses = retargeter.kinematics.compute_kinematics(retargeter.default_joint_state).tool_poses
    return GoalToolPose(
        retargeter.tool_frames, poses.position.unsqueeze(3), poses.quaternion.unsqueeze(3)
    )


def test_concrete_world_config_and_runtime_update_share_one_adapter_and_reset_state():
    retargeter = MotionRetargeter(_config(scene_model=_scene("initial", 3.0)))
    goal = _goal(retargeter)
    retargeter.solve_frame(goal)
    initial_world = retargeter.scene_collision_checker
    assert initial_world.get_obstacle_names() == ["initial"]

    retargeter.update_world(_scene("replacement", 4.0))
    world = retargeter.scene_collision_checker
    assert world is initial_world
    assert world.get_obstacle_names() == ["replacement"]
    assert retargeter._local_ik_solver.scene_collision_checker is world
    assert retargeter.config.scene_model is world
    assert retargeter.world_generation == 1
    assert retargeter.last_result is None
    assert retargeter._prev_solution is None

    retargeter.update_world(_scene("latest", 5.0))
    assert retargeter.scene_collision_checker is world
    assert world.get_obstacle_names() == ["latest"]
    assert retargeter.world_generation == 2


def test_runtime_world_rejects_solver_built_without_robot_spheres():
    retargeter = MotionRetargeter(_config(load_collision_spheres=False))
    with pytest.raises(RuntimeError, match="load_collision_spheres=True"):
        retargeter.update_world(_scene("late", 4.0))


def test_mpc_world_update_installs_the_same_adapter_in_ik_and_trajopt_children():
    retargeter = MotionRetargeter(_config(use_mpc=True, scene_model=SceneCfg()))
    retargeter.update_world(_scene("mpc", 4.0))
    world = retargeter.scene_collision_checker
    mpc = retargeter._mpc_solver
    assert mpc.scene_collision_checker is world
    assert mpc._ik.scene_collision_checker is world
    assert mpc._trajopt._scene_collision_checker is world
    assert mpc._trajopt._pose_ik.scene_collision_checker is world


def test_criteria_updates_recompile_then_require_same_ordered_tool_topology():
    retargeter = MotionRetargeter(_config())
    retargeter.solve_frame(_goal(retargeter))
    retargeter.update_tool_pose_criteria(_criteria(0.25))

    assert retargeter.config.tool_pose_criteria["panda_hand"].terminal_pose_axes_weight_factor.device == retargeter.default_joint_state.device
    assert retargeter._prev_solution is None
    assert retargeter.last_result is None
    with pytest.raises(ValueError, match="ordered tool frames"):
        retargeter.update_tool_pose_criteria(
            {
                "panda_link7": ToolPoseCriteria.track_position(),
                "panda_hand": ToolPoseCriteria.track_position(),
            }
        )


def test_failed_local_rows_hold_prior_solution_and_report_failure(monkeypatch):
    retargeter = MotionRetargeter(_config())
    goal = _goal(retargeter)
    first = retargeter.solve_frame(goal)
    prior = first.joint_state.position.clone()

    def failed_local(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            solution=prior[:, None] + 0.5,
            success=torch.zeros((1, 1), dtype=torch.bool, device=prior.device),
            js_solution=None,
        )

    monkeypatch.setattr(retargeter._local_ik_solver, "solve_pose", failed_local)
    result = retargeter.solve_frame(goal)
    assert torch.allclose(result.joint_state.position, prior)
    assert torch.equal(retargeter.last_failure_mask, torch.tensor([True], device=prior.device))
    assert torch.allclose(retargeter._prev_solution, prior)


def test_destroy_is_idempotent_and_rejects_future_mutation_or_solves():
    retargeter = MotionRetargeter(_config())
    retargeter.destroy()
    retargeter.destroy()
    assert retargeter.is_destroyed
    with pytest.raises(RuntimeError, match="destroyed"):
        retargeter.reset()
    with pytest.raises(RuntimeError, match="destroyed"):
        retargeter.update_world(_scene("late", 4.0))
    with pytest.raises(RuntimeError, match="destroyed"):
        retargeter.solve_frame(_goal(retargeter))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_world_and_failure_lifecycle_remain_device_resident(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    retargeter = MotionRetargeter(_config(
        device_cfg=DeviceCfg(torch.device("mps"), torch.float32), scene_model=SceneCfg()
    ))
    goal = _goal(retargeter)
    retargeter.solve_frame(goal)
    retargeter.update_world(_scene("mps", 4.0))
    output = retargeter.solve_frame(goal)
    assert retargeter.scene_collision_checker.device_cfg.device.type == "mps"
    assert output.joint_state.device.type == "mps"

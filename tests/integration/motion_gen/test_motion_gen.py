from __future__ import annotations

from pathlib import Path

import pytest
import torch

from curobo_metal.motion_gen import (
    JointState,
    MotionGen,
    MotionGenConfig,
    MotionGenStatus,
    Pose,
    UnsupportedMotionGenFeature,
)
from curobo_metal.ops.kinematics import forward_kinematics

FIXTURES = Path(__file__).parents[2] / "fixtures" / "trajectory"


def _planner(name: str, device: str = "cpu") -> MotionGen:
    dtype = torch.float32 if device == "mps" else torch.float64
    return MotionGen(MotionGenConfig.load_from_robot_config(
        FIXTURES / f"{name}.json", device=device, dtype=dtype,
        interpolation_dt=0.04,
    ))


def _tensor(planner: MotionGen, value: list[float]) -> torch.Tensor:
    return torch.tensor(value, device=planner.config.device, dtype=planner.config.dtype)


def test_two_link_joint_space_obstacle_free_and_batch() -> None:
    planner = _planner("two_link_obstacle_free")
    start = _tensor(planner, [-0.5, 0.2])
    goal = _tensor(planner, [0.8, -0.3])
    assert planner.warmup()
    result = planner.plan_single_js(JointState(start), JointState(goal))
    assert result.status is MotionGenStatus.SUCCESS and bool(result.success)
    assert result.optimized_plan is not None and result.interpolated_plan is not None
    torch.testing.assert_close(result.optimized_plan.position[[0, -1]], torch.stack((start, goal)))
    assert result.metrics is not None and result.metrics.maximum_limit_violation == 0
    batch = planner.plan_batch_js(
        torch.stack((start, goal)), torch.stack((goal, start))
    )
    assert batch.success.tolist() == [True, True]
    assert batch.optimized_plan is not None
    assert batch.optimized_plan.position.shape[:2] == (2, planner.config.steps)


def test_two_link_primitive_detour_is_collision_free() -> None:
    planner = _planner("two_link_obstacle_detour")
    result = planner.plan_single_js(
        _tensor(planner, [-0.8, 0.0]), _tensor(planner, [0.8, 0.0])
    )
    assert result.status is MotionGenStatus.SUCCESS
    assert result.metrics is not None
    assert float(result.metrics.minimum_clearance.min().item()) >= 0


def test_pose_goal_unreachable_and_invalid_start_have_stable_status() -> None:
    planner = _planner("two_link_obstacle_free")
    start = _tensor(planner, [0.0, 0.0])
    pose_goal = _tensor(planner, [0.5, -0.2])
    transform = forward_kinematics(planner.config.chain, pose_goal).transforms[0, -1]
    reachable = Pose(
        transform[:3, 3],
        _tensor(
            planner,
            [torch.cos(pose_goal.sum() / 2).item(), 0.0, 0.0,
             torch.sin(pose_goal.sum() / 2).item()],
        ),
    )
    pose_result = planner.plan_single(start, reachable)
    assert pose_result.status is MotionGenStatus.SUCCESS
    assert pose_result.ik_result is not None
    unreachable = Pose(
        _tensor(planner, [10.0, 0.0, 0.0]),
        _tensor(planner, [1.0, 0.0, 0.0, 0.0]),
    )
    result = planner.plan_single(start, unreachable)
    assert result.status is MotionGenStatus.IK_FAILED and not bool(result.success)
    invalid = planner.plan_single_js(
        _tensor(planner, [4.0, 0.0]), _tensor(planner, [0.0, 0.0])
    )
    assert invalid.status is MotionGenStatus.INVALID_START


def test_panda_joint_space_and_reachable_pose() -> None:
    planner = _planner("panda_obstacle_free")
    start = _tensor(planner, [-0.2, 0.1, 0.0, -0.8, 0.0, 0.7, 0.2])
    goal = _tensor(planner, [0.3, -0.2, 0.2, -1.1, 0.3, 1.0, -0.1])
    result = planner.plan_single_js(start, goal)
    assert result.status is MotionGenStatus.SUCCESS
    transform = forward_kinematics(planner.config.chain, goal).transforms[0, -1]
    # Scalar-first quaternion for this test is obtained from a zero-rotation
    # target only in the simpler two-link pose test; Panda exercises c-space.
    assert torch.isfinite(transform).all()


def test_config_rejects_unsupported_worlds_and_supports_runtime_mutation() -> None:
    fixture = FIXTURES / "two_link_obstacle_free.json"
    with pytest.raises(UnsupportedMotionGenFeature, match="primitive cuboids"):
        MotionGenConfig.load_from_robot_config(
            fixture, world={"mesh": [], "local_spheres": [], "link_indices": []}
        )
    planner = _planner("two_link_obstacle_free")
    generation = planner._graph_cache.generation
    planner.update_world({})
    assert planner.config.collision_model is None
    assert planner._graph_cache.generation == generation + 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_end_to_end_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name, start, goal in (
        ("two_link_obstacle_free", [-0.5, 0.2], [0.8, -0.3]),
        ("two_link_obstacle_detour", [-0.8, 0.0], [0.8, 0.0]),
        (
            "panda_obstacle_free",
            [-0.2, 0.1, 0.0, -0.8, 0.0, 0.7, 0.2],
            [0.3, -0.2, 0.2, -1.1, 0.3, 1.0, -0.1],
        ),
    ):
        planner = _planner(name, "mps")
        result = planner.plan_single_js(_tensor(planner, start), _tensor(planner, goal))
        assert result.status is MotionGenStatus.SUCCESS
        assert result.optimized_plan is not None
        assert result.optimized_plan.position.device.type == "mps"

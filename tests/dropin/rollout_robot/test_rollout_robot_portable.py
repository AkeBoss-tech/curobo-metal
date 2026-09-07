"""Focused portable lifecycle coverage for the pinned RobotRollout surface."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class _Transition:
    """Small eager transition that makes rollout ownership testable."""

    def __init__(self, cfg):
        self.action_dim = 2
        self.action_horizon = self.horizon = 3
        self._dt = torch.tensor([0.1])
        self.action_bound_lows = torch.tensor([-1.0, -2.0])
        self.action_bound_highs = torch.tensor([1.0, 2.0])
        self.default_joint_position = torch.zeros(2)
        self.updated_batches = []

    def update_batch_size(self, batch_size):
        self.updated_batches.append(batch_size)

    def update_traj_dt(self, dt):
        self._dt = torch.as_tensor(dt).reshape(-1)

    def forward(self, start_state, actions, *args, **kwargs):
        del start_state, args, kwargs
        return JointState.from_position(actions)

    def filter_robot_state(self, state):
        return state

    def get_robot_command(self, current, actions, shift_steps=1, **kwargs):
        del kwargs
        return JointState.from_position(current.position + actions[:, shift_steps - 1])


def _cfg(*, scene_collision_cfg=None, device_cfg=DeviceCfg("cpu")):
    return RobotRolloutCfg(
        device_cfg=device_cfg,
        transition_model_cfg=SimpleNamespace(class_type=_Transition),
        scene_collision_cfg=scene_collision_cfg,
    )


def test_scene_config_builds_portable_checker_and_external_checker_wins():
    scene_cfg = SceneCollisionCfg(
        scene_model=SceneCfg(cuboid=[Cuboid("wall", [1, 0, 0, 1, 0, 0, 0], [1, 1, 1], device_cfg=DeviceCfg("cpu"))])
    , device_cfg=DeviceCfg("cpu"))
    owned = RobotRollout(_cfg(scene_collision_cfg=scene_cfg, device_cfg=DeviceCfg("cpu")))
    assert owned.scene_collision_checker is not None
    assert owned.scene_collision_checker.check_obstacle_exists("wall")

    external = object()
    supplied = RobotRollout(_cfg(scene_collision_cfg=scene_cfg, device_cfg=DeviceCfg("cpu")), scene_collision_checker=external)
    assert supplied.scene_collision_checker is external


def test_action_bounds_normalize_transition_constants_and_batch_validation():
    rollout = RobotRollout(_cfg(device_cfg=DeviceCfg("cpu")))
    torch.testing.assert_close(rollout.action_bounds, torch.tensor([[-1.0, -2.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="positive"):
        rollout.update_batch_size(0)
    with pytest.raises(TypeError, match="integer"):
        rollout.update_batch_size(True)
    with pytest.raises(TypeError, match="integer"):
        rollout.sample_random_actions(True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_goal_device_validation_is_explicit_on_mps():
    rollout = RobotRollout(_cfg(device_cfg=DeviceCfg(torch.device("mps"))))
    cpu_goal = GoalRegistry(current_js=JointState.from_position(torch.zeros(1, 2)))
    with pytest.raises(ValueError, match=r"goal\..* device"):
        rollout.update_params(cpu_goal)

    mps_goal = GoalRegistry(current_js=JointState.from_position(torch.zeros(1, 2, device="mps")))
    assert rollout.update_params(mps_goal)

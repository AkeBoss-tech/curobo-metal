"""Focused portable lifecycle coverage for ``RobotCostManager``.

The checks deliberately use no CUDA/Warp implementation details: they prove
that reconfiguration, mutable config helpers, and eager CPU/MPS execution
retain device-resident tensor semantics.
"""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.portable import CSpaceDistCostCfg, ToolPoseCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


class _RecorderCost:
    enabled = True

    def __init__(self):
        self.calls = []

    def enable_cost(self):
        self.enabled = True

    def disable_cost(self):
        self.enabled = False

    def setup_batch_tensors(self, batch_size, horizon):
        self.calls.append(("setup", batch_size, horizon))

    def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))

    def update_dt(self, dt):
        self.calls.append(("dt", dt))


def test_failed_reconfiguration_preserves_last_valid_manager():
    manager = RobotCostManager()
    good = RobotCostManagerCfg(start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0))
    manager.initialize_from_config(good)
    original_cost = manager.get_cost("start_cspace_dist")

    bad = RobotCostManagerCfg(tool_pose_cfg=ToolPoseCostCfg(weight=[1.0, 1.0]))
    with pytest.raises(ValueError, match="tool_frames"):
        manager.initialize_from_config(bad)

    assert manager.config is good
    assert manager.get_cost("start_cspace_dist") is original_cost
    assert manager.get_cost_component_names() == ["start_cspace_dist"]


def test_batch_reset_and_dt_lifecycle_validate_inputs_and_forward_calls():
    manager = RobotCostManager()
    recorder = _RecorderCost()
    manager.register_cost("recorder", recorder)

    manager.setup_batch_tensors(2, 3)
    manager.setup_batch_tensors(2, 3)
    assert recorder.calls == [("setup", 2, 3)]
    with pytest.raises(ValueError, match="non-negative"):
        manager.setup_batch_tensors(True, 3)
    with pytest.raises(TypeError, match="integer dtype"):
        manager.reset(torch.tensor([1.0]))
    manager.reset(torch.tensor([1], dtype=torch.int32), preserve=True)
    manager.update_dt(0.25)
    assert recorder.calls[-2][0] == "reset"
    assert recorder.calls[-2][1]["preserve"] is True
    assert recorder.calls[-1] == ("dt", 0.25)


def test_cfg_factory_accepts_configs_and_mutable_weights_are_shape_safe():
    direct = CSpaceDistCostCfg(weight=[1.0, 2.0])
    cfg = RobotCostManagerCfg.create({"start_cspace_dist_cfg": direct})
    assert cfg.start_cspace_dist_cfg is direct
    cfg.update_regularization_weight(distance_weight=[3.0, 4.0])
    torch.testing.assert_close(direct.weight, torch.tensor([3.0, 4.0]))

    with pytest.raises(TypeError, match="dict"):
        RobotCostManagerCfg.create([])
    with pytest.raises(TypeError, match="dict or"):
        RobotCostManagerCfg.create({"start_cspace_dist_cfg": 1.0})
    with pytest.raises(ValueError, match="distance_weight"):
        cfg.update_regularization_weight(distance_weight=[1.0, 2.0, 3.0])


def test_enabled_tool_pose_cost_requires_robot_kinematics_state():
    cfg = RobotCostManagerCfg(tool_pose_cfg=ToolPoseCostCfg(weight=[1.0, 1.0], tool_frames=["ee"]))
    manager = RobotCostManager().initialize_from_config(cfg)
    state = RobotState(JointState.from_position(torch.zeros(1, 2, 1)))
    goal = GoalRegistry(
        link_goal_poses=GoalToolPose(
            ["ee"],
            torch.zeros(1, 2, 1, 1, 3),
            torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]).expand(1, 2, 1, 1, 4),
        )
    )
    with pytest.raises(ValueError, match="state.tool_poses"):
        manager.compute_costs(state, goal=goal)


def test_joint_torque_must_match_joint_trajectory_shape_and_device():
    state = RobotState(
        JointState.from_position(torch.zeros(1, 2, 2)),
        joint_torque=torch.zeros(1, 2, 1),
    )
    with pytest.raises(ValueError, match="joint_torque must match"):
        RobotCostManager().initialize_from_config(RobotCostManagerCfg()).compute_costs(state)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_cost_manager_rejects_cpu_config_before_allocating_buffers():
    manager = RobotCostManager(DeviceCfg(torch.device("mps")))
    with pytest.raises(ValueError, match="not cost manager device"):
        manager.initialize_from_config(
            RobotCostManagerCfg(start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0))
        )
    assert manager.costs == {}
    assert manager._initialized is False

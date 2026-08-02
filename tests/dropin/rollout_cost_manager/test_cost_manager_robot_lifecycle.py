"""Robot-cost-manager lifecycle tests that do not require CUDA/Warp."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


class _RobotModel:
    def __init__(self, total_spheres=2, self_collision_config=None):
        self.total_spheres = total_spheres
        self._self_collision_config = self_collision_config
        self.tool_frames = []

    def get_self_collision_config(self):
        return self._self_collision_config


class _Transition:
    def __init__(self, total_spheres=2, self_collision_config=None, interpolation_steps=1):
        self.robot_model = _RobotModel(total_spheres, self_collision_config)
        self.interpolation_steps = interpolation_steps


class _SceneChecker:
    def get_sphere_distance(self, spheres, env_query_idx=None):
        del env_query_idx
        return torch.ones(spheres.shape[:-1], device=spheres.device, dtype=spheres.dtype)


def test_initialize_is_reconfigurable_and_skips_unbound_scene_costs():
    manager = RobotCostManager(DeviceCfg())
    no_checker = RobotCostManagerCfg(scene_collision_cfg=SceneCollisionCostCfg(weight=1.0))
    manager.initialize_from_config(no_checker)
    assert manager.get_cost_component_names() == []

    checker = _SceneChecker()
    with_checker = RobotCostManagerCfg(scene_collision_cfg=SceneCollisionCostCfg(weight=1.0))
    manager.initialize_from_config(with_checker, _Transition(total_spheres=2), checker)
    assert manager.get_cost_component_names() == ["scene_collision"]
    assert manager.get_cost("scene_collision").config.scene_collision_checker is checker
    assert manager.get_cost("scene_collision").config.num_spheres == 2

    # Re-initialization must not leave old collision terms active.
    manager.initialize_from_config(RobotCostManagerCfg())
    assert manager.get_cost_component_names() == []


def test_self_collision_interpolation_is_execution_local_and_zero_sphere_disables():
    cfg = RobotCostManagerCfg(self_collision_cfg=SelfCollisionCostCfg(weight=6.0))
    transition = _Transition(
        total_spheres=2,
        self_collision_config=SimpleNamespace(collision_pairs=torch.tensor([[0, 1]])),
        interpolation_steps=3,
    )
    manager = RobotCostManager(DeviceCfg()).initialize_from_config(cfg, transition)
    component = manager.get_cost("self_collision")
    assert component is not None
    torch.testing.assert_close(component.weight, torch.tensor([2.0]))
    # The caller configuration is reusable across manager instances.
    torch.testing.assert_close(cfg.self_collision_cfg.weight, torch.tensor([6.0]))

    empty = RobotCostManager(DeviceCfg()).initialize_from_config(
        RobotCostManagerCfg(self_collision_cfg=SelfCollisionCostCfg(weight=1.0)),
        _Transition(total_spheres=0, self_collision_config=SimpleNamespace()),
    )
    assert empty.has_cost("self_collision")
    assert empty.get_cost("self_collision").enabled is False


def test_uninitialized_update_is_noop_and_state_device_is_never_copied_implicitly():
    manager = RobotCostManager(DeviceCfg())
    manager.update_params(dt=0.25, tool_pose_criteria="not inspected before initialization")
    assert manager._initialized is False

    manager.initialize_from_config(RobotCostManagerCfg())
    state = RobotState(JointState.from_position(torch.zeros(1, 2, 1)))
    assert manager.compute_costs(state).is_empty()

    if torch.backends.mps.is_available():
        mps_manager = RobotCostManager(DeviceCfg(torch.device("mps")))
        with pytest.raises(ValueError, match="device"):
            mps_manager.compute_costs(state)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_cost_manager_keeps_collision_autograd_on_mps():
    device_cfg = DeviceCfg(torch.device("mps"))
    position = torch.ones((1, 3, 2), device="mps", requires_grad=True)
    spheres = torch.tensor(
        [[
            [[0.0, 0.0, 0.0, 0.3], [0.2, 0.0, 0.0, 0.3]],
            [[0.0, 0.0, 0.0, 0.3], [0.2, 0.0, 0.0, 0.3]],
            [[0.0, 0.0, 0.0, 0.3], [0.2, 0.0, 0.0, 0.3]],
        ]],
        device="mps",
        requires_grad=True,
    )
    state = RobotState(
        JointState.from_position(position),
        cuda_robot_model_state=SimpleNamespace(robot_spheres=spheres),
    )
    manager = RobotCostManager(device_cfg).initialize_from_config(
        RobotCostManagerCfg(
            self_collision_cfg=SelfCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        )
    )
    result = manager.compute_costs(state)
    assert result.names == ["self_collision"]
    result.get_sum(False).sum().backward()
    assert state.joint_state.position.device.type == "mps"
    assert spheres.grad is not None and torch.isfinite(spheres.grad).all()

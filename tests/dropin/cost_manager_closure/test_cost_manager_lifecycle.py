"""Focused lifecycle tests for the portable robot cost-manager closure."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.portable import CSpaceCostCfg
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


def _cspace_config(device_cfg: DeviceCfg = DeviceCfg("cpu")) -> RobotCostManagerCfg:
    return RobotCostManagerCfg(
        cspace_cfg=CSpaceCostCfg(
            weight=1.0,
            dof=1,
            squared_l2_regularization_weight=[2.0, 0.0],
            device_cfg=device_cfg,
        )
    )


def test_per_batch_current_dt_is_expanded_over_horizon_and_keeps_autograd() -> None:
    position = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], requires_grad=True
    )
    state = RobotState(JointState.from_position(position))
    current = JointState(torch.zeros(2, 1), velocity=torch.zeros(2, 1), device_cfg=DeviceCfg("cpu"))
    goal = GoalRegistry(current_js=current, current_state_dt=torch.tensor([0.5, 0.25]))

    manager = RobotCostManager(device_cfg=DeviceCfg("cpu")).initialize_from_config(_cspace_config(device_cfg=DeviceCfg("cpu")))
    values = manager.compute_costs(state, goal=goal)
    assert values.names == ["cspace"]
    assert values.values[0].shape == (2, 3, 1)
    values.get_sum(False).sum().backward()
    assert position.grad is not None and torch.isfinite(position.grad).all()
    # The registry remains reusable: normalization is an execution-local view.
    assert goal.current_state_dt.shape == (2,)


def test_goal_index_and_time_layouts_fail_before_cost_evaluation() -> None:
    state = RobotState(JointState.from_position(torch.zeros(2, 3, 1)))
    manager = RobotCostManager(device_cfg=DeviceCfg("cpu")).initialize_from_config(_cspace_config(device_cfg=DeviceCfg("cpu")))

    bad_index = GoalRegistry(idxs_goal_js=torch.tensor([0.0, 1.0]))
    with pytest.raises(TypeError, match="idxs_goal_js"):
        manager.compute_costs(state, goal=bad_index)

    malformed_dt = GoalRegistry(current_state_dt=torch.ones(2, 2, 1))
    with pytest.raises(ValueError, match="current_state_dt"):
        manager.compute_costs(state, goal=malformed_dt)


def test_config_factory_and_mutable_updates_never_hide_cross_device_tensors() -> None:
    with pytest.raises(TypeError, match="cspace_cfg must"):
        RobotCostManagerCfg(cspace_cfg=object())

    cfg = RobotCostManagerCfg.create(
        {"cspace_cfg": {"weight": 1.0, "device_cfg": DeviceCfg("cpu")}}, DeviceCfg("cpu")
    , device_cfg=DeviceCfg("cpu"))
    assert cfg.cspace_cfg is not None
    public_self = SelfCollisionCostCfg(weight=1.0, device_cfg=DeviceCfg("cpu"))
    assert RobotCostManagerCfg.create({"self_collision_cfg": public_self}, device_cfg=DeviceCfg("cpu")).self_collision_cfg is public_self

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    mps_cfg = RobotCostManagerCfg.create(
        {"scene_collision_cfg": {"weight": 1.0}}, device_cfg=DeviceCfg("mps")
    )
    with pytest.raises(ValueError, match="device"):
        mps_cfg.update_collision_activation_distance(torch.tensor(0.1))

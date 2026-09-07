"""Lifecycle tests for the pinned C-space cost public configuration APIs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from curobo._src.cost.cost_cspace_base import BaseCSpaceCost
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _limits(dof: int = 2, device: str = "cpu") -> JointLimits:
    cfg = DeviceCfg(torch.device(device))

    def pair(value: float) -> torch.Tensor:
        return cfg.to_device([[-value] * dof, [value] * dof])

    return JointLimits(
        [f"j{index}" for index in range(dof)],
        pair(1), pair(2), pair(3), pair(4), pair(5), cfg,
    )


def _state(device: str = "cpu") -> JointState:
    return JointState.from_position(torch.zeros((2, 3, 2), device=device))


def test_base_cspace_target_buffer_is_independent_and_validate_is_strict() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0, 1.0], activation_distance=[0.0, 0.0], dof=2,
        cost_type=CSpaceCostType.POSITION, joint_limits=_limits(), cspace_target_weight=2.0,
        device_cfg=DeviceCfg("cpu"),
    )
    cost = BaseCSpaceCost(cfg)
    assert not cost.cspace_target_enabled
    torch.testing.assert_close(cost.cspace_target_weight, torch.zeros(1))
    cost.setup_batch_tensors(2, 3)
    cost.enable_cspace_target()
    torch.testing.assert_close(cost.cspace_target_weight, torch.tensor([2.0]))
    cost.disable_cspace_target()
    torch.testing.assert_close(cfg.cspace_target_weight, torch.tensor([2.0]))
    assert cost.validate_input(_state(), torch.zeros((2, 3, 2)))
    target = JointState.from_position(torch.zeros((2, 2)))
    assert cost.validate_input(
        _state(), target_joint_state=target, idxs_target_joint_state=torch.tensor([0, 1])
    )
    with pytest.raises(ValueError, match="horizon"):
        cost.validate_input(JointState.from_position(torch.zeros((2, 4, 2))))
    with pytest.raises(ValueError, match="out-of-range"):
        cost.validate_input(
            _state(), target_joint_state=target, idxs_target_joint_state=torch.tensor([0, 2])
        )


def test_teleport_state_conversion_accepts_position_only_bounds() -> None:
    class PositionOnly:
        position = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])

    cfg = CSpaceCostCfg(
        weight=[1.0] * 5, activation_distance=[0.0] * 5,
        cost_type=CSpaceCostType.STATE, dof=2,
        device_cfg=DeviceCfg("cpu"),
    )
    cfg.set_bounds(PositionOnly(), teleport_mode=True)
    assert cfg.cost_type is CSpaceCostType.POSITION
    assert cfg.joint_limits is not None
    assert cfg.joint_limits is not PositionOnly


def test_cspace_bounds_reject_cross_device_record_when_mps_is_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple Metal")
    cfg = CSpaceCostCfg(
        weight=[1.0, 1.0], activation_distance=[0.0, 0.0], dof=2,
        cost_type=CSpaceCostType.POSITION, device_cfg=DeviceCfg(torch.device("mps")),
    )
    with pytest.raises(ValueError, match="device_cfg"):
        cfg.set_bounds(_limits())


def test_dist_cfg_enforces_weight_lifecycle_and_transition_selection() -> None:
    cfg = CSpaceDistCostCfg(
        weight=1.5, dof=2, only_terminal_cost=True,
        terminal_dof_weight=[2.0, 3.0], non_terminal_dof_weight=[7.0, 8.0],
        device_cfg=DeviceCfg("cpu"),
    )
    torch.testing.assert_close(cfg.non_terminal_dof_weight, torch.zeros(2))
    buffer = cfg.terminal_dof_weight
    assert cfg.update_terminal_dof_weight([4.0, 5.0]) is cfg
    assert cfg.terminal_dof_weight is buffer
    torch.testing.assert_close(buffer, torch.tensor([4.0, 5.0]))
    with pytest.raises(ValueError, match="contain 2 values"):
        cfg.update_terminal_dof_weight([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="provided dof weights"):
        CSpaceDistCostCfg(weight=1.0, dof=2, terminal_dof_weight=[1.0, 2.0, 3.0], device_cfg=DeviceCfg("cpu"))

    @dataclass
    class Transition:
        action_dim: int = 2
        null_space_weight: torch.Tensor = torch.tensor([2.0, 4.0])
        cspace_distance_weight: torch.Tensor = torch.tensor([1.0, 3.0])

    result = CSpaceDistCostCfg(weight=1.0, use_null_space=True, only_terminal_cost=False, device_cfg=DeviceCfg("cpu"))
    assert result.initialize_from_transition_model(Transition()) is result
    torch.testing.assert_close(result.terminal_dof_weight, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result.non_terminal_dof_weight, torch.tensor([2.0, 4.0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_base_cspace_lifecycle_and_validation_stay_on_mps() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0, 1.0], activation_distance=[0.0, 0.0], dof=2,
        cost_type=CSpaceCostType.POSITION, joint_limits=_limits(device="mps"),
        cspace_target_weight=1.0, device_cfg=DeviceCfg(torch.device("mps")),
    )
    cost = BaseCSpaceCost(cfg)
    cost.setup_batch_tensors(2, 3)
    position = torch.zeros((2, 3, 2), device="mps", requires_grad=True)
    assert cost.validate_input(JointState.from_position(position))
    cost.enable_cspace_target()
    assert cost.cspace_target_weight.device.type == "mps"

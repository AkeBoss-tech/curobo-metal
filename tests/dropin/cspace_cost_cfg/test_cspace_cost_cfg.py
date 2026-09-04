"""Pinned CSpaceCostCfg configuration semantics on portable backends."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_state import StateCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


def _limits(dof: int = 2, device_cfg: DeviceCfg = DeviceCfg()) -> JointLimits:
    def pair(scale: float) -> torch.Tensor:
        return device_cfg.to_device([[-scale] * dof, [scale] * dof])

    return JointLimits(
        [f"joint_{index}" for index in range(dof)],
        pair(1.0), pair(2.0), pair(3.0), pair(4.0), pair(5.0), device_cfg,
    )


def test_state_configuration_normalizes_values_and_selects_runtime_cost() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0] * 5,
        activation_distance=[0.1] * 5,
        squared_l2_regularization_weight=[0.2] * 5,
        cost_type="state",
        dof=2,
        joint_limits=_limits(),
        cspace_target_weight=0.5,
        cspace_non_terminal_weight_factor=0.25,
        cspace_target_dof_weight=[2.0, 3.0],
    )
    assert cfg.cost_type is CSpaceCostType.STATE
    assert cfg.class_type is StateCSpaceCost
    assert cfg.cspace_target_weight.shape == (1,)
    torch.testing.assert_close(cfg.cspace_target_dof_weight, torch.tensor([2.0, 3.0]))
    assert cfg.joint_limits is not None


def test_bounds_are_cloned_and_teleport_conversion_uses_position_terms() -> None:
    original = _limits()
    cfg = CSpaceCostCfg(
        weight=[1.0, 2.0, 3.0, 4.0, 5.0],
        activation_distance=[0.0, 0.1, 0.2, 0.3, 0.4],
        squared_l2_regularization_weight=[.1, .2, .3, .4, .5],
        cost_type=CSpaceCostType.STATE,
        dof=2,
    )
    assert cfg.set_bounds(original, teleport_mode=True) is cfg
    assert cfg.cost_type is CSpaceCostType.POSITION
    assert cfg.class_type is PositionCSpaceCost
    torch.testing.assert_close(cfg.weight, torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(cfg.activation_distance, torch.tensor([0.0, 0.4]))
    torch.testing.assert_close(cfg.squared_l2_regularization_weight, torch.tensor([.1, .2]))
    assert cfg.joint_limits is not original


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dof": True}, "dof"),
        ({"dof": 1.5}, "dof"),
        ({"weight": [1.0, -1.0]}, "weight"),
        ({"activation_distance": [0.0, -0.1]}, "activation_distance"),
        ({"cspace_target_dof_weight": [1.0, float("nan")]}, "cspace_target_dof_weight"),
    ],
)
def test_invalid_configuration_values_fail_at_construction(kwargs, message: str) -> None:
    values = dict(
        weight=[1.0, 1.0], activation_distance=[0.0, 0.0],
        cost_type=CSpaceCostType.POSITION, dof=2,
    )
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        CSpaceCostCfg(**values)


def test_retiming_flags_match_the_pinned_configuration_contract() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0, 1.0], activation_distance=[0.0, 0.0],
        cost_type=CSpaceCostType.POSITION, dof=2, retime_weights=True,
    )
    assert cfg.retime_weights is True


@dataclass
class _Transition:
    action_dim: int
    bounds: JointLimits
    teleport_mode: bool = False

    def get_state_bounds(self) -> JointLimits:
        return self.bounds


def test_transition_initialization_validates_action_dimension_and_bounds() -> None:
    cfg = CSpaceCostCfg(
        weight=[1.0] * 5, activation_distance=[0.0] * 5,
        cost_type=CSpaceCostType.STATE,
    )
    cfg.initialize_from_transition_model(_Transition(2, _limits(), teleport_mode=True))
    assert cfg.dof == 2
    assert cfg.cost_type is CSpaceCostType.POSITION
    assert cfg.class_type is PositionCSpaceCost

    with pytest.raises(TypeError, match="action_dim"):
        cfg.initialize_from_transition_model(_Transition(True, _limits()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_validated_configuration_and_cost_stay_on_mps() -> None:
    device_cfg = DeviceCfg(torch.device("mps"))
    cfg = CSpaceCostCfg(
        weight=[1.0, 0.0], activation_distance=[0.1, 0.0],
        cost_type=CSpaceCostType.POSITION, dof=2, device_cfg=device_cfg,
        joint_limits=_limits(device_cfg=device_cfg), cspace_target_weight=1.0,
    )
    position = torch.tensor([[[1.1, 0.0]]], device="mps", requires_grad=True)
    result = PositionCSpaceCost(cfg)(
        JointState.from_position(position),
        target_joint_state=JointState.from_position(torch.zeros_like(position)),
    )
    result.sum().backward()
    assert result.device.type == position.grad.device.type == "mps"

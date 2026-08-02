"""Portable behavior for the pinned RobotRolloutCfg assembly boundary."""

from copy import deepcopy
from dataclasses import dataclass

import pytest

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class _TransitionCfg:
    robot: object
    device_cfg: DeviceCfg
    value: int

    @classmethod
    def create(cls, data, robot_cfg, device_cfg):
        return cls(robot_cfg, device_cfg, int(data["value"]))


@dataclass
class _ManagerCfg:
    device_cfg: DeviceCfg
    value: int

    @classmethod
    def create(cls, data, device_cfg):
        return cls(device_cfg, int(data["value"]))


def test_component_factory_compiles_all_config_groups_without_mutating_input():
    device = DeviceCfg()
    robot = object()
    payload = {
        "sum_horizon": True,
        "sampler_seed": 17,
        "transition_model_cfg": {"value": 3},
        "cost_cfg": {"value": 1},
        "constraint_cfg": {"value": 2},
        "hybrid_cost_constraint_cfg": {"value": 4},
        "convergence_cfg": {"value": 5},
    }
    original = deepcopy(payload)
    cfg = RobotRolloutCfg.create_with_component_types(
        payload, robot, device, _TransitionCfg, _ManagerCfg
    )
    assert payload == original
    assert cfg.transition_model_cfg == _TransitionCfg(robot, device, 3)
    assert [item.value for item in cfg.get_cost_manager_configs()] == [1, 2, 4, 5]
    assert [item.value for item in cfg.get_cost_manager_configs(False)] == [1, 4, 5]
    assert cfg.sum_horizon and cfg.sampler_seed == 17


def test_direct_config_rejects_component_type_and_scalar_contract_mismatches():
    with pytest.raises(TypeError, match="transition_model_config_instance_type"):
        RobotRolloutCfg(DeviceCfg(), transition_model_config_instance_type=_TransitionCfg,
                         transition_model_cfg=object())
    with pytest.raises(TypeError, match="cost_manager_config_instance_type"):
        RobotRolloutCfg(DeviceCfg(), cost_manager_config_instance_type=_ManagerCfg,
                         cost_cfg=RobotCostManagerCfg())
    with pytest.raises(ValueError, match="nonnegative"):
        RobotRolloutCfg(DeviceCfg(), sampler_seed=-1)
    with pytest.raises(TypeError, match="sum_horizon"):
        RobotRolloutCfg(DeviceCfg(), sum_horizon=1)


def test_generic_object_sentinel_preserves_precompiled_mappings_for_solver_assembly():
    transition = {"portable": True}
    cost = {"weight": 1.0}
    cfg = RobotRolloutCfg.create_with_component_types(
        {"transition_model_cfg": transition, "cost_cfg": cost}, object(), DeviceCfg(),
        transition_model_config_instance_type=object,
        cost_manager_config_instance_type=object,
    )
    assert cfg.transition_model_cfg == transition and cfg.transition_model_cfg is not transition
    assert cfg.cost_cfg == cost and cfg.cost_cfg is not cost
    assert cfg.get_cost_manager_configs() == [cfg.cost_cfg]


def test_factory_rejects_non_mapping_and_component_without_create():
    with pytest.raises(TypeError, match="data_dict"):
        RobotRolloutCfg.create_with_component_types([], object())
    with pytest.raises(TypeError, match="callable create"):
        RobotRolloutCfg.create_with_component_types(
            {"cost_cfg": {}}, object(), cost_manager_config_instance_type=str
        )

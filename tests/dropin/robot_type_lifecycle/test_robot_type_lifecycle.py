"""Lifecycle coverage for the pinned ``curobo._src.types.robot`` facade."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from curobo._src.robot.dynamics import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg


_FIXTURE = (
    Path(__file__).parents[2] / "compat" / "types_config" / "fixtures" / "tiny_robot.yml"
)


def _mapping() -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": str(_FIXTURE.with_name("tiny_tree.urdf")),
                "base_link": "base",
                "tool_frames": ["tip"],
                "cspace": {"joint_names": ["shoulder"], "default_joint_position": [0.2]},
            },
            "load_dynamics": True,
        }
    }


def test_mapping_factory_has_typed_dynamics_and_does_not_mutate_input(tmp_path: Path) -> None:
    source = _mapping()
    original = deepcopy(source)
    cfg = RobotCfg.create(source, load_collision_spheres=False, num_envs=2)

    assert source == original
    assert cfg.cspace.joint_names == ["shoulder"]
    assert cfg.kinematics.collision_spheres == []
    assert isinstance(cfg.dynamics, DynamicsCfg)
    assert (
        cfg.dynamics.kinematics_config.robot_cfg
        is cfg.kinematics.kinematics_config.robot_cfg
    )
    assert cfg.dynamics.get_gravity().device.type == "cpu"

    target = tmp_path / "portable_robot.yml"
    cfg.write_config(target)
    assert RobotCfg.create(target).cspace.joint_names == ["shoulder"]


def test_precompiled_kinematics_mapping_preserves_typed_ownership() -> None:
    device_cfg = DeviceCfg()
    kin = KinematicsCfg.from_robot_yaml_file(str(_FIXTURE), device_cfg=device_cfg)
    cfg = RobotCfg.create({"kinematics": kin, "load_dynamics": True}, device_cfg=device_cfg)

    assert cfg.kinematics is not kin
    assert isinstance(cfg.dynamics, DynamicsCfg)
    assert cfg.dynamics.kinematics_config.robot_cfg is cfg.kinematics.kinematics_config.robot_cfg
    assert cfg.cspace.joint_names == ["shoulder"]


def test_direct_model_clone_and_factory_identity_are_safe() -> None:
    cfg = RobotCfg.from_basic(_FIXTURE.with_name("tiny_tree.urdf"), "base", ["tip"])
    clone = cfg.clone()

    assert RobotCfg.create(cfg) is cfg
    assert clone is not cfg
    assert clone.kinematics is not cfg.kinematics
    clone.kinematics.tool_frames[:] = ["arm"]
    assert cfg.kinematics.tool_frames == ["tip"]
    assert clone.cspace is clone.kinematics.cspace


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_dynamics_factory_keeps_typed_device_placement() -> None:
    device_cfg = DeviceCfg("mps")
    cfg = RobotCfg.from_basic(
        _FIXTURE.with_name("tiny_tree.urdf"), "base", ["tip"],
        device_cfg=device_cfg, load_dynamics=True,
    )
    assert isinstance(cfg.dynamics, DynamicsCfg)
    assert cfg.dynamics.get_gravity().device.type == "mps"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tool_frames": []}, "tool_frames"),
        ({"base_link": ""}, "base_link"),
        ({"load_dynamics": 1}, "load_dynamics"),
    ],
)
def test_basic_factory_rejects_malformed_value_inputs(kwargs: dict, message: str) -> None:
    values = {
        "urdf_path": _FIXTURE.with_name("tiny_tree.urdf"),
        "base_link": "base",
        "tool_frames": ["tip"],
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        RobotCfg.from_basic(**values)

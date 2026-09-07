"""Portable compilation checks for the pinned retargeter configuration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.types.device_cfg import DeviceCfg


def _criteria():
    return OrderedDict(
        [
            ("tool_a", replace(ToolPoseCriteria.track_position(), device_cfg=DeviceCfg("cpu"), project_distance_to_goal=False)),
            ("tool_b", replace(ToolPoseCriteria.track_orientation(), device_cfg=DeviceCfg("cpu"), project_distance_to_goal=False)),
        ]
    )


def _config(**overrides):
    values = dict(
        robot="franka.yml",
        tool_pose_criteria=_criteria(),
        self_collision_check=False,
        num_envs=3,
    )
    values.update(overrides)
    return MotionRetargeterCfg.create(**values)


def test_config_compiles_criteria_to_declared_device_without_mutating_input():
    criteria = _criteria()
    optimizer = {"solver": {"name": "portable"}}
    cfg = _config(
        tool_pose_criteria=criteria,
        device_cfg={"device": "cpu", "dtype": "float64"},
        ik_optimizer_configs=[optimizer],
    )

    assert cfg.tool_frames == ["tool_a", "tool_b"]
    assert cfg.batch_shape == (3, 2)
    assert cfg.device_cfg == DeviceCfg(torch.device("cpu"), torch.float64)
    assert cfg.tool_pose_criteria["tool_a"] is not criteria["tool_a"]
    assert cfg.tool_pose_criteria["tool_a"].terminal_pose_axes_weight_factor.dtype == torch.float64
    assert criteria["tool_a"].terminal_pose_axes_weight_factor.dtype == torch.float32
    cfg.ik_optimizer_configs[0]["compiled"] = True
    assert cfg.ik_optimizer_configs[0] is not optimizer
    assert "compiled" not in optimizer


def test_config_rejects_invalid_solver_boundaries_early():
    with pytest.raises(ValueError, match="nonempty list"):
        _config(ik_optimizer_configs=[])
    with pytest.raises(TypeError, match="must be a bool"):
        _config(use_mpc=1)
    with pytest.raises(ValueError, match="cold_start"):
        _config(mpc_warm_start_num_iters=4, mpc_cold_start_num_iters=3)
    with pytest.raises(ValueError, match="link names"):
        _config(tool_pose_criteria={"": ToolPoseCriteria.track_position()})
    with pytest.raises(TypeError, match="robot must"):
        _config(robot=object())


def test_with_device_recompiles_without_aliasing_criteria_or_optimizer_lists():
    cfg = _config(ik_optimizer_configs=["ik/custom.yml"])
    other = cfg.with_device(DeviceCfg(torch.device("cpu"), torch.float64))

    assert other is not cfg
    assert other.device_cfg.dtype == torch.float64
    assert other.tool_frames == cfg.tool_frames
    assert other.tool_pose_criteria["tool_a"] is not cfg.tool_pose_criteria["tool_a"]
    assert other.tool_pose_criteria["tool_a"].terminal_pose_axes_weight_factor.dtype == torch.float64
    other.ik_optimizer_configs.append("ik/second.yml")
    assert cfg.ik_optimizer_configs == ["ik/custom.yml"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_criteria_compilation_is_device_resident_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    source = _criteria()
    cfg = _config(tool_pose_criteria=source, device_cfg="mps")

    assert cfg.device_cfg.device.type == "mps"
    assert all(
        value.terminal_pose_axes_weight_factor.device.type == "mps"
        and value.project_distance_to_goal.device.type == "mps"
        for value in cfg.tool_pose_criteria.values()
    )
    assert all(
        value.terminal_pose_axes_weight_factor.device.type == "cpu"
        for value in source.values()
    )

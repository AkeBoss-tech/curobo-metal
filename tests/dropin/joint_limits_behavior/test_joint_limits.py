"""Behavioral coverage for the portable named JointLimits record."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.types import JointLimits
from curobo._src.types.device_cfg import DeviceCfg


def _limits(device_cfg: DeviceCfg = DeviceCfg(), names: list[str] | None = None) -> JointLimits:
    names = ["shoulder", "elbow"] if names is None else names
    dof = len(names)
    base = torch.arange(1, dof + 1, dtype=torch.float32)
    pair = torch.stack((-base, base))
    return JointLimits(names, pair, pair * 2, pair * 3, pair * 4, pair * 5, device_cfg)


def test_data_device_validation_and_clone_are_explicit() -> None:
    cfg = DeviceCfg(torch.device("cpu"), dtype=torch.float64)
    value = JointLimits.from_data_dict(
        {
            "joint_names": ["a", "b"], "position": [[-1, -2], [1, 2]],
            "velocity": [[-3, -4], [3, 4]], "acceleration": [[-5, -6], [5, 6]],
            "jerk": [[-7, -8], [7, 8]], "effort": [[-9, -10], [9, 10]],
        }, cfg,
    )
    assert value.device_cfg == cfg
    assert value.position.dtype == torch.float64
    assert value.position.device.type == "cpu"
    clone = value.clone()
    clone.position[0, 0] = -0.5
    assert value.position[0, 0].item() == -1.0
    with pytest.raises(ValueError, match="joint_names must be unique"):
        _limits(names=["a", "a"])
    with pytest.raises(ValueError, match="must not contain NaN"):
        JointLimits(["a"], torch.tensor([[float("nan")], [1.0]]), *[torch.tensor([[-1.0], [1.0]])] * 3)
    with pytest.raises(KeyError, match="velocity"):
        JointLimits.from_data_dict({"joint_names": ["a"], "position": [[-1.0], [1.0]]})


def test_copy_reindex_and_merge_preserve_named_limit_safety() -> None:
    value = _limits()
    position_buffer = value.position
    source = _limits(names=["elbow", "shoulder"])
    source.position = source.position[:, [1, 0]].contiguous()
    source.velocity = source.velocity[:, [1, 0]].contiguous()
    source.acceleration = source.acceleration[:, [1, 0]].contiguous()
    source.jerk = source.jerk[:, [1, 0]].contiguous()
    source.effort = source.effort[:, [1, 0]].contiguous()
    assert value.copy_(source) is value
    assert value.position is position_buffer
    assert value.joint_names == ["elbow", "shoulder"]
    selected = value.reindex(["shoulder"])
    assert selected.joint_names == ["shoulder"]
    assert selected.position.shape == (2, 1)
    previous_names = value.joint_names.copy()
    previous_position = value.position.clone()
    with pytest.raises(ValueError, match="unknown"):
        value.inplace_reindex(["elbow", "missing"])
    assert value.joint_names == previous_names
    torch.testing.assert_close(value.position, previous_position)
    arm = _limits(names=["arm"])
    gripper = _limits(names=["gripper"])
    combined = arm.merge(gripper)
    assert combined.joint_names == ["arm", "gripper"]
    torch.testing.assert_close(combined.position, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
    conflicting = _limits(names=["arm"])
    conflicting.position = torch.tensor([[-2.0], [2.0]])
    with pytest.raises(ValueError, match="conflicting"):
        arm.merge(conflicting)
    replaced = arm.merge(conflicting, overwrite=True)
    torch.testing.assert_close(replaced.position, conflicting.position)


def test_optional_effort_copy_and_mixed_effort_merge_boundary() -> None:
    with_effort = _limits(names=["a"])
    without_effort = JointLimits(["a"], with_effort.position, with_effort.velocity, with_effort.acceleration, with_effort.jerk)
    target_effort = with_effort.effort
    assert with_effort.copy_(without_effort) is with_effort
    assert with_effort.effort is target_effort
    empty_effort = JointLimits(["b"], without_effort.position, without_effort.velocity, without_effort.acceleration, without_effort.jerk)
    with pytest.raises(ValueError, match="effort is defined for only some"):
        with_effort.merge(empty_effort)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_joint_limit_lifecycle_stays_on_mps_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = DeviceCfg(torch.device("mps"))
    value = _limits(cfg)
    assert value.position.device.type == "mps"
    selected = value.reindex(["elbow"])
    assert selected.velocity.device.type == "mps"
    merged = selected.merge(value.reindex(["shoulder"]))
    assert merged.position.device.type == "mps"
    copied = value.clone()
    old_buffer = copied.jerk
    copied.copy_(value)
    assert copied.jerk is old_buffer
    assert copied.to(DeviceCfg(torch.device("cpu"))).position.device.type == "cpu"

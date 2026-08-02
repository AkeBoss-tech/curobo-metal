"""Behavioral coverage for the portable CSpaceParams record."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.types import CSpaceParams, JointLimits
from curobo._src.types.device_cfg import DeviceCfg


def _limits(device_cfg: DeviceCfg = DeviceCfg()) -> JointLimits:
    tensor = device_cfg.to_device
    return JointLimits(
        ["a", "b", "c"],
        tensor([[-2.0, -3.0, -4.0], [2.0, 3.0, 4.0]]),
        tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]]),
        tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]),
        tensor([[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]),
        device_cfg=device_cfg,
    )


def test_scalar_expansion_and_vector_contract_are_per_dof() -> None:
    cfg = CSpaceParams(
        ["a", "b", "c"],
        default_joint_position=[0.0, 1.0, 2.0],
        cspace_distance_weight=[1.0, 2.0, 3.0],
        null_space_weight=[0.0, 1.0, 2.0],
        max_acceleration=3.0,
        max_jerk=40.0,
    )
    assert cfg.max_acceleration.shape == cfg.max_jerk.shape == (3,)
    torch.testing.assert_close(cfg.max_acceleration, torch.full((3,), 3.0))
    torch.testing.assert_close(cfg.null_space_maximum_distance, torch.full((3,), 0.1))

    with pytest.raises(ValueError, match="batch/horizon"):
        CSpaceParams(["a", "b"], default_joint_position=[[0.0, 1.0]])
    with pytest.raises(ValueError, match="joint_names must be unique"):
        CSpaceParams(["a", "a"])
    with pytest.raises(ValueError, match="max_jerk must be strictly positive"):
        CSpaceParams(["a"], max_jerk=0.0)
    with pytest.raises(ValueError, match="velocity_scale must be non-negative"):
        CSpaceParams(["a"], velocity_scale=-0.1)


def test_reindex_supports_reduction_and_is_atomic_on_invalid_names() -> None:
    cfg = CSpaceParams(
        ["a", "b", "c"], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
        max_acceleration=[2.0, 3.0, 4.0], position_limit_clip=[0.1, 0.2, 0.3],
    )
    cfg.inplace_reindex(["c", "a"])
    assert cfg.joint_names == ["c", "a"]
    torch.testing.assert_close(cfg.default_joint_position, torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(cfg.position_limit_clip, torch.tensor([0.3, 0.1]))

    old_names = cfg.joint_names.copy()
    old_value = cfg.max_acceleration.clone()
    with pytest.raises(ValueError, match="unknown"):
        cfg.inplace_reindex(["a", "missing"])
    assert cfg.joint_names == old_names
    torch.testing.assert_close(cfg.max_acceleration, old_value)


def test_copy_clone_and_limit_scaling_preserve_buffer_and_input_ownership() -> None:
    cfg = CSpaceParams(["a", "b", "c"], position_limit_clip=[0.2, 0.1, 0.5])
    clone = cfg.clone()
    target_buffer = cfg.max_acceleration
    clone.max_acceleration.add_(7.0)
    assert cfg.copy_(clone) is cfg
    assert cfg.max_acceleration is target_buffer
    torch.testing.assert_close(cfg.max_acceleration, clone.max_acceleration)

    limits = _limits()
    scaled = cfg.scale_joint_limits(limits)
    torch.testing.assert_close(limits.position[0], torch.tensor([-2.0, -3.0, -4.0]))
    torch.testing.assert_close(scaled.position[0], torch.tensor([-1.8, -2.9, -3.5]))
    torch.testing.assert_close(scaled.position[1], torch.tensor([1.8, 2.9, 3.5]))
    with pytest.raises(ValueError, match="match CSpaceParams"):
        CSpaceParams(["b", "a", "c"]).scale_joint_limits(limits)
    with pytest.raises(ValueError, match="collapses"):
        CSpaceParams(["a", "b", "c"], position_limit_clip=4.0).scale_joint_limits(limits)


def test_load_from_joint_limits_validates_and_centers() -> None:
    cfg = CSpaceParams.load_from_joint_limits(
        torch.tensor([2.0, 4.0]), torch.tensor([-2.0, 0.0]), ["a", "b"]
    )
    torch.testing.assert_close(cfg.default_joint_position, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(cfg.null_space_maximum_distance, torch.ones(2))
    with pytest.raises(ValueError, match="rank-1"):
        CSpaceParams.load_from_joint_limits(torch.ones(1, 2), torch.zeros(2), ["a", "b"])
    with pytest.raises(ValueError, match="lower limits"):
        CSpaceParams.load_from_joint_limits(torch.tensor([0.0]), torch.tensor([0.0]), ["a"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_cspace_parameters_and_scaling_stay_on_mps_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"))
    cfg = CSpaceParams(
        ["a", "b", "c"], default_joint_position=[0.0, 0.0, 0.0],
        max_acceleration=2.0, position_limit_clip=[0.1, 0.2, 0.3], device_cfg=device_cfg,
    )
    result = cfg.scale_joint_limits(_limits(device_cfg))
    assert cfg.default_joint_position.device.type == "mps"
    assert result.position.device.type == "mps"
    cfg.inplace_reindex(["c", "a"])
    assert cfg.max_jerk.device.type == "mps"

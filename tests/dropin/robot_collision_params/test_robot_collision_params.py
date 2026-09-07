"""Portable lifecycle coverage for robot collision parameters and limits."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.types import JointLimits, SelfCollisionKinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg


def _limits(device_cfg: DeviceCfg = DeviceCfg("cpu")) -> JointLimits:
    pair = torch.tensor([[-2.0, -1.0, -0.5], [2.0, 1.0, 0.5]])
    return JointLimits(["a", "b", "c"], pair, pair * 2, pair * 3, pair * 4, pair * 5, device_cfg)


def _collision_cfg(device: torch.device = torch.device("cpu")) -> SelfCollisionKinematicsCfg:
    return SelfCollisionKinematicsCfg(
        num_spheres=4,
        sphere_padding=torch.tensor([0.0, 0.1, 0.2, 0.3], device=device),
        collision_pairs=torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]], device=device),
    )


def test_collision_config_clone_copy_mask_and_reindex_are_device_local() -> None:
    value = _collision_cfg()
    assert value.num_collision_checks == 4
    assert not value.is_empty
    mask = value.pair_mask()
    assert mask.dtype == torch.bool and mask.device == value.device
    assert mask[0, 1] and mask[1, 0] and not mask[0, 0]

    clone = value.clone()
    clone.sphere_padding[1] = 0.9
    assert value.sphere_padding[1].item() == pytest.approx(0.1)
    pair_buffer = value.collision_pairs
    assert value.copy_(clone) is value
    assert value.collision_pairs is pair_buffer
    assert value.sphere_padding[1].item() == pytest.approx(0.9)

    reduced = value.reindex_spheres([3, 0, 2])
    assert reduced.num_spheres == 3
    assert reduced.collision_pairs.tolist() == [[1, 2], [0, 2]]
    torch.testing.assert_close(reduced.sphere_padding, torch.tensor([0.3, 0.0, 0.2]))
    with pytest.raises(ValueError, match="duplicates"):
        value.reindex_spheres([0, 0])
    with pytest.raises(ValueError, match="out-of-range"):
        value.reindex_spheres([4])


def test_collision_config_to_and_shape_changing_copy_are_explicit() -> None:
    value = _collision_cfg()
    converted = value.to(DeviceCfg(torch.device("cpu"), dtype=torch.float64))
    assert converted.sphere_padding.dtype == torch.float64
    assert converted.collision_pairs.dtype == value.collision_pairs.dtype
    assert converted.collision_pairs.data_ptr() != value.collision_pairs.data_ptr()

    target = _collision_cfg()
    old_pairs = target.collision_pairs
    assert target.copy_(value.reindex_spheres([0, 1])) is target
    assert target.num_spheres == 2
    assert target.collision_pairs.shape == (1, 2)
    assert target.collision_pairs is not old_pairs
    with pytest.raises(TypeError, match="SelfCollisionKinematicsCfg"):
        target.copy_(object())  # type: ignore[arg-type]


def test_joint_limits_safe_index_copy_and_query_lifecycle() -> None:
    value = _limits(device_cfg=DeviceCfg("cpu"))
    buffer = value.position
    reordered = value.reindex(["c", "a", "b"])
    assert value.copy_(reordered) is value
    assert value.position is buffer
    assert value.joint_names == ["c", "a", "b"]

    value.inplace_reindex(["a", "b"])
    assert value.dof == 2
    assert value.position.shape == (2, 2)
    assert value.joint_names == ["a", "b"]
    torch.testing.assert_close(value.lower_limits("velocity"), torch.tensor([-4.0, -2.0]))
    torch.testing.assert_close(value.velocity_upper_limits, torch.tensor([4.0, 2.0]))
    with pytest.raises(ValueError, match="unknown limit_type"):
        value.lower_limits("temperature")

    data = value.as_dict(clone=True)
    assert data["joint_names"] == ["a", "b"]
    assert data["position"] is not value.position
    position = torch.tensor([[-3.0, 0.25], [0.25, 3.0]])
    torch.testing.assert_close(value.clamp_position(position), torch.tensor([[-2.0, 0.25], [0.25, 1.0]]))
    with pytest.raises(ValueError, match="configured joint values"):
        value.clamp_position(torch.zeros(3))


def test_joint_limit_margin_is_validated_and_differentiable() -> None:
    value = _limits(device_cfg=DeviceCfg("cpu"))
    margin = torch.tensor([0.1, 0.2, 0.1], requires_grad=True)
    shrunk = value.with_position_margin(margin)
    torch.testing.assert_close(shrunk.position, torch.tensor([[-1.9, -0.8, -0.4], [1.9, 0.8, 0.4]]))
    shrunk.position.square().sum().backward()
    assert margin.grad is not None and torch.isfinite(margin.grad).all()
    with pytest.raises(ValueError, match="collapses"):
        value.with_position_margin([2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        value.with_position_margin(-0.1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_portable_parameter_lifecycle_runs_on_mps_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = torch.device("mps")
    cfg = _collision_cfg(device).reindex_spheres(torch.tensor([3, 1, 0], device=device))
    assert cfg.pair_mask().device.type == "mps"
    limits = _limits(DeviceCfg(device))
    narrowed = limits.inplace_reindex(["b", "a"])
    assert narrowed.position.device.type == "mps"
    result = narrowed.with_position_margin(0.1)
    assert result.clamp_position(torch.tensor([[4.0, -4.0]], device=device)).device.type == "mps"

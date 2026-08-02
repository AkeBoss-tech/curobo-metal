"""Portable lifecycle coverage for RobotCollisionGeometry."""

import pytest
import torch

from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo.types import DeviceCfg


def _geometry() -> RobotCollisionGeometry:
    return RobotCollisionGeometry(torch.tensor([2, 0, 2, 1], dtype=torch.int32), 3)


def test_topology_is_unbatched_and_validated() -> None:
    geometry = _geometry()
    assert geometry.num_spheres == 4
    assert geometry.num_links == 3
    assert geometry.device.type == "cpu"
    assert geometry.dtype == torch.int32
    torch.testing.assert_close(geometry.link_mask(2), torch.tensor([True, False, True, False]))

    with pytest.raises(ValueError, match="shape"):
        RobotCollisionGeometry(torch.zeros((1, 2), dtype=torch.int32), 2)
    with pytest.raises(TypeError, match="integer"):
        RobotCollisionGeometry(torch.zeros(2), 2)
    with pytest.raises(ValueError, match="out-of-range"):
        RobotCollisionGeometry(torch.tensor([0, 2], dtype=torch.int32), 2)
    with pytest.raises(ValueError, match="out of range"):
        geometry.link_mask(3)


def test_clone_detach_copy_and_contiguous_preserve_ownership() -> None:
    geometry = _geometry()
    clone = geometry.clone()
    assert clone.link_sphere_idx_map.data_ptr() != geometry.link_sphere_idx_map.data_ptr()
    clone.link_sphere_idx_map[0] = 1
    assert geometry.link_sphere_idx_map[0].item() == 2
    assert geometry.copy_(clone) is None
    assert geometry.link_sphere_idx_map[0].item() == 1
    assert not geometry.detach().link_sphere_idx_map.requires_grad

    sliced = RobotCollisionGeometry(torch.tensor([0, 1, 2, 2], dtype=torch.int64)[::2], 3)
    assert not sliced.link_sphere_idx_map.is_contiguous()
    assert sliced.contiguous().link_sphere_idx_map.is_contiguous()
    assert geometry.contiguous() is geometry

    with pytest.raises(ValueError, match="matching num_links"):
        geometry.copy_(RobotCollisionGeometry(torch.tensor([0, 1, 0, 1]), 2))
    with pytest.raises(ValueError, match="matching.*shapes"):
        geometry.copy_(RobotCollisionGeometry(torch.tensor([0, 1]), 3))


def test_device_moves_never_cast_the_integer_index_map() -> None:
    geometry = _geometry()
    assert geometry.to() is geometry
    copied = geometry.to(copy=True)
    assert copied is not geometry
    assert copied.dtype == torch.int32
    assert copied.link_sphere_idx_map.data_ptr() != geometry.link_sphere_idx_map.data_ptr()
    converted = geometry.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert converted.dtype == torch.int32
    assert converted.device.type == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_link_masks_and_cpu_round_trip_need_no_fallback() -> None:
    geometry = _geometry().to("mps")
    assert geometry.device.type == "mps"
    assert geometry.dtype == torch.int32
    assert geometry.link_mask(2).device.type == "mps"
    torch.testing.assert_close(
        geometry.to("cpu").link_sphere_idx_map,
        torch.tensor([2, 0, 2, 1], dtype=torch.int32),
    )

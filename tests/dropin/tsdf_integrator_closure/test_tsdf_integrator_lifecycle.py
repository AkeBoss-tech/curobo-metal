"""Portable BlockSparseTSDFIntegrator update-boundary regression coverage."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _observation(device: str = "cpu", cameras: int = 1, depth: float = 1.0) -> CameraObservation:
    values = torch.full((cameras, 4, 4), depth, device=device)
    if cameras == 1:
        values = values[0]
    intrinsics = torch.tensor(
        ((4.0, 0.0, 1.5), (0.0, 4.0, 1.5), (0.0, 0.0, 1.0)), device=device
    )
    if cameras > 1:
        intrinsics = intrinsics.expand(cameras, -1, -1).clone()
        pose = Pose.from_batch_list([[0, 0, 0, 1, 0, 0, 0]] * cameras).to(device=device)
    else:
        pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device)
    return CameraObservation(depth_image=values, intrinsics=intrinsics, pose=pose, depth_to_meter=1.0)


def _integrator(device: str = "cpu", cameras: int = 1, **kwargs) -> BlockSparseTSDFIntegrator:
    return BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
        grid_shape=(4, 4, 4), voxel_size=0.1, origin=torch.tensor((-0.2, -0.2, 0.8)),
        device=device, num_cameras=cameras, **kwargs,
    ))


def test_batched_camera_update_matches_configured_camera_axis_and_advances_one_frame():
    integrator = _integrator(cameras=2, image_height=4, image_width=4)
    integrator.integrate(_observation(cameras=2))
    state = integrator.mapper._mapper.state
    assert integrator.get_stats()["frame_count"] == 1
    assert state.weight.device.type == "cpu"
    assert state.generation.tolist() == [1]
    assert state.weight.gt(0).any()


def test_invalid_later_frame_does_not_decay_or_mutate_existing_map_state():
    integrator = _integrator(time_decay=0.5, minimum_tsdf_weight=0.01)
    integrator.integrate(_observation())
    before = integrator.mapper._mapper.state
    saved = {name: getattr(before, name).clone() for name in ("tsdf", "weight", "occupancy", "generation")}

    with pytest.raises(ValueError, match="num_cameras"):
        integrator.integrate(_observation(cameras=2))

    after = integrator.mapper._mapper.state
    assert integrator.get_stats()["frame_count"] == 1
    for name, value in saved.items():
        torch.testing.assert_close(getattr(after, name), value)


def test_invalid_depths_are_data_masks_and_do_not_create_observed_voxels():
    integrator = _integrator()
    observation = _observation(depth=0.0)
    observation.depth_image[0, 0] = float("nan")
    integrator.integrate(observation)
    state = integrator.mapper._mapper.state
    assert state.weight.eq(0).all()
    assert state.generation.eq(0).all()
    assert integrator.get_stats()["frame_count"] == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_batched_mps_camera_update_never_uses_cpu_fallback():
    integrator = _integrator("mps", cameras=2, image_height=4, image_width=4)
    integrator.integrate(_observation("mps", cameras=2))
    state = integrator.mapper._mapper.state
    assert state.tsdf.device.type == state.weight.device.type == "mps"
    # Cross-device frames are rejected instead of hiding an input copy.
    with pytest.raises(ValueError, match="map device"):
        integrator.integrate(_observation("cpu", cameras=2))

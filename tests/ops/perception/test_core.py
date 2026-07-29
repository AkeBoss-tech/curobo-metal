import numpy as np
import pytest
import torch

from curobo_metal.ops.perception import (
    CameraObservation, PerceptionConfig, PerceptionMapper, integrate_depth,
)
from curobo_metal.reference.perception import (
    CameraObservation as RefCamera,
    PerceptionConfig as RefConfig,
    empty_state as ref_empty,
    integrate as ref_integrate,
)


def _obs(device="cpu", dtype=torch.float32, depth=None, cameras=1):
    if depth is None:
        depth = torch.ones((cameras, 3, 3), device=device, dtype=dtype)
    intrinsics = torch.tensor([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]],
                              device=device, dtype=dtype).expand(cameras, -1, -1).clone()
    poses = torch.eye(4, device=device, dtype=dtype).expand(cameras, -1, -1).clone()
    return CameraObservation(depth, intrinsics, poses)


def _cfg(environments=1):
    return PerceptionConfig((3, 3, 5), 0.25, grid_center=(0, 0, 0.5),
                            truncation_distance=0.25, depth_min=0.1,
                            depth_max=2, environments=environments)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numpy_oracle_parity(dtype):
    cfg = _cfg()
    mapper = PerceptionMapper(cfg, dtype=dtype)
    actual = mapper.update(_obs(dtype=dtype))
    rcfg = RefConfig(cfg.shape, cfg.voxel_size, cfg.grid_center,
                     cfg.truncation_distance, cfg.depth_min, cfg.depth_max,
                     cfg.max_weight, cfg.occupancy_threshold, cfg.unobserved_esdf)
    robs = RefCamera(np.ones((3, 3)), np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]]),
                     np.eye(4))
    expected = ref_integrate(rcfg, ref_empty(rcfg), robs)
    np.testing.assert_allclose(actual.tsdf[0].detach().numpy(), expected.tsdf, atol=1e-6)
    np.testing.assert_array_equal(actual.occupancy[0].numpy(), expected.occupancy)
    np.testing.assert_allclose(actual.esdf[0].numpy(), expected.esdf, atol=1e-6)


def test_depth_is_differentiable_away_from_occupancy_transitions():
    cfg = _cfg()
    mapper = PerceptionMapper(cfg)
    depth = torch.ones((1, 3, 3), requires_grad=True)
    state = integrate_depth(cfg, mapper.state, _obs(depth=depth))
    state.tsdf.sum().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert depth.grad.abs().sum() > 0


def test_multi_camera_batched_environments_partial_update_reset_and_query():
    cfg = _cfg(environments=2)
    mapper = PerceptionMapper(cfg)
    depth = torch.stack((torch.ones((2, 3, 3)), torch.full((2, 3, 3), 1.25)))
    obs = _obs(depth=depth, cameras=2)
    state = mapper.update(obs)
    assert state.generation.tolist() == [1, 1]
    assert not torch.equal(state.tsdf[0], state.tsdf[1])
    mapper.reset(torch.tensor([1], dtype=torch.int64))
    assert mapper.state.generation.tolist() == [1, 0]
    assert mapper.state.weight[1].count_nonzero() == 0
    points = torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]])
    result = mapper.query(points, env_indices=torch.tensor([0, 1]))
    assert result.valid.all()
    assert result.distance[0, 0] < 0
    assert result.distance[1, 0] > 0
    mapper.update(_obs(), env_indices=torch.tensor([1]))
    assert mapper.state.generation.tolist() == [1, 1]


def test_transformed_camera_matches_world_plane():
    cfg = PerceptionConfig((3, 3, 5), 0.25, grid_center=(1, 0, 0.5),
                           truncation_distance=.25, depth_min=.1, depth_max=2)
    mapper = PerceptionMapper(cfg)
    obs = _obs()
    pose = obs.camera_to_world.clone()
    pose[..., 0, 3] = 1
    state = mapper.update(CameraObservation(obs.depth, obs.intrinsics, pose))
    assert state.occupancy.any()
    occupied_centers = torch.nonzero(state.occupancy[0], as_tuple=False)
    assert set(occupied_centers[:, 0].tolist()) == {0, 1, 2}


def test_rejected_indexed_update_does_not_mutate_state():
    mapper = PerceptionMapper(_cfg(environments=2))
    before = mapper.state
    with pytest.raises(ValueError, match="duplicates"):
        mapper.update(_obs(depth=torch.ones((2, 1, 3, 3))),
                      env_indices=torch.tensor([0, 0]))
    assert mapper.state is before


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_fallback_disabled_mps_pipeline(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    mapper = PerceptionMapper(_cfg(), device="mps")
    state = mapper.update(_obs(device="mps"))
    assert state.esdf.device.type == "mps"
    result = mapper.query(torch.tensor([[0., 0., 1.]], device="mps"))
    assert result.distance.device.type == "mps"

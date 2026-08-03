"""Portable dense mapper pose-refiner lifecycle coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)
from curobo._src.types.pose import Pose
from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper


def _native(*, device: str = "cpu", environments: int = 2) -> PerceptionMapper:
    return PerceptionMapper(PerceptionConfig(
        shape=(2, 2, 2), voxel_size=0.1, grid_center=(0.0, 0.0, 0.2),
        truncation_distance=0.04, environments=environments,
    ), device=device)


def _inputs(*, device: str = "cpu", batch: int = 2):
    depth = torch.ones(batch, 2, 2, device=device)
    intrinsics = torch.tensor(((8.0, 0.0, 0.5), (0.0, 8.0, 0.5), (0.0, 0.0, 1.0)), device=device)
    pose = torch.eye(4, device=device).expand(batch, -1, -1).clone()
    return depth, intrinsics, pose


def _refiner(native: PerceptionMapper) -> BlockSparseRaycastPoseRefiner:
    return BlockSparseRaycastPoseRefiner(native, BlockSparseRaycastRefinerCfg(
        max_iterations=2, minimum_valid_depth_pixels=4, depth_maximum_distance=2.0,
    ))


def test_refiner_routes_batched_depth_pose_and_environment_selection_with_state_lifecycle():
    refiner = _refiner(_native())
    depth, intrinsics, pose = _inputs()
    output, loss, iterations = refiner.refine_pose(
        depth, intrinsics, pose, env_indices=torch.tensor([1, 0], dtype=torch.int64)
    )

    assert isinstance(output, Pose)
    assert output.position.shape == (2, 3)
    assert loss.shape == iterations.shape == (2,)
    assert output.position.device.type == loss.device.type == "cpu"
    state = refiner.last_state
    assert state is not None
    assert state.environment_indices.tolist() == [1, 0]
    assert state.best_n_valid.tolist() == [4, 4]
    cloned = state.clone()
    cloned.depth.add_(1)
    assert not torch.equal(cloned.depth, state.depth)
    refiner.reset()
    assert refiner.last_state is None
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        refiner.reset_cuda_graph()


def test_refiner_preserves_single_tuple_contract_and_rejects_bad_batch_routing():
    refiner = _refiner(_native(environments=1))
    depth, intrinsics, pose = _inputs(batch=1)
    output, loss, iterations = refiner.refine_pose(depth[0], intrinsics, pose[0])
    assert isinstance(output, Pose) and isinstance(loss, float) and isinstance(iterations, int)
    with pytest.raises(ValueError, match="env_indices"):
        refiner.refine_pose(depth, intrinsics, pose, env_indices=torch.tensor([0, 0], dtype=torch.int64))
    with pytest.raises(ValueError, match="exceeds mapper environments"):
        refiner.refine_pose(torch.ones(2, 2, 2), intrinsics, torch.eye(4).expand(2, -1, -1))


def test_refiner_config_and_raw_raycast_boundaries_are_explicit():
    with pytest.raises(ValueError, match="distance_threshold"):
        BlockSparseRaycastRefinerCfg(distance_threshold=0.0)
    with pytest.raises(ValueError, match="lambda parameters"):
        BlockSparseRaycastRefinerCfg(lambda_min=2.0, lambda_max=1.0)
    with pytest.raises(ValueError, match="iterations"):
        BlockSparseRaycastRefinerCfg(iterations=True)

    native = _native(environments=1)
    refiner = BlockSparseRaycastPoseRefiner(native, BlockSparseRaycastRefinerCfg(
        n_samples_per_ray=2, max_iterations=1, minimum_valid_depth_pixels=4,
    ))
    depth, intrinsics, pose = _inputs(batch=1)
    with pytest.raises(NotImplementedError, match="n_samples_per_ray"):
        refiner.refine_pose(depth[0], intrinsics, pose[0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_refiner_batch_executes_on_mps_without_fallback():
    refiner = _refiner(_native(device="mps"))
    depth, intrinsics, pose = _inputs(device="mps")
    output, loss, iterations = refiner.refine_pose(depth, intrinsics, pose)
    assert output.position.device.type == loss.device.type == iterations.device.type == "mps"
    assert refiner.last_state.map_generation.device.type == "mps"

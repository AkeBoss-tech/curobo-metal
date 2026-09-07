from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def test_feature_grid_shape_contract_is_checked_at_configuration_time() -> None:
    common = dict(
        max_blocks=8,
        voxel_size=0.02,
        grid_shape=(32, 32, 32),
        truncation_distance=0.04,
        device="cpu",
    )
    with pytest.raises(ValueError, match="feature_dim > 0 requires"):
        BlockSparseTSDFIntegratorCfg(**common, feature_dim=3)
    with pytest.raises(ValueError, match="require feature_dim > 0"):
        BlockSparseTSDFIntegratorCfg(
            **common, feature_grid_height=2, feature_grid_width=2
        )


def test_mapper_forwards_visible_capacity_to_fixed_camera_scratch() -> None:
    mapper = Mapper(
        MapperCfg(
            extent_meters_xyz=(0.32, 0.32, 0.32),
            voxel_size=0.02,
            esdf_voxel_size=0.02,
            image_height=8,
            image_width=8,
            num_cameras=2,
            max_visible_blocks_per_integration=7,
            max_support_pixels_per_block_camera=5,
            feature_integration_kernel="grouped",
            device="cpu",
        )
    )
    camera = mapper.integrator._tsdf_integrator._camera_integrator
    assert mapper.integrator._tsdf_integrator.config.max_visible_blocks_per_integration == 7
    assert camera.pool_indices.shape == (7,)
    assert camera.support_counts.shape == (7, 2)
    assert camera.support_pixels.shape == (7, 2, 5)
    assert camera.use_tiled_feature_kernel is False
    assert camera.clear_pool_indices.shape == (mapper.tsdf.config.max_blocks,)


def test_feature_input_validation_precedes_sparse_map_mutation() -> None:
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=8,
            voxel_size=0.02,
            grid_shape=(512, 512, 512),
            truncation_distance=0.04,
            image_height=4,
            image_width=4,
            feature_dim=3,
            feature_grid_height=2,
            feature_grid_width=2,
            device="cpu",
        )
    )
    observation = CameraObservation(
        depth_image=torch.ones((1, 4, 4), dtype=torch.float32),
        rgb_image=torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
        intrinsics=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        pose=Pose(
            position=torch.zeros((1, 3), dtype=torch.float32),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
        feature_grid=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="dtype must be torch.float16"):
        integrator.integrate(observation)
    assert int(integrator.tsdf.data.num_allocated.item()) == 0

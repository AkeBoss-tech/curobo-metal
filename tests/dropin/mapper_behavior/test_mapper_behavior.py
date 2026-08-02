"""High-level Mapper lifecycle checks over the real dense CPU/MPS backend."""

import pytest
import torch

from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose


def _camera(device="cpu"):
    return CameraObservation(
        depth_image=torch.ones((4, 4), device=device),
        intrinsics=torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5),
                                 (0.0, 0.0, 1.0)), device=device),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).to(device=device),
        depth_to_meter=1.0,
    )


def _mapper(device="cpu"):
    return Mapper(MapperCfg(
        (0.4, 0.4, 0.4), voxel_size=0.1,
        grid_center=torch.tensor((0.0, 0.0, 0.4)),
        enable_static=True, device=device,
    ))


def test_mapper_validates_observation_aliases_without_mutating_state():
    mapper = _mapper()
    generation = mapper._mapper.state.generation.clone()
    camera = _camera()
    with pytest.raises(TypeError, match="at most one positional"):
        mapper.integrate(camera, camera)
    with pytest.raises(TypeError, match="cannot be combined"):
        mapper.integrate(camera, observation=camera)
    with pytest.raises(TypeError, match="requires a camera"):
        mapper.integrate()
    with pytest.raises(NotImplementedError, match="LiDAR"):
        mapper.integrate(lidar_observation=LidarObservation(range_image=torch.ones((2, 2))))
    assert torch.equal(mapper._mapper.state.generation, generation)


def test_mapper_cache_query_and_static_mutation_share_one_dense_lifecycle():
    mapper = _mapper()
    mapper.integrate(_camera())
    assert not mapper.is_esdf_current
    first = mapper.compute_esdf()
    assert mapper.is_esdf_current
    assert mapper.compute_esdf() is first
    assert mapper.get_stats()["esdf_compute_count"] == 1
    result = mapper.query(torch.tensor([[0.0, 0.0, 0.4]]))
    assert result.distance.shape == result.valid.shape == (1, 1)
    assert result.distance.device.type == mapper.device.type

    mapper.update_static_obstacles(SceneCfg(cuboid=[Cuboid(
        "box", pose=[0, 0, 0.4, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )]))
    assert not mapper.is_esdf_current
    assert mapper.get_voxel_grid().feature_tensor.device.type == mapper.device.type


def test_mapper_render_accepts_source_camera_matrix_batches():
    mapper = _mapper()
    mapper.update_static_obstacles(SceneCfg(cuboid=[Cuboid(
        "box", pose=[0, 0, 0.4, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1],
    )]))
    intrinsics = torch.tensor(((8.0, 0.0, 1.5), (0.0, 8.0, 1.5),
                               (0.0, 0.0, 1.0))).expand(2, -1, -1).clone()
    poses = torch.eye(4).expand(2, -1, -1).clone()
    depth, normals, valid = mapper.render(intrinsics, poses, (4, 4))
    assert depth.shape == valid.shape == (2, 4, 4)
    assert normals.shape == (2, 4, 4, 3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mapper_query_cache_and_batched_render_remain_on_mps():
    mapper = _mapper("mps")
    mapper.integrate(_camera("mps"))
    grid = mapper.compute_esdf()
    query = mapper.query(torch.tensor([[0.0, 0.0, 0.4]], device="mps"))
    intrinsics = torch.tensor((8.0, 8.0, 1.5, 1.5), device="mps").expand(2, -1)
    poses = torch.eye(4, device="mps").expand(2, -1, -1)
    depth, normals, valid = mapper.render(intrinsics, poses, (4, 4))
    assert grid.feature_tensor.device.type == query.distance.device.type == "mps"
    assert depth.device.type == normals.device.type == valid.device.type == "mps"

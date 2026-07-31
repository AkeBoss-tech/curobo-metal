import importlib
import json
from pathlib import Path

import pytest
import torch

from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.util.utils_coords import (
    voxel_to_world,
    world_to_voxel,
)
from curobo._src.perception.mapper.util.utils_quantization import (
    pack_site_coords,
    unpack_site_coords_torch,
)
from curobo._src.perception.optim_pose_lm import solve_lm_step
from curobo._src.perception.pose_estimation.util import (
    compute_pose_point_to_plane_cholesky,
    omega_to_quaternion,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


API_INVENTORY = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "api_compat"
    / "upstream-api.json"
)


def test_every_pinned_mapper_module_imports_without_warp():
    inventory = json.loads(API_INVENTORY.read_text())
    mapper_modules = (
        module["name"]
        for module in inventory["modules"]
        if module["name"].startswith("curobo._src.perception.mapper")
    )
    for module_name in mapper_modules:
        importlib.import_module(module_name)
    for name in (
        "optim_pose_lm",
        "pose_estimation.geometry",
        "pose_estimation.util",
        "pose_estimation.wp_mesh_sdf_alignment",
    ):
        importlib.import_module(f"curobo._src.perception.{name}")


def test_mapper_fuses_depth_esdf_checkpoint_and_mesh(tmp_path):
    mapper = Mapper(MapperCfg(
        extent_meters_xyz=(0.4, 0.4, 0.4),
        voxel_size=0.1,
        grid_center=torch.tensor((0.0, 0.0, 0.4)),
        truncation_distance=0.15,
        device="cpu",
    ))
    observation = CameraObservation(
        depth_image=torch.ones((8, 8)),
        intrinsics=torch.tensor(((8.0, 0.0, 3.5), (0.0, 8.0, 3.5), (0.0, 0.0, 1.0))),
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0]),
        depth_to_meter=1.0,
    )
    mapper.integrate(observation)
    assert mapper.get_stats()["observed_voxels"] > 0
    esdf = mapper.compute_esdf()
    assert esdf.feature_tensor.shape == (4, 4, 4)
    mesh = mapper.extract_mesh()
    assert mesh.vertices.shape[-1] == 3
    checkpoint = tmp_path / "map.pt"
    mapper.save_blocks(checkpoint)
    mapper.reset()
    mapper.import_blocks(checkpoint)
    assert mapper.get_stats()["observed_voxels"] > 0


def test_esdf_coordinate_quantization_and_lm_are_portable_and_differentiable():
    mapper = Mapper(MapperCfg((0.2, 0.2, 0.2), voxel_size=0.1, device="cpu"))
    mapper._mapper.state.occupancy[0, 0, 0, 0] = True
    values, gradients = BlockSparseESDFIntegrator(
        BlockSparseESDFIntegratorCfg(voxel_size=0.1)
    ).compute(mapper)
    assert values.shape == (1, 2, 2, 2)
    assert gradients.shape == (1, 2, 2, 2, 3)
    world = voxel_to_world(0, 0, 0, (0, 0, 0), (2, 2, 2), 0.1)
    assert world_to_voxel(*world, (0, 0, 0), (2, 2, 2), 0.1) == (0, 0, 0)
    packed = torch.tensor([pack_site_coords(3, 4, 5)])
    assert unpack_site_coords_torch(packed).tolist() == [[3, 4, 5]]
    jtj = torch.eye(6, requires_grad=True)
    step = solve_lm_step(jtj, torch.ones(6), torch.tensor(0.1), torch.eye(6))
    step.sum().backward()
    assert jtj.grad is not None


def test_pose_estimation_utilities():
    quaternion = omega_to_quaternion(torch.zeros(3))
    assert torch.allclose(quaternion, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    source = torch.tensor(((0.0, 0.0, 0.1), (1.0, 0.0, 0.1), (0.0, 1.0, 0.1)))
    target = source.clone()
    target[:, 2] = 0
    normals = torch.tensor(((0.0, 0.0, 1.0),) * 3)
    delta = compute_pose_point_to_plane_cholesky(source, target, normals)
    assert torch.isfinite(delta).all()


def test_raw_kernel_boundaries_are_precise():
    from curobo._src.perception.mapper.kernel.builder.builder_hash import make_hash_kernels

    with pytest.raises(NotImplementedError, match="raw Warp/CUDA"):
        make_hash_kernels(8)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mapper_mps_without_cpu_fallback():
    mapper = Mapper(MapperCfg((0.2, 0.2, 0.2), voxel_size=0.1, device="mps"))
    assert mapper._mapper.state.tsdf.device.type == "mps"

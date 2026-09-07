
from curobo.types import DeviceCfg
import torch
from types import SimpleNamespace

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mesh_extractor import extract_mesh_block_sparse
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def test_sparse_depth_fusion_extracts_curved_sphere_zero_crossing():
    """A rendered sphere must reconstruct as curved geometry, not block planes."""
    height = width = 64
    focal = 80.0
    u = torch.arange(width, dtype=torch.float32)
    v = torch.arange(height, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    x = (uu - (width - 1) / 2) / focal
    y = (vv - (height - 1) / 2) / focal
    # Intersect projective rays t[x,y,1] with a radius-0.2 sphere at z=1.
    a = x.square() + y.square() + 1.0
    discriminant = 4.0 - 4.0 * a * 0.96
    valid = discriminant >= 0
    depth = torch.where(valid, (2.0 - discriminant.clamp_min(0).sqrt()) / (2.0 * a), 0.0)
    intrinsics = torch.tensor(
        ((focal, 0.0, (width - 1) / 2), (0.0, focal, (height - 1) / 2), (0.0, 0.0, 1.0))
    )
    observation = CameraObservation(
        depth_image=depth,
        rgb_image=torch.full((height, width, 3), 180, dtype=torch.uint8),
        intrinsics=intrinsics,
        pose=Pose.from_list([0, 0, 0, 1, 0, 0, 0], device_cfg=DeviceCfg("cpu")),
        depth_to_meter=1.0,
    )
    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            # Exercise the genuinely sparse runtime rather than the bounded
            # dense small-volume optimization.
            grid_shape=(257, 257, 257),
            max_blocks=1000,
            block_size=4,
            voxel_size=0.025,
            # Sparse origins are map centers; signed block coordinates cover
            # both sides of the sphere around x=y=0.
            origin=torch.tensor([0.0, 0.0, 1.0]),
            truncation_distance=0.075,
            image_height=height,
            image_width=width,
            device="cpu",
        )
    )
    for _ in range(3):
        integrator.integrate(observation)

    vertices, normals, colors = extract_mesh_block_sparse(integrator.tsdf)
    assert len(vertices) > 0
    radial_error = (
        torch.linalg.vector_norm(vertices - vertices.new_tensor([0.0, 0.0, 1.0]), dim=-1)
        - 0.2
    ).abs()
    assert radial_error.median() < 0.04
    assert torch.quantile(radial_error, 0.9) < 0.08
    assert vertices[:, 0].max() - vertices[:, 0].min() > 0.2
    assert vertices[:, 1].max() - vertices[:, 1].min() > 0.2
    assert normals.shape == colors.shape == vertices.shape


def test_sparse_adjacent_blocks_weld_shared_plane_vertices_and_faces():
    """Cells crossing a block boundary produce one deterministic seam."""
    block_size = 2
    block_data = torch.zeros((2, block_size**3, 2), dtype=torch.float32)
    block_coords = torch.tensor((0, 0, 0, 1, 0, 0), dtype=torch.int32)
    # A plane crossing x=2.0: block 0 contributes x=1.5 and block 1 x=2.5.
    for block, block_x in enumerate((0, 1)):
        for linear in range(block_size**3):
            x = linear // 4 + block_x * block_size
            block_data[block, linear, 0] = float(x) - 1.5
            block_data[block, linear, 1] = 1.0
    data = SimpleNamespace(
        block_data=block_data,
        block_grid_rgb=torch.ones((2, 1, 4), dtype=torch.float32),
        block_coords=block_coords,
        block_size=block_size,
        origin=torch.zeros(3),
        voxel_size=1.0,
        num_allocated=torch.tensor(2, dtype=torch.int32),
    )
    tsdf = SimpleNamespace(data=data, _portable_sparse=True)

    first = extract_mesh_block_sparse(tsdf, return_faces=True)
    second = extract_mesh_block_sparse(tsdf, return_faces=True)
    vertices, faces, normals, colors = first
    vertices_again, faces_again, normals_again, colors_again = second

    assert vertices.shape == (9, 3)
    assert faces.shape == (8, 3)
    assert normals.shape == colors.shape == vertices.shape
    assert torch.unique(vertices, dim=0).shape == vertices.shape
    assert bool((faces >= 0).all()) and bool((faces < len(vertices)).all())
    assert torch.allclose(vertices, vertices_again)
    assert torch.equal(faces, faces_again)
    assert torch.allclose(normals, normals_again)
    assert torch.allclose(colors, colors_again)
    assert torch.allclose(normals, torch.tensor((1.0, 0.0, 0.0)).expand_as(normals))

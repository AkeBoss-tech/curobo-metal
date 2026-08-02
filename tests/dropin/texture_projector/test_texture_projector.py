import pytest
import torch

from curobo._src.perception.mapper.projector_texture import (
    ProjectiveTextureProjector,
    ProjectiveTextureProjectorCfg,
)
from curobo._src.perception.mapper.storage import (
    BlockDataView,
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
    OccupiedVoxels,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


class _Renderer:
    def render_depth(self, intrinsics, pose, image_shape):
        del intrinsics, pose
        return torch.ones((1, *image_shape))


def _projector(device="cpu"):
    tsdf = BlockSparseTSDF(BlockSparseTSDFCfg(grid_shape=(2, 2, 2), voxel_size=0.1, device=device))
    return ProjectiveTextureProjector(
        tsdf, _Renderer(),
        ProjectiveTextureProjectorCfg(1, 4, 4, 0.01, 4.0, 0.1),
    )


def _observation(device="cpu", depth=True):
    rgb = torch.zeros((4, 4, 3), dtype=torch.uint8, device=device)
    rgb[..., 0] = 17
    rgb[..., 1] = 99
    rgb[..., 2] = 201
    return CameraObservation(
        rgb_image=rgb,
        depth_image=torch.ones((4, 4), device=device) if depth else None,
        intrinsics=torch.tensor(((4.0, 0, 1.5), (0, 4.0, 1.5), (0, 0, 1)), dtype=torch.float32, device=device),
        pose=Pose(torch.zeros(3, device=device), torch.tensor((1.0, 0, 0, 0), device=device)),
        depth_to_meter=1.0,
    )


def _voxels(device="cpu"):
    centers = torch.tensor(((0.0, 0, 1.0), (0.0, 0, 1.4), (2.0, 0, 1.0)), device=device)
    rgbw = torch.zeros((3, 1, 4), device=device)
    view = BlockDataView(rgbw, torch.zeros((3, 3), dtype=torch.int32, device=device), 3,
                         torch.zeros(3, device=device), 0.1, 1, (1, 1, 3))
    return OccupiedVoxels(centers, torch.arange(3, device=device), view)


def test_projects_visible_rgbd_voxels_and_rejects_occlusion():
    projector = _projector()
    textured = projector.texture_occupied_voxels(
        _voxels(), _observation(), camera_min_distance=None, camera_max_distance=None,
        texture_depth_tolerance_m=0.05,
    )
    assert textured.texture_valid.tolist() == [True, False, False]
    assert textured.texture_colors[0].tolist() == [17, 99, 201]
    assert textured.colors_uint8()[0].tolist() == [17, 99, 201]


def test_batches_atlas_and_mesh_fallback_are_portable():
    projector = _projector()
    prepared = projector.prepare_mesh_projection(_observation(), texture_depth_tolerance_m=None)
    assert prepared.texture_atlas.shape == (4, 4, 3)
    vertices = torch.tensor(((0.0, 0, 1.0), (0.1, 0, 1.0), (0, 0.1, 1.0)))
    result = projector.project_mesh(vertices, torch.tensor(((0, 1, 2),)), torch.zeros_like(vertices),
                                    torch.full((3, 3), 7, dtype=torch.uint8), prepared,
                                    camera_min_distance=None, camera_max_distance=None)
    assert result.texture_image.shape == (4, 4, 3)
    assert result.vertex_colors.shape == (3, 3)
    assert torch.allclose(result.vertex_colors[0], torch.tensor((17, 99, 201)) / 255.0)
    assert torch.allclose(result.texture_uvs[0], torch.tensor((0.5, 0.5)))


def test_missing_depth_uses_renderer_and_bad_batches_are_explicit():
    projector = _projector()
    textured = projector.texture_occupied_voxels(
        _voxels(), _observation(depth=False), camera_min_distance=None, camera_max_distance=None,
        texture_depth_tolerance_m=None,
    )
    assert textured.texture_valid[0]
    with pytest.raises(ValueError, match="complete camera batches"):
        ProjectiveTextureProjector(_projector()._tsdf, _Renderer(),
            ProjectiveTextureProjectorCfg(2, 4, 4, 0.01, 4.0, 0.1)
        ).prepare_mesh_projection(_observation(), texture_depth_tolerance_m=None)


def test_batched_cameras_build_row_major_atlas_and_choose_nearest_visible_view():
    projector = ProjectiveTextureProjector(
        _projector()._tsdf, _Renderer(), ProjectiveTextureProjectorCfg(2, 4, 4, 0.01, 4.0, 0.1)
    )
    rgb = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    rgb[0, ..., 0] = 30
    rgb[1, ..., 1] = 90
    observation = CameraObservation(
        rgb_image=rgb, depth_image=torch.ones((2, 4, 4)),
        intrinsics=torch.tensor([((4.0, 0, 1.5), (0, 4.0, 1.5), (0, 0, 1))] * 2),
        pose=Pose(torch.tensor(((0.0, 0, 0), (0.0, 0, 0.1))),
                  torch.tensor(((1.0, 0, 0, 0),) * 2)), depth_to_meter=1.0,
    )
    prepared = projector.prepare_mesh_projection(observation, texture_depth_tolerance_m=0.2)
    assert prepared.texture_atlas.shape == (4, 8, 3)
    textured = projector.texture_occupied_voxels(
        _voxels(), observation, camera_min_distance=None, camera_max_distance=None,
        texture_depth_tolerance_m=0.2,
    )
    # The camera at z=0.1 is closer to the point but still depth-consistent.
    assert textured.texture_colors[0].tolist() == [0, 90, 0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_texture_projection_stays_on_mps_without_cpu_fallback():
    projector = _projector("mps")
    textured = projector.texture_occupied_voxels(
        _voxels("mps"), _observation("mps"), camera_min_distance=None, camera_max_distance=None,
        texture_depth_tolerance_m=0.05,
    )
    assert textured.texture_colors.device.type == "mps"
    assert textured.texture_valid[0].item()

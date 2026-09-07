"""Portable high-level TSDF integration over the dense CPU/MPS mapper.

The public configuration intentionally accepts the V2 block-sparse options.
The map state is dense (and therefore has no Warp hash/pool ABI), while depth
fusion, mesh/voxel extraction, persistence, and region mutation are real
PyTorch operations on CPU or MPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh
from curobo._src.perception.mapper.block_allocation import calculate_tsdf_max_blocks
from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    MAX_POOL_IDX,
    _validate_color_grid_size,
    _validate_feature_block_grid_size,
    _validate_feature_channels_per_thread,
    _validate_feature_grid_shape,
    resolve_feature_integration_kernel,
    validate_grid_shape_for_hash_layout,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import BlockSparseKernels, make_block_sparse_kernels
from curobo._src.perception.mapper.kernel.wp_integrate_camera_project import CameraProjectIntegrator
from curobo._src.perception.mapper.kernel.wp_integrate_lidar_project import LidarProjectIntegrator
from curobo._src.perception.mapper.kernel.wp_stamp_obstacles import stamp_scene_obstacles
from curobo._src.perception.mapper.kernel.wp_voxel_extraction import extract_matching_voxels_block_sparse, extract_occupied_voxels_block_sparse, extract_surface_voxels_block_sparse
from curobo._src.perception.mapper.mesh_extractor import extract_mesh_block_sparse
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.perception.mapper.storage import BlockSparseTSDF, BlockSparseTSDFCfg, MatchedVoxels
from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    is_sparse_block_payload,
    load_block_checkpoint,
    prepare_blocks_for_import,
    validate_block_metadata_for_target,
    validate_sparse_block_payload,
)
from curobo._src.types.lidar import LidarObservation
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import profile_class_methods
from .mapper import Mapper
from .mapper_cfg import MapperCfg
from .projector_texture import ProjectiveTextureProjector, ProjectiveTextureProjectorCfg
from .storage import BlockDataView, OccupiedVoxels
from .sparse_runtime import PortableSparseTSDF
from curobo._src.types.camera import CameraObservation
from curobo_metal.ops.perception.core import dense_esdf

# These symbols name raw Warp launch helpers in upstream.  The portable
# integrator implements its lifecycle directly and deliberately does not
# expose a fake kernel implementation.
clear_static_channel = None
decay_and_recycle = None
decay_frustum_aware_multi_sensor = None


@dataclass
class BlockSparseTSDFIntegratorCfg:
    voxel_size: float = 0.005
    origin: torch.Tensor | None = None
    truncation_distance: float = 0.04
    max_blocks: Optional[int] = None
    hash_capacity: Optional[int] = None
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 1.0
    time_decay: float = 1.0
    minimum_tsdf_weight: float = 0.1
    grid_shape: Tuple[int, int, int] | None = None
    roughness: float = 3.0
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    texture_num_cameras: Optional[int] = None
    texture_camera_image_height: Optional[int] = None
    texture_camera_image_width: Optional[int] = None
    lidar_num_sensors: int = 0
    lidar_image_height: Optional[int] = None
    lidar_image_width: Optional[int] = None
    lidar_feature_grid_height: Optional[int] = None
    lidar_feature_grid_width: Optional[int] = None
    lidar_linear_interpolation_max_allowable_difference_vox: float = 2.0
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox: float = 0.5
    max_visible_blocks_per_lidar_integration: Optional[int] = None
    max_support_pixels_per_block_lidar: int = 32
    enable_static: bool = False
    static_obstacle_color: Tuple[int, int, int] = (20, 20, 20)
    seeding_method: str = "gather"
    num_cameras: int = 1
    device: str = "cuda:0"
    block_size: int = 8
    feature_dim: int = 0
    feature_block_grid_size: int = 1
    feature_grid_height: Optional[int] = None
    feature_grid_width: Optional[int] = None
    max_visible_blocks_per_integration: Optional[int] = None
    max_support_pixels_per_block_camera: int = 32
    feature_channels_per_thread: int = 4
    max_feature_tile_channels: int = 4096
    color_grid_size: int = 1
    feature_integration_kernel: str = "auto"
    profile_integration_kernel_timings: bool = False
    accumulator_w_max: float = 10000.0

    def __post_init__(self) -> None:
        if self.grid_shape is None:
            raise ValueError("grid_shape is required by the portable bounded mapper")
        if len(self.grid_shape) != 3 or any(int(n) <= 0 for n in self.grid_shape):
            raise ValueError("grid_shape must contain three positive dimensions")
        self.grid_shape = tuple(int(n) for n in self.grid_shape)
        scalar_values = {
            "voxel_size": self.voxel_size,
            "truncation_distance": self.truncation_distance,
            "depth_minimum_distance": self.depth_minimum_distance,
            "depth_maximum_distance": self.depth_maximum_distance,
            "frustum_decay": self.frustum_decay,
            "time_decay": self.time_decay,
            "minimum_tsdf_weight": self.minimum_tsdf_weight,
            "roughness": self.roughness,
            "accumulator_w_max": self.accumulator_w_max,
        }
        if not all(math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError("TSDF configuration scalars must be finite")
        if self.voxel_size <= 0 or self.truncation_distance <= 0 or self.accumulator_w_max <= 0:
            raise ValueError("voxel_size, truncation_distance, and accumulator_w_max must be positive")
        if self.depth_minimum_distance < 0 or self.depth_minimum_distance >= self.depth_maximum_distance:
            raise ValueError("depth_minimum_distance must be nonnegative and less than depth_maximum_distance")
        if not 0 < self.time_decay <= 1 or not 0 < self.frustum_decay <= 1:
            raise ValueError("time_decay and frustum_decay must be in (0, 1]")
        if self.minimum_tsdf_weight < 0 or self.roughness <= 0:
            raise ValueError("minimum_tsdf_weight must be nonnegative and roughness positive")
        if self.block_size < 1 or self.block_size > 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be 1 or a power of two no greater than 32")
        if self.seeding_method not in {"gather", "scatter"}:
            raise ValueError("seeding_method must be 'gather' or 'scatter'")
        if self.feature_integration_kernel not in {"auto", "grouped", "tiled"}:
            raise ValueError("feature_integration_kernel must be 'auto', 'grouped', or 'tiled'")
        _validate_color_grid_size(self.color_grid_size, self.block_size)
        _validate_feature_block_grid_size(self.feature_block_grid_size, self.block_size)
        _validate_feature_channels_per_thread(self.feature_channels_per_thread)
        _validate_feature_grid_shape(
            self.feature_dim, self.feature_grid_height, self.feature_grid_width
        )
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be positive")
        if self.max_support_pixels_per_block_camera <= 0:
            raise ValueError("max_support_pixels_per_block_camera must be positive")
        if self.max_feature_tile_channels <= 0:
            raise ValueError("max_feature_tile_channels must be positive")
        if self.image_height is not None or self.image_width is not None:
            if self.image_height is None or self.image_width is None:
                raise ValueError("image_height and image_width must be specified together")
            if self.image_height <= 0 or self.image_width <= 0:
                raise ValueError("image_height and image_width must be positive")
        if (self.texture_camera_image_height is None) != (self.texture_camera_image_width is None):
            raise ValueError("texture_camera_image_height and texture_camera_image_width must be specified together")
        if self.texture_num_cameras is None:
            self.texture_num_cameras = self.num_cameras
        if self.texture_num_cameras <= 0:
            raise ValueError("texture_num_cameras must be positive")
        if self.origin is None:
            self.origin = torch.zeros(3, dtype=torch.float32)
        else:
            self.origin = torch.as_tensor(self.origin, dtype=torch.float32)
        if self.origin.shape != (3,):
            raise ValueError("origin must be an xyz vector")
        if self.max_blocks is None:
            # This is an exposed sizing estimate only: the portable backend is dense.
            self.max_blocks = max(1, int(math.prod(self.grid_shape) / self.block_size**3))
        if self.hash_capacity is None:
            self.hash_capacity = max(1, int(math.ceil(self.max_blocks * 2.0)))
        if self.max_blocks > MAX_POOL_IDX:
            raise ValueError("max_blocks exceeds the packed hash pool capacity")
        if self.max_visible_blocks_per_integration is None:
            self.max_visible_blocks_per_integration = self.max_blocks
        if not 0 < self.max_visible_blocks_per_integration <= self.max_blocks:
            raise ValueError("max_visible_blocks_per_integration must be in [1, max_blocks]")
        if self.max_visible_blocks_per_lidar_integration is None:
            self.max_visible_blocks_per_lidar_integration = self.max_blocks
        if not 0 < self.max_visible_blocks_per_lidar_integration <= self.max_blocks:
            raise ValueError("max_visible_blocks_per_lidar_integration must be in [1, max_blocks]")


class BlockSparseTSDFIntegrator:
    """cuRobo-shaped TSDF facade with a bounded dense portable implementation."""

    def __init__(self, config: BlockSparseTSDFIntegratorCfg, kernels: Optional[BlockSparseKernels] = None):
        if kernels is not None:
            raise NotImplementedError("custom Warp block-sparse kernels are unavailable on CPU/MPS")
        self.config = config
        self.cfg = config  # historical portable spelling
        # A nominal 512^3 map exists specifically to be sparse.  Keep smaller
        # volumes on the dense production path, but never materialize a huge
        # TSDF/ESDF/gradient cube merely to emulate a block pool.
        if (
            math.prod(config.grid_shape) > 16_777_216
            or config.lidar_num_sensors
            or config.feature_dim
            or config.color_grid_size > 1
        ):
            self.mapper = None
            self._tsdf = PortableSparseTSDF(config)
            self._tsdf.kernels = make_block_sparse_kernels(config)
            self._camera_integrator = SimpleNamespace(
                max_visible_blocks_per_integration=config.max_visible_blocks_per_integration,
                use_tiled_feature_kernel=config.feature_integration_kernel != "grouped",
                pool_indices=torch.zeros(
                    config.max_visible_blocks_per_integration,
                    dtype=torch.int32,
                    device=self._tsdf.device,
                ),
            )
            self._initialize_texture_projector()
            self._frame_count = 0
            return
        # Integrator grid_shape is source (z,y,x); MapperCfg extent is world
        # (x,y,z).  Keep this conversion explicit so lower-corner origins and
        # dense native indices agree for rectangular maps.
        extent = (
            int(config.grid_shape[2]) * config.voxel_size,
            int(config.grid_shape[1]) * config.voxel_size,
            int(config.grid_shape[0]) * config.voxel_size,
        )
        center = config.origin + torch.as_tensor(extent, dtype=config.origin.dtype) / 2
        self.mapper = Mapper(MapperCfg(
            extent_meters_xyz=extent,
            voxel_size=config.voxel_size,
            grid_center=center,
            truncation_distance=config.truncation_distance,
            minimum_tsdf_weight=config.minimum_tsdf_weight,
            depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance,
            decay_factor=config.time_decay,
            frustum_decay_factor=config.frustum_decay,
            block_size=config.block_size,
            enable_static=config.enable_static,
            static_obstacle_color=config.static_obstacle_color,
            device=config.device,
            accumulator_w_max=config.accumulator_w_max,
        ))
        # Preserve the component spelling used by V2 callers.  This remains a
        # bounded dense lifecycle object, not a CUDA/Warp block-pool ABI.
        self._tsdf = self.mapper.tsdf
        self._initialize_texture_projector()
        self._frame_count = 0

    def _initialize_texture_projector(self) -> None:
        """Create the source-visible projector when its dimensions are configured."""
        height = self.config.texture_camera_image_height or self.config.image_height
        width = self.config.texture_camera_image_width or self.config.image_width
        self._texture_projector = None
        if height is None or width is None:
            return
        self._texture_projector = ProjectiveTextureProjector(
            self._tsdf,
            BlockSparseTSDFRenderer(self),
            ProjectiveTextureProjectorCfg(
                texture_num_cameras=self.config.texture_num_cameras,
                image_height=height,
                image_width=width,
                depth_minimum_distance=self.config.depth_minimum_distance,
                depth_maximum_distance=self.config.depth_maximum_distance,
                voxel_size=self.config.voxel_size,
            ),
        )

    @property
    def tsdf(self) -> BlockSparseTSDF:
        return self._tsdf

    @property
    def _voxel_size(self) -> float:
        """Metric spacing of the portable dense map.

        These source-shaped read-only configuration fields are useful to
        callers that receive an integrator rather than its ``.tsdf`` storage.
        They describe the actual dense map and do not imply a Warp block pool.
        """
        return self.config.voxel_size

    @property
    def _origin(self) -> torch.Tensor:
        """World-space lower corner of the bounded dense map."""
        device = self._tsdf.device if getattr(self._tsdf, "_portable_sparse", False) else self.mapper.device
        return self.config.origin.to(device=device)

    @property
    def _truncation_distance(self) -> float:
        """Metric TSDF truncation distance."""
        return self.config.truncation_distance

    def reset(self):
        self._frame_count = 0
        if getattr(self._tsdf, "_portable_sparse", False):
            return self._tsdf.reset()
        return self.mapper.reset()

    def _render_sparse(
        self, intrinsics: torch.Tensor, pose, image_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render portable dense or sparse geometry with the source shapes."""
        if self.mapper is not None:
            return self.mapper.render(intrinsics, pose, image_shape)
        matrices = intrinsics.to(device=self._tsdf.device, dtype=torch.float32)
        if matrices.ndim == 2:
            matrices = matrices.unsqueeze(0)
            batched = False
        else:
            batched = True
        positions = pose.position.to(device=self._tsdf.device, dtype=torch.float32).reshape(-1, 3)
        quaternions = pose.quaternion.to(device=self._tsdf.device, dtype=torch.float32).reshape(-1, 4)
        if len(matrices) != len(positions) or len(positions) != len(quaternions):
            raise ValueError("intrinsics and pose must have the same camera count")
        height, width = image_shape
        depths = []
        mesh = self.extract_mesh(surface_only=True)
        points = torch.as_tensor(mesh.vertices, device=self._tsdf.device, dtype=torch.float32)
        rotations = pose.__class__(positions, quaternions).get_rotation()
        for camera in range(len(matrices)):
            depth = torch.full(
                (height * width,), torch.inf, device=self._tsdf.device, dtype=torch.float32
            )
            if len(points):
                local = (points - positions[camera]) @ rotations[camera]
                z = local[:, 2]
                safe_z = z.clamp_min(torch.finfo(z.dtype).tiny)
                u = torch.floor(
                    matrices[camera, 0, 0] * local[:, 0] / safe_z
                    + matrices[camera, 0, 2]
                ).long()
                v = torch.floor(
                    matrices[camera, 1, 1] * local[:, 1] / safe_z
                    + matrices[camera, 1, 2]
                ).long()
                valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                if bool(valid.any().item()):
                    linear = v[valid] * width + u[valid]
                    depth.scatter_reduce_(0, linear, z[valid], reduce="amin", include_self=True)
            depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
            depths.append(depth.reshape(height, width))
        depth = torch.stack(depths)
        valid = depth > 0
        normals = torch.zeros(depth.shape + (3,), device=depth.device, dtype=depth.dtype)
        for index in range(depth.shape[0]):
            d = depth[index]
            k = matrices[index]
            yy, xx = torch.meshgrid(
                torch.arange(height, device=depth.device, dtype=depth.dtype),
                torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij"
            )
            points_camera = torch.stack(((xx-k[0, 2]) / k[0, 0] * d,
                                         (yy-k[1, 2]) / k[1, 1] * d, d), -1)
            dx = points_camera[:, 2:] - points_camera[:, :-2]
            dy = points_camera[2:] - points_camera[:-2]
            nrm = torch.cross(dx[1:-1], dy[:, 1:-1], dim=-1)
            nrm = nrm / torch.linalg.vector_norm(nrm, dim=-1, keepdim=True).clamp_min(1e-6)
            normals[index, 1:-1, 1:-1] = nrm
        normals = torch.where(valid[..., None], normals, torch.zeros_like(normals))
        if not batched:
            return depth[0], normals[0], valid[0]
        return depth, normals, valid

    def import_blocks(self, blocks: Dict[str, torch.Tensor]) -> int:
        if isinstance(blocks, (str, bytes)) or hasattr(blocks, "__fspath__"):
            if getattr(self._tsdf, "_portable_sparse", False):
                checkpoint = load_block_checkpoint(blocks)
                validate_block_metadata_for_target(checkpoint["block_metadata"], self._tsdf)
                payload = prepare_blocks_for_import(
                    checkpoint["blocks"], checkpoint["block_metadata"],
                    import_weight=None,
                    minimum_tsdf_weight=self.config.minimum_tsdf_weight,
                    block_empty_threshold=0.0,
                )
                if not is_sparse_block_payload(payload):
                    raise NotImplementedError(
                        "portable sparse integrators can import only compact block checkpoints"
                    )
                result = self._tsdf.import_blocks(payload)
            else:
                result = self.mapper.import_blocks(blocks)
        elif isinstance(blocks, dict):
            if getattr(self._tsdf, "_portable_sparse", False):
                if not is_sparse_block_payload(blocks):
                    raise NotImplementedError(
                        "portable sparse integrators can import only compact block payloads"
                    )
                validate_sparse_block_payload(blocks, build_block_metadata(self._tsdf))
                result = self._tsdf.import_blocks(blocks)
            else:
                # Dense state dictionaries are intentionally portable; Warp block-pool
                # dictionaries have a different memory contract and must not be guessed.
                required = {"tsdf", "weight", "occupancy", "esdf", "gradient", "generation"}
                if not required.issubset(blocks):
                    raise NotImplementedError("importing Warp block-pool payloads requires CUDA/Warp")
                self.mapper._mapper.load_state_dict(blocks)
                result = int((self.mapper._mapper.state.weight > 0).sum().item())
        else:
            raise TypeError("blocks must be a portable dense state dictionary or checkpoint path")
        self._frame_count = int(result > 0)
        return result

    def integrate(
        self,
        observation: Optional[CameraObservation | LidarObservation] = None,
        *,
        camera_observation: Optional[CameraObservation] = None,
        lidar_observation: Optional[LidarObservation] = None,
    ):
        if observation is not None and (camera_observation is not None or lidar_observation is not None):
            raise ValueError("observation cannot be combined with camera_observation or lidar_observation")
        if observation is None and camera_observation is None and lidar_observation is None:
            raise ValueError("integrate() requires observation, camera_observation, or lidar_observation")
        if lidar_observation is not None:
            if not getattr(self._tsdf, "_portable_sparse", False):
                raise NotImplementedError("LiDAR integration requires bounded sparse storage")
            self._validate_lidar_observation(lidar_observation)
            visible = self._tsdf.integrate_lidar(
                lidar_observation,
                visible_capacity=self.config.max_visible_blocks_per_lidar_integration,
            )
            self._frame_count += 1
            self._last_lidar_visible = visible
            return None
        # The portable mapper has no projective frustum-recycling kernel, but
        # global temporal decay is meaningful and can be implemented exactly
        # over its dense state before each camera update.
        selected_observation = observation if observation is not None else camera_observation
        self._validate_camera_observation(selected_observation)
        if getattr(self._tsdf, "_portable_sparse", False):
            if self._frame_count and self.config.time_decay != 1.0:
                self._tsdf.decay_and_recycle(self.config.time_decay)
            self._tsdf.integrate(
                selected_observation,
                visible_capacity=self.config.max_visible_blocks_per_integration,
            )
            self._frame_count += 1
            return None
        # Validation must precede every mutable operation.  In particular a
        # malformed later frame cannot decay/recycle a previously valid map.
        if self._frame_count:
            self._apply_frame_decay(camera_observation=selected_observation)
        result = self._integrate_camera_frame(
            selected_observation, advance_frame=False, apply_decay=False
        )
        self._frame_count += 1
        return result

    def _integrate_camera_frame(self, observation, *, advance_frame=True, apply_decay=True):
        self._validate_camera_observation(observation)
        if apply_decay:
            if self._frame_count:
                self._apply_frame_decay(camera_observation=observation)
        result = self.mapper.integrate(camera_observation=observation)
        if advance_frame:
            self._frame_count += 1
        return result

    def _validate_camera_observation(self, observation: CameraObservation) -> int:
        """Validate a camera update before it can mutate dense map state.

        Source V2 expects a camera-axis batch.  The portable facade also
        accepts an unbatched image for the common ``num_cameras == 1`` case,
        but otherwise enforces the same camera count, resolution, dtype, and
        device invariants.  Valid zero/NaN/out-of-range depth pixels remain a
        *data mask* handled by the production fusion operator rather than a
        malformed frame.
        """
        if not isinstance(observation, CameraObservation):
            raise TypeError(
                "camera observation must be a CameraObservation, got "
                f"{type(observation).__name__}"
            )
        if (
            observation.feature_grid is not None
            and observation.depth_image is not None
            and observation.feature_grid.device != observation.depth_image.device
        ):
            raise ValueError(
                f"feature_grid device {observation.feature_grid.device} does not match "
                f"depth_image device {observation.depth_image.device}"
            )
        observation.validate(require_depth=True, require_intrinsics=True, require_pose=True)
        depth = observation.depth_image
        assert depth is not None  # narrowed by validate() above
        count = 1 if depth.ndim == 2 else int(depth.shape[0])
        if count != self.config.num_cameras:
            raise ValueError(
                f"Expected num_cameras={self.config.num_cameras}, got depth_image camera count {count}"
            )
        if self.config.image_height is not None and tuple(depth.shape[-2:]) != (
            self.config.image_height, self.config.image_width,
        ):
            raise ValueError(
                "depth_image spatial shape must match configured image_height/image_width"
            )
        map_device = (
            self._tsdf.device
            if getattr(self._tsdf, "_portable_sparse", False)
            else self.mapper._mapper.state.tsdf.device
        )
        map_dtype = torch.float32 if getattr(self._tsdf, "_portable_sparse", False) else self.mapper._mapper.state.tsdf.dtype
        if depth.device.type != map_device.type or (
            depth.device.index not in (None, map_device.index)
            and map_device.index not in (None, depth.device.index)
        ):
            raise ValueError("camera observation must share the TSDF map device")
        if depth.dtype != map_dtype:
            raise TypeError("camera depth_image must share the TSDF map floating dtype")
        assert observation.intrinsics is not None and observation.pose is not None
        intrinsics = observation.intrinsics
        intrinsics_count = 1 if intrinsics.ndim == 2 else int(intrinsics.shape[0])
        if intrinsics_count not in (1, count):
            raise ValueError("camera intrinsics count must be one or match depth_image cameras")
        matrix = observation.pose.get_matrix()
        pose_count = 1 if matrix.ndim == 2 else int(matrix.shape[0])
        if pose_count not in (1, count):
            raise ValueError("camera pose count must be one or match depth_image cameras")
        if observation.image_segmentation is not None:
            # The geometry-only map does not interpret semantic labels, but a
            # shape mismatch is still a caller error rather than a reason to
            # pass a malformed auxiliary image into a future backend.
            if tuple(observation.image_segmentation.shape) != tuple(depth.shape):
                raise ValueError("image_segmentation must have the same shape as depth_image")
        feature_grid = observation.feature_grid
        if feature_grid is not None:
            if self.config.feature_dim == 0:
                raise ValueError("feature_grid was provided but feature_dim == 0")
            if feature_grid.dtype != torch.float16:
                raise ValueError("feature_grid dtype must be torch.float16")
            if feature_grid.stride(-1) != 1:
                raise ValueError("feature_grid must have stride 1 on the channel dim")
            if feature_grid.device != depth.device:
                raise ValueError(
                    f"feature_grid device {feature_grid.device} does not match "
                    f"depth_image device {depth.device}"
                )
            if feature_grid.ndim != 4:
                raise ValueError("feature_grid must have shape [C,H,W,feature_dim]")
            if feature_grid.shape[0] != count:
                raise ValueError("feature_grid camera count must match depth_image cameras")
            if tuple(feature_grid.shape[1:3]) != (
                self.config.feature_grid_height,
                self.config.feature_grid_width,
            ):
                raise ValueError(
                    "feature_grid spatial shape must match configured "
                    "feature_grid_height/feature_grid_width"
                )
            if feature_grid.shape[-1] != self.config.feature_dim:
                raise ValueError("feature_grid channel count must match feature_dim")
        return count

    def _camera_frustum_mask(self, observation) -> torch.Tensor:
        """Return dense cells projecting into at least one camera image.

        The pinned implementation tracks frustum flags per allocated sparse
        block.  We can preserve the useful per-frame semantic without a raw
        block pool by projecting the actual bounded dense voxel centers.  The
        mask intentionally describes a camera frustum, not a depth hit: cells
        behind the measured surface still receive the configured in-view
        decay, just as they do in the source lifecycle.
        """
        if observation is None:
            return torch.zeros_like(self.mapper._mapper.state.occupancy)
        if not hasattr(observation, "depth_image") or observation.depth_image is None:
            raise ValueError("camera observation requires depth_image for frustum decay")
        if observation.intrinsics is None or observation.pose is None:
            raise ValueError("camera observation requires intrinsics and pose for frustum decay")

        state = self.mapper._mapper.state
        dtype, device = state.tsdf.dtype, state.tsdf.device
        depth = observation.depth_image.to(device=device)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim != 3 or depth.shape[0] == 0:
            raise ValueError("camera depth_image must have shape [H,W] or [C,H,W]")
        camera_count, height, width = depth.shape
        intrinsics = observation.intrinsics.to(device=device, dtype=dtype)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape != (camera_count, 3, 3):
            raise ValueError("camera intrinsics must have shape [3,3] or [C,3,3]")
        matrices = observation.pose.get_matrix().to(device=device, dtype=dtype)
        if matrices.ndim == 2:
            matrices = matrices.unsqueeze(0)
        if matrices.shape != (camera_count, 4, 4):
            raise ValueError("camera pose must contain one transform per camera")

        centers = torch.stack(torch.meshgrid(
            *[(torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2)
              * self.config.voxel_size + center
              for n, center in zip(state.tsdf.shape[1:], self.mapper._mapper.config.grid_center)],
            indexing="ij",
        ), -1).reshape(-1, 3)
        rotation, translation = matrices[:, :3, :3], matrices[:, :3, 3]
        local = torch.einsum("cij,nj->cni", rotation.transpose(-1, -2), centers) - torch.einsum(
            "cij,cj->ci", rotation.transpose(-1, -2), translation
        )[:, None, :]
        z = local[..., 2]
        safe_z = torch.where(z.abs() > torch.finfo(dtype).eps, z, torch.ones_like(z))
        u = torch.round(intrinsics[:, 0, 0, None] * local[..., 0] / safe_z + intrinsics[:, 0, 2, None])
        v = torch.round(intrinsics[:, 1, 1, None] * local[..., 1] / safe_z + intrinsics[:, 1, 2, None])
        visible = (torch.isfinite(u) & torch.isfinite(v) & (z > 0)
                   & (u >= 0) & (u < width) & (v >= 0) & (v < height)).any(0)
        return visible.reshape_as(state.occupancy)[None] if state.occupancy.ndim == 3 else visible.reshape_as(state.occupancy)

    def _apply_frame_decay(self, camera_observation=None, lidar_observation=None):
        if lidar_observation is not None:
            raise NotImplementedError("LiDAR frustum decay requires Warp/CUDA")
        if self.config.time_decay == 1.0 and self.config.frustum_decay == 1.0:
            return None
        state = self.mapper._mapper.state
        if self.config.frustum_decay != 1.0 and camera_observation is not None:
            in_view = self._camera_frustum_mask(camera_observation)
            factor = torch.where(
                in_view,
                torch.full_like(state.weight, self.config.time_decay * self.config.frustum_decay),
                torch.full_like(state.weight, self.config.time_decay),
            )
        else:
            factor = torch.full_like(state.weight, self.config.time_decay)
        weight = state.weight * factor
        expired = (weight > 0) & (weight < self.config.minimum_tsdf_weight)
        occupancy = state.occupancy & ~expired
        tsdf = torch.where(expired, torch.ones_like(state.tsdf), state.tsdf)
        esdf, gradient = dense_esdf(
            occupancy, self.config.voxel_size, self.mapper._mapper.config.unobserved_esdf,
            dtype=state.tsdf.dtype,
        )
        self.mapper._replace_state(
            tsdf=tsdf, weight=torch.where(expired, torch.zeros_like(weight), weight),
            occupancy=occupancy, esdf=esdf, gradient=gradient,
            generation=state.generation + 1,
        )
        return None

    def _integrate_lidar_frame(self, observation, *, advance_frame=True):
        del observation, advance_frame
        raise NotImplementedError("LiDAR TSDF integration requires Warp/CUDA")

    def _validate_lidar_observation(self, observation):
        if not isinstance(observation, LidarObservation):
            raise TypeError("lidar observation must be a LidarObservation")
        observation.validate(require_range=True, require_pose=True, require_calibration=True)
        assert observation.range_image is not None
        if observation.range_image.shape[0] != self.config.lidar_num_sensors:
            raise ValueError("range_image sensor count must match lidar_num_sensors")
        if self.config.lidar_image_height is not None and tuple(observation.range_image.shape[-2:]) != (
            self.config.lidar_image_height,
            self.config.lidar_image_width,
        ):
            raise ValueError("range_image shape must match configured LiDAR image dimensions")
        if observation.range_image.device.type != self._tsdf.device.type:
            raise ValueError("LiDAR observation must share the TSDF map device")
        return self.config.lidar_num_sensors

    def recycle_empty_blocks(self) -> int:
        if getattr(self._tsdf, "_portable_sparse", False):
            return self._tsdf.decay_and_recycle(1.0)
        # Dense storage contains no unallocated pool blocks.  Recycle truly empty
        # observed cells by clearing weights below the configured observation floor.
        state = self.mapper._mapper.state
        mask = (state.weight > 0) & (state.weight < self.config.minimum_tsdf_weight)
        count = int(mask.sum().item())
        if count:
            occupancy = state.occupancy & ~mask
            esdf, gradient = dense_esdf(
                occupancy, self.config.voxel_size, self.mapper._mapper.config.unobserved_esdf,
                dtype=state.tsdf.dtype,
            )
            self.mapper._replace_state(
                tsdf=torch.where(mask, torch.ones_like(state.tsdf), state.tsdf),
                weight=torch.where(mask, torch.zeros_like(state.weight), state.weight),
                occupancy=occupancy, esdf=esdf, gradient=gradient,
                generation=state.generation + 1,
            )
        return count

    def clear_region(self, bounds_min, bounds_max) -> int:
        if getattr(self._tsdf, "_portable_sparse", False):
            return self._tsdf.clear_region(bounds_min, bounds_max)
        return self.mapper.clear_region(bounds_min, bounds_max)

    def clear_blocks(self, pool_indices) -> int:
        if getattr(self._tsdf, "_portable_sparse", False):
            return self._tsdf.clear_blocks(pool_indices)
        return self.mapper.clear_blocks(pool_indices)

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = False, level: float = 0.0) -> Mesh:
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        if getattr(self._tsdf, "_portable_sparse", False):
            vertices, faces, normals, colors = extract_mesh_block_sparse(
                self._tsdf,
                level=level,
                surface_only=surface_only,
                refine_iterations=refine_iterations,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
                return_faces=True,
            )
            return Mesh(
                name="block_sparse_tsdf_mesh",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=vertices,
                faces=faces,
                vertex_normals=normals,
                vertex_colors=colors,
            )
        mesh = self.mapper.extract_mesh(refine_iterations, surface_only)
        vertices = torch.as_tensor(mesh.vertices)
        mesh.vertex_normals = torch.zeros_like(vertices)
        # Geometry-only depth fusion has no RGB accumulator.  A deterministic
        # opaque neutral color makes the source Mesh field usable without
        # pretending texture/feature fusion occurred.
        mesh.vertex_colors = torch.full(
            (vertices.shape[0], 3), 0.5, device=vertices.device, dtype=vertices.dtype,
        )
        return mesh

    def extract_mesh_tensors(self, level: float = 0.0, surface_only: bool = False, refine_iterations: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        # The pinned low-level helper and raw tensor API intentionally expose
        # triangle-soup vertices.  ``extract_mesh`` above additionally offers
        # welded indexed faces for consumers that retain a Mesh object.
        if getattr(self._tsdf, "_portable_sparse", False):
            vertices, normals, colors = extract_mesh_block_sparse(
                self._tsdf,
                level=level,
                surface_only=surface_only,
                refine_iterations=refine_iterations,
                minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            )
            faces = torch.arange(vertices.shape[0], device=vertices.device, dtype=torch.int32).reshape(-1, 3)
            colors = (colors.clamp(0, 1) * 255).to(torch.uint8)
            return vertices, faces, normals, colors
        mesh = self.extract_mesh(refine_iterations=refine_iterations, surface_only=surface_only, level=level)
        vertices = torch.as_tensor(mesh.vertices)
        normals = torch.as_tensor(mesh.vertex_normals, device=vertices.device, dtype=vertices.dtype)
        colors = (torch.as_tensor(mesh.vertex_colors, device=vertices.device, dtype=vertices.dtype)
                  .clamp(0, 1) * 255).to(torch.uint8)
        return vertices, torch.as_tensor(mesh.faces, device=vertices.device, dtype=torch.int32), normals, colors

    def _get_texture_projector(self, texture_observations) -> ProjectiveTextureProjector:
        """Build a validated dense projective-texture adapter on demand.

        Texture dimensions are optional in the portable configuration so a
        caller can use a normal geometry-only map.  When an RGB observation is
        supplied we infer omitted dimensions from that observation; explicit
        configuration continues to be checked by the projector.
        """
        sample = texture_observations
        if not hasattr(sample, "rgb_image"):
            values = list(texture_observations)
            if not values:
                raise ValueError("texture_observations must contain at least one CameraObservation")
            sample = values[0]
        rgb = getattr(sample, "rgb_image", None)
        if rgb is None or rgb.ndim not in (3, 4) or rgb.shape[-1] != 3:
            raise ValueError("texture observations require rgb_image with shape [H,W,3] or [C,H,W,3]")
        height, width = int(rgb.shape[-3]), int(rgb.shape[-2])
        expected_cameras = self.config.texture_num_cameras
        if rgb.ndim == 4 and rgb.shape[0] != expected_cameras:
            raise ValueError("batched rgb_image camera count must match texture_num_cameras")
        projector = ProjectiveTextureProjector(
            self._tsdf,
            BlockSparseTSDFRenderer(self),
            ProjectiveTextureProjectorCfg(
                texture_num_cameras=expected_cameras,
                image_height=self.config.texture_camera_image_height or height,
                image_width=self.config.texture_camera_image_width or width,
                depth_minimum_distance=self.config.depth_minimum_distance,
                depth_maximum_distance=self.config.depth_maximum_distance,
                voxel_size=self.config.voxel_size,
            ),
        )
        self._texture_projector = projector
        return projector

    def extract_textured_mesh(self, texture_observations: CameraObservation | Sequence[CameraObservation], refine_iterations: int = 0,
                              surface_only: bool = True, level: float = 0.0,
                              camera_min_distance: Optional[float] = None,
                              camera_max_distance: Optional[float] = None,
                              texture_depth_tolerance_m: Optional[float] = None) -> Mesh:
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        vertices, faces, normals, colors = self.extract_mesh_tensors(
            level=level, surface_only=surface_only, refine_iterations=refine_iterations,
        )
        projector = self._texture_projector or self._get_texture_projector(texture_observations)
        projection = projector.prepare_mesh_projection(
            texture_observations, texture_depth_tolerance_m=texture_depth_tolerance_m,
        )
        return projector.project_mesh(
            vertices, faces, normals, colors, projection,
            camera_min_distance=camera_min_distance, camera_max_distance=camera_max_distance,
        )

    @staticmethod
    def _validate_subvoxel_factor(subvoxel_factor: int):
        if not isinstance(subvoxel_factor, int) or subvoxel_factor < 1:
            raise ValueError("subvoxel_factor must be a positive integer")
        return subvoxel_factor

    @staticmethod
    def _validate_max_points(max_points: Optional[int]):
        if max_points is not None and (not isinstance(max_points, int) or max_points < 1):
            raise ValueError("max_points must be a positive integer or None")
        return max_points

    def _limit_voxels_for_point_cap(
        self,
        voxels: OccupiedVoxels,
        *,
        subvoxel_factor: int,
        max_points: Optional[int],
    ) -> OccupiedVoxels:
        """Downsample source voxels before expansion, retaining full cells."""
        factor = self._validate_subvoxel_factor(subvoxel_factor)
        limit = self._validate_max_points(max_points)
        if limit is None or len(voxels) * factor**3 <= limit:
            return voxels
        source_count = limit // factor**3
        if source_count == 0:
            selected = torch.empty(0, dtype=torch.long, device=voxels.centers.device)
        else:
            selected = torch.linspace(
                0, len(voxels) - 1, steps=source_count, device=voxels.centers.device
            ).round().to(torch.long)
        colors = None if voxels.texture_colors is None else voxels.texture_colors[selected]
        valid = None if voxels.texture_valid is None else voxels.texture_valid[selected]
        return OccupiedVoxels(
            voxels.centers[selected],
            voxels.block_idx_per_voxel[selected],
            voxels.block_data,
            texture_colors=colors,
            texture_valid=valid,
            subvoxel_factor=1,
        )

    def _expand_subvoxels(
        self, voxels: OccupiedVoxels, *, subvoxel_factor: int
    ) -> OccupiedVoxels:
        """Expand voxel centers to an evenly spaced subvoxel lattice."""
        factor = self._validate_subvoxel_factor(subvoxel_factor)
        if factor == 1 or len(voxels) == 0:
            return OccupiedVoxels(
                voxels.centers,
                voxels.block_idx_per_voxel,
                voxels.block_data,
                texture_colors=voxels.texture_colors,
                texture_valid=voxels.texture_valid,
                subvoxel_factor=factor,
            )
        axis = (
            (torch.arange(factor, device=voxels.centers.device, dtype=voxels.centers.dtype) + 0.5)
            / factor
            - 0.5
        ) * float(self.config.voxel_size)
        offsets = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), -1
        ).reshape(-1, 3)
        count = int(offsets.shape[0])
        centers = (voxels.centers[:, None] + offsets[None]).reshape(-1, 3)
        indices = voxels.block_idx_per_voxel.repeat_interleave(count)
        colors = (
            None
            if voxels.texture_colors is None
            else voxels.texture_colors.repeat_interleave(count, dim=0)
        )
        valid = (
            None
            if voxels.texture_valid is None
            else voxels.texture_valid.repeat_interleave(count, dim=0)
        )
        return OccupiedVoxels(
            centers,
            indices,
            voxels.block_data,
            texture_colors=colors,
            texture_valid=valid,
            subvoxel_factor=factor,
        )

    def extract_surface_voxels(self, sdf_threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(centers, colors, signed_distances_m)`` near the surface.

        This differs intentionally from :meth:`extract_occupied_voxels`: it
        exposes *observed* cells on both sides of the zero crossing, matching
        the source debug/export API.  Geometry-only maps return deterministic
        neutral colors because they have no RGB accumulator.
        """
        threshold = self.config.truncation_distance if sdf_threshold is None else float(sdf_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("sdf_threshold must be finite and non-negative")
        if getattr(self._tsdf, "_portable_sparse", False):
            data = self._tsdf.data
            high_water = int(data.num_allocated.item())
            active = torch.nonzero(
                data.block_to_hash_slot[:high_water] >= 0, as_tuple=False
            ).flatten().to(torch.int32)
            voxels = self._extract_sparse_matched_voxels(
                active, surface_only=True, sdf_threshold=threshold
            )
            distances = []
            if len(voxels):
                pools = voxels.block_idx_per_voxel.to(device=self._tsdf.device, dtype=torch.long)
                # The sparse extractor returns block ownership but not a local
                # voxel slot.  Match each center to its block lattice to recover
                # the normalized TSDF value without manufacturing a dense map.
                block_size = int(data.block_size)
                blocks_xyz = torch.tensor(
                    [math.ceil(int(data.grid_shape[2]) / block_size),
                     math.ceil(int(data.grid_shape[1]) / block_size),
                     math.ceil(int(data.grid_shape[0]) / block_size)],
                    dtype=torch.float32, device=self._tsdf.device,
                )
                center_offset = torch.tensor(
                    [int(data.grid_shape[2]), int(data.grid_shape[1]), int(data.grid_shape[0])],
                    dtype=torch.float32, device=self._tsdf.device,
                ) * 0.5
                voxel_xyz = (
                    (voxels.centers - data.origin.to(self._tsdf.device))
                    / float(data.voxel_size) + center_offset - 0.5
                )
                local = voxel_xyz - (
                    data.block_coords.view(-1, 3)[pools].float()
                    + torch.floor(blocks_xyz * 0.5)
                ) * block_size
                local_idx = (
                    local[:, 0].round().long() * block_size * block_size
                    + local[:, 1].round().long() * block_size
                    + local[:, 2].round().long()
                )
                dynamic = data.block_data[pools, local_idx]
                distances = torch.where(
                    dynamic[:, 1] >= float(self.config.minimum_tsdf_weight),
                    dynamic[:, 0] / dynamic[:, 1].clamp_min(1.0e-6),
                    torch.full_like(dynamic[:, 0], float("inf")),
                )
                if data.has_static:
                    static = data.static_block_data[pools, local_idx]
                    distances = torch.minimum(distances, static.float())
                distances = distances * float(data.truncation_distance)
            distances = torch.as_tensor(distances, device=self._tsdf.device, dtype=torch.float32)
            return voxels.centers, voxels.colors_uint8(), distances

        state = self.mapper._mapper.state
        observed = state.weight >= self.config.minimum_tsdf_weight
        mask = observed & (state.tsdf.abs() <= threshold / self.config.truncation_distance)
        coordinates = torch.nonzero(mask[0], as_tuple=False)
        dtype = state.tsdf.dtype
        center = state.tsdf.new_tensor(self.mapper._mapper.config.grid_center)
        actual_shape = state.tsdf.shape[1:]
        centers = (coordinates.to(dtype) - (state.tsdf.new_tensor(actual_shape) - 1) / 2)
        centers = centers * self.config.voxel_size + center
        colors = torch.full((len(centers), 3), 128, device=state.tsdf.device, dtype=torch.uint8)
        distances = state.tsdf[0][mask[0]] * self.config.truncation_distance
        return centers, colors, distances

    def extract_occupied_voxels(self, surface_only: bool = False, sdf_threshold: float = None, *,
                                subvoxel_factor: int = 1, max_points: Optional[int] = None,
                                texture_observations: CameraObservation | Sequence[CameraObservation] | None = None, camera_min_distance: Optional[float] = None,
                                camera_max_distance: Optional[float] = None, texture_depth_tolerance_m: Optional[float] = None) -> OccupiedVoxels:
        subvoxel_factor = self._validate_subvoxel_factor(subvoxel_factor)
        max_points = self._validate_max_points(max_points)
        if getattr(self._tsdf, "_portable_sparse", False):
            data = self._tsdf.data
            high_water = int(data.num_allocated.item())
            active = torch.nonzero(
                data.block_to_hash_slot[:high_water] >= 0, as_tuple=False
            ).flatten().to(torch.int32)
            voxels = self._extract_sparse_matched_voxels(
                active, surface_only=surface_only, sdf_threshold=sdf_threshold
            )
            voxels = self._limit_voxels_for_point_cap(
                voxels, subvoxel_factor=subvoxel_factor, max_points=max_points
            )
            voxels = self._expand_subvoxels(voxels, subvoxel_factor=subvoxel_factor)
            if texture_observations is None:
                return voxels
            projector = self._texture_projector or self._get_texture_projector(texture_observations)
            return projector.texture_occupied_voxels(
                voxels, texture_observations, camera_min_distance=camera_min_distance,
                camera_max_distance=camera_max_distance,
                texture_depth_tolerance_m=texture_depth_tolerance_m,
            )

        state = self.mapper._mapper.state
        observed = state.weight >= self.config.minimum_tsdf_weight
        if surface_only:
            threshold = self.config.voxel_size if sdf_threshold is None else float(sdf_threshold)
            if not math.isfinite(threshold) or threshold < 0:
                raise ValueError("sdf_threshold must be finite and non-negative")
            candidate_mask = observed & (state.tsdf.abs() <= threshold / self.config.truncation_distance)
        else:
            candidate_mask = observed & (state.tsdf <= 0)
        coordinates = torch.nonzero(candidate_mask[0], as_tuple=False)
        dtype = state.tsdf.dtype
        actual_shape = state.tsdf.shape[1:]
        centers = (coordinates.to(dtype) - (state.tsdf.new_tensor(actual_shape) - 1) / 2)
        centers = centers * self.config.voxel_size + state.tsdf.new_tensor(self.mapper._mapper.config.grid_center)
        flat_indices = (coordinates[:, 0] * actual_shape[1] * actual_shape[2]
                        + coordinates[:, 1] * actual_shape[2] + coordinates[:, 2]).to(torch.long)
        # Ask the existing map for its source-shaped BlockDataView, then
        # replace only its candidates with the correct TSDF observation mask.
        view = self.mapper.extract_occupied_voxels().block_data
        if max_points is not None and len(centers) * subvoxel_factor ** 3 > max_points:
            source_count = max_points // subvoxel_factor ** 3
            if source_count == 0:
                centers, flat_indices = centers[:0], flat_indices[:0]
            else:
                selected = torch.linspace(0, len(centers) - 1, steps=source_count,
                                          device=centers.device).round().to(torch.long)
                centers, flat_indices = centers[selected], flat_indices[selected]
        if subvoxel_factor > 1 and len(centers):
            axis = ((torch.arange(subvoxel_factor, device=centers.device, dtype=dtype) + 0.5)
                    / subvoxel_factor - 0.5) * self.config.voxel_size
            offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            centers = (centers[:, None] + offsets[None]).reshape(-1, 3)
            flat_indices = flat_indices.repeat_interleave(len(offsets))
        voxels = OccupiedVoxels(centers, flat_indices, view, subvoxel_factor=subvoxel_factor)
        if texture_observations is None:
            return voxels
        projector = self._texture_projector or self._get_texture_projector(texture_observations)
        return projector.texture_occupied_voxels(
            voxels, texture_observations, camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance,
            texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> MatchedVoxels:
        if getattr(self._tsdf, "_portable_sparse", False):
            return self._extract_sparse_matching_feature_voxels(
                feature_vector,
                top_k,
                surface_only=surface_only,
                sdf_threshold=sdf_threshold,
                minimum_score=minimum_score,
                feature_projector=feature_projector,
            )
        return self.mapper.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector,
        )

    def _sparse_block_data_view(self) -> BlockDataView:
        """Return the source-shaped read-only view over portable block storage."""
        data = self._tsdf.data
        return BlockDataView(
            rgb_grid=data.block_grid_rgb,
            coords=data.block_coords,
            num_allocated=int(data.num_allocated.item()),
            origin=data.origin,
            voxel_size=float(data.voxel_size),
            block_size=int(data.block_size),
            grid_shape=tuple(data.grid_shape),
            color_grid_size=int(data.color_grid_size),
            feature_block_grid_size=int(data.feature_block_grid_size),
            features=data.block_features,
            feature_weight=data.block_feature_weight,
            feature_dim=int(data.feature_dim),
        )

    def _extract_sparse_matched_voxels(
        self,
        block_pool_idx: torch.Tensor,
        *,
        surface_only: bool,
        sdf_threshold: Optional[float],
    ) -> OccupiedVoxels:
        """Materialize observed voxels in selected portable sparse blocks."""
        data = self._tsdf.data
        view = self._sparse_block_data_view()
        device = self._tsdf.device
        if block_pool_idx.numel() == 0:
            return OccupiedVoxels(
                torch.empty((0, 3), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.int32, device=device),
                view,
            )

        pools = block_pool_idx.to(device=device, dtype=torch.long)
        dynamic = data.block_data[pools].float()
        dynamic_weight = dynamic[..., 1]
        valid_dynamic = dynamic_weight >= float(self.config.minimum_tsdf_weight)
        dynamic_sdf = dynamic[..., 0] / dynamic_weight.clamp_min(1.0e-6)

        valid = valid_dynamic
        sdf = torch.where(valid_dynamic, dynamic_sdf, torch.full_like(dynamic_sdf, float("inf")))
        if data.has_static:
            static_sdf = data.static_block_data[pools].float()
            valid_static = static_sdf < 1.0e9
            sdf = torch.minimum(sdf, static_sdf)
            valid = valid | valid_static

        threshold = self.config.voxel_size if sdf_threshold is None else float(sdf_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("sdf_threshold must be finite and non-negative")
        # ``block_data[..., 0] / [..., 1]`` is the normalized TSDF value,
        # while the public threshold is expressed in metres.
        normalized_threshold = threshold / float(data.truncation_distance)
        keep = valid & (
            (sdf.abs() < normalized_threshold) if surface_only else (sdf <= 0.0)
        )
        block_rank, local_idx = torch.nonzero(keep, as_tuple=True)
        if block_rank.numel() == 0:
            return OccupiedVoxels(
                torch.empty((0, 3), dtype=torch.float32, device=device),
                torch.empty((0,), dtype=torch.int32, device=device),
                view,
            )

        block_size = int(data.block_size)
        # Portable integration stores its local lattice in torch.meshgrid
        # order (z changes fastest), so decode the flattened slot likewise.
        local_x = torch.div(local_idx, block_size * block_size, rounding_mode="floor")
        remainder = local_idx.remainder(block_size * block_size)
        local_y = torch.div(remainder, block_size, rounding_mode="floor")
        local_z = remainder.remainder(block_size)
        local = torch.stack((local_x, local_y, local_z), dim=-1).float()
        block_coords = data.block_coords.view(-1, 3)[pools[block_rank]].float()
        blocks_xyz = torch.tensor(
            [
                math.ceil(int(data.grid_shape[2]) / block_size),
                math.ceil(int(data.grid_shape[1]) / block_size),
                math.ceil(int(data.grid_shape[0]) / block_size),
            ],
            dtype=torch.float32,
            device=device,
        )
        voxel_xyz = (block_coords + torch.floor(blocks_xyz * 0.5)) * block_size + local
        center_offset = torch.tensor(
            [int(data.grid_shape[2]), int(data.grid_shape[1]), int(data.grid_shape[0])],
            dtype=torch.float32, device=device,
        ) * 0.5
        centers = data.origin.to(device=device, dtype=torch.float32) + (
            voxel_xyz + 0.5 - center_offset
        ) * float(data.voxel_size)
        return OccupiedVoxels(
            centers,
            pools[block_rank].to(torch.int32),
            view,
        )

    def _extract_sparse_matching_feature_voxels(
        self,
        feature_vector: torch.Tensor,
        top_k: int,
        *,
        surface_only: bool,
        sdf_threshold: Optional[float],
        minimum_score: Optional[float],
        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> MatchedVoxels:
        """PyTorch implementation of upstream per-block cosine matching."""
        data = self._tsdf.data
        device = self._tsdf.device
        if not data.has_features:
            raise RuntimeError("extract_matching_feature_voxels() requires feature_dim > 0")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        query = feature_vector.to(device=device, dtype=torch.float32)
        if query.ndim != 1:
            raise ValueError(f"feature_vector must be shape (D,), got {tuple(query.shape)}")
        if feature_projector is None and query.shape[0] != data.feature_dim:
            raise ValueError(
                f"feature_vector must be shape ({data.feature_dim},), got {tuple(query.shape)}"
            )

        high_water = int(data.num_allocated.item())
        active = torch.nonzero(
            data.block_to_hash_slot[:high_water] >= 0, as_tuple=False
        ).flatten()
        empty_idx = torch.empty(0, dtype=torch.int32, device=device)
        empty_scores = torch.empty(0, dtype=torch.float32, device=device)
        if active.numel() == 0:
            return MatchedVoxels(
                self._extract_sparse_matched_voxels(
                    empty_idx, surface_only=surface_only, sdf_threshold=sdf_threshold
                ),
                empty_idx,
                empty_scores,
            )

        features = data.block_features[active].float().sum(dim=1)
        weights = data.block_feature_weight[active].float().sum(dim=1).clamp_min(1.0e-6)
        descriptors = features / weights.unsqueeze(-1)
        if feature_projector is not None:
            with torch.inference_mode():
                descriptors = feature_projector(descriptors)
            if (
                not isinstance(descriptors, torch.Tensor)
                or descriptors.ndim != 2
                or descriptors.shape[0] != active.numel()
            ):
                shape = getattr(descriptors, "shape", None)
                raise ValueError(
                    "feature_projector must return shape "
                    f"({active.numel()}, D), got {shape}"
                )
            descriptors = descriptors.to(device=device, dtype=torch.float32)
        if query.shape[0] != descriptors.shape[1]:
            raise ValueError(
                f"feature_vector must be shape ({descriptors.shape[1]},), got {tuple(query.shape)}"
            )

        query = torch.nn.functional.normalize(query, dim=0, eps=1.0e-6)
        descriptors = torch.nn.functional.normalize(descriptors, dim=1, eps=1.0e-6)
        scores = torch.nan_to_num(descriptors @ query, nan=-float("inf"))
        result = torch.topk(scores, min(top_k, int(active.numel())), sorted=True)
        pools = active[result.indices]
        selected_scores = result.values
        if minimum_score is not None:
            keep = selected_scores >= float(minimum_score)
            pools = pools[keep]
            selected_scores = selected_scores[keep]
        pools = pools.to(torch.int32)
        voxels = self._extract_sparse_matched_voxels(
            pools, surface_only=surface_only, sdf_threshold=sdf_threshold
        )
        return MatchedVoxels(voxels, pools, selected_scores.float())

    def get_matching_feature_voxels(
        self,
        feature_vector: torch.Tensor,
        top_k: int,
        surface_only: bool = False,
        sdf_threshold: Optional[float] = None,
        minimum_score: Optional[float] = None,
        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> MatchedVoxels:
        return self.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector
        )

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False) -> Dict[str, Any]:
        if getattr(self._tsdf, "_portable_sparse", False):
            stats = self._tsdf.get_stats(scan_pool=scan_pool, scan_hash=scan_hash)
            stats.update(
                {
                    "observed_voxels": int((self._tsdf.data.block_data[: int(self._tsdf.data.num_allocated.item()), :, 1] > 0).sum().item()),
                    "frame_count": self._frame_count,
                    "memory_mb": self.memory_usage_mb(),
                    "last_camera_integration": {
                        "implementation": "sparse_pytorch",
                        "profile_kernel_timings": self.config.profile_integration_kernel_timings,
                    },
                    "last_camera_integration_kernel_timings_ms": {},
                }
            )
            stats["last_integration"] = dict(stats["last_camera_integration"])
            stats["last_integration_kernel_timings_ms"] = {}
            if hasattr(self, "_last_lidar_visible"):
                stats["last_lidar_integration"] = {
                    "num_visible_blocks": self._last_lidar_visible,
                    "implementation": "sparse_pytorch",
                }
            return stats
        del scan_pool, scan_hash
        stats = self.mapper.get_stats()
        stats.update({
            "frame_count": self._frame_count,
            "memory_mb": self.mapper.memory_usage_mb(),
            "storage": "dense_portable",
            "last_camera_integration": {
                "implementation": "dense_pytorch",
                "profile_kernel_timings": self.config.profile_integration_kernel_timings,
            },
            "last_camera_integration_kernel_timings_ms": {},
        })
        stats["last_integration"] = dict(stats["last_camera_integration"])
        stats["last_integration_kernel_timings_ms"] = {}
        return stats

    def memory_usage_mb(self) -> float:
        if getattr(self._tsdf, "_portable_sparse", False):
            tensors = (
                value
                for value in vars(self._tsdf.data).values()
                if isinstance(value, torch.Tensor)
            )
            return sum(value.numel() * value.element_size() for value in tensors) / 2**20
        return self.mapper.memory_usage_mb()

    def update_static_obstacles(self, scene: SceneData, env_idx: int = 0, debug: bool = False) -> None:
        del debug
        return self.mapper.update_static_obstacles(scene, env_idx)


# These read-only portable convenience fields are intentionally installed at
# runtime; the pinned CUDA/ Warp class did not declare them.
BlockSparseTSDFIntegrator.voxel_size = property(BlockSparseTSDFIntegrator._voxel_size.fget)
BlockSparseTSDFIntegrator.origin = property(BlockSparseTSDFIntegrator._origin.fget)
BlockSparseTSDFIntegrator.truncation_distance = property(
    BlockSparseTSDFIntegrator._truncation_distance.fget
)

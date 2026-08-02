"""Portable high-level TSDF integration over the dense CPU/MPS mapper.

The public configuration intentionally accepts the V2 block-sparse options.
The map state is dense (and therefore has no Warp hash/pool ABI), while depth
fusion, mesh/voxel extraction, persistence, and region mutation are real
PyTorch operations on CPU or MPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

from .mapper import Mapper
from .mapper_cfg import MapperCfg
from .projector_texture import ProjectiveTextureProjector, ProjectiveTextureProjectorCfg
from .storage import OccupiedVoxels
from curobo_metal.ops.perception.core import dense_esdf


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
        if self.feature_dim:
            raise NotImplementedError("feature-volume integration requires Warp/CUDA")
        if self.lidar_num_sensors:
            raise NotImplementedError("LiDAR TSDF integration requires Warp/CUDA")
        if self.seeding_method not in {"gather", "scatter"}:
            raise ValueError("seeding_method must be 'gather' or 'scatter'")
        if self.feature_integration_kernel not in {"auto", "grouped", "tiled"}:
            raise ValueError("feature_integration_kernel must be 'auto', 'grouped', or 'tiled'")
        if self.num_cameras <= 0:
            raise ValueError("num_cameras must be positive")
        if self.max_support_pixels_per_block_camera <= 0 or self.max_feature_tile_channels <= 0:
            raise ValueError("portable camera scratch capacities must be positive")
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
        if self.max_visible_blocks_per_integration is None:
            self.max_visible_blocks_per_integration = self.max_blocks
        if not 0 < self.max_visible_blocks_per_integration <= self.max_blocks:
            raise ValueError("max_visible_blocks_per_integration must be in [1, max_blocks]")


class BlockSparseTSDFIntegrator:
    """cuRobo-shaped TSDF facade with a bounded dense portable implementation."""

    def __init__(self, config: BlockSparseTSDFIntegratorCfg, kernels=None):
        if kernels is not None:
            raise NotImplementedError("custom Warp block-sparse kernels are unavailable on CPU/MPS")
        self.config = config
        self.cfg = config  # historical portable spelling
        extent = tuple(int(n) * config.voxel_size for n in config.grid_shape)
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
        self._frame_count = 0

    @property
    def tsdf(self):
        return self.mapper.tsdf

    @property
    def voxel_size(self) -> float:
        """Metric spacing of the portable dense map.

        These source-shaped read-only configuration fields are useful to
        callers that receive an integrator rather than its ``.tsdf`` storage.
        They describe the actual dense map and do not imply a Warp block pool.
        """
        return self.config.voxel_size

    @property
    def origin(self) -> torch.Tensor:
        """World-space lower corner of the bounded dense map."""
        return self.config.origin.to(device=self.mapper.device)

    @property
    def truncation_distance(self) -> float:
        """Metric TSDF truncation distance."""
        return self.config.truncation_distance

    def reset(self):
        self._frame_count = 0
        return self.mapper.reset()

    def import_blocks(self, blocks):
        if isinstance(blocks, (str, bytes)) or hasattr(blocks, "__fspath__"):
            result = self.mapper.import_blocks(blocks)
        elif isinstance(blocks, dict):
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

    def integrate(self, observation=None, *, camera_observation=None, lidar_observation=None):
        if observation is not None and (camera_observation is not None or lidar_observation is not None):
            raise ValueError("observation cannot be combined with camera_observation or lidar_observation")
        if observation is None and camera_observation is None and lidar_observation is None:
            raise ValueError("integrate() requires observation, camera_observation, or lidar_observation")
        if lidar_observation is not None:
            raise NotImplementedError("LiDAR TSDF integration requires Warp/CUDA")
        # The portable mapper has no projective frustum-recycling kernel, but
        # global temporal decay is meaningful and can be implemented exactly
        # over its dense state before each camera update.
        selected_observation = observation if observation is not None else camera_observation
        self._apply_frame_decay(camera_observation=selected_observation)
        result = self.mapper.integrate(
            observation=observation, camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        self._frame_count += 1
        return result

    def _integrate_camera_frame(self, observation, *, advance_frame=True, apply_decay=True):
        if apply_decay:
            self._apply_frame_decay(camera_observation=observation)
        result = self.mapper.integrate(camera_observation=observation)
        if advance_frame:
            self._frame_count += 1
        return result

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
        del observation
        raise NotImplementedError("LiDAR TSDF integration requires Warp/CUDA")

    def recycle_empty_blocks(self):
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

    def clear_region(self, bounds_min, bounds_max):
        return self.mapper.clear_region(bounds_min, bounds_max)

    def clear_blocks(self, pool_indices):
        return self.mapper.clear_blocks(pool_indices)

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = False, level: float = 0.0):
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
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

    def extract_mesh_tensors(self, level: float = 0.0, surface_only: bool = False, refine_iterations: int = 0):
        mesh = self.extract_mesh(refine_iterations=refine_iterations, surface_only=surface_only, level=level)
        vertices = torch.as_tensor(mesh.vertices)
        normals = torch.as_tensor(mesh.vertex_normals, device=vertices.device, dtype=vertices.dtype)
        colors = (torch.as_tensor(mesh.vertex_colors, device=vertices.device, dtype=vertices.dtype)
                  .clamp(0, 1) * 255).to(torch.uint8)
        return vertices, torch.as_tensor(mesh.faces, device=vertices.device, dtype=torch.int32), normals, colors

    def _texture_projector(self, texture_observations) -> ProjectiveTextureProjector:
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
        return ProjectiveTextureProjector(
            self._tsdf,
            self.mapper,
            ProjectiveTextureProjectorCfg(
                texture_num_cameras=expected_cameras,
                image_height=self.config.texture_camera_image_height or height,
                image_width=self.config.texture_camera_image_width or width,
                depth_minimum_distance=self.config.depth_minimum_distance,
                depth_maximum_distance=self.config.depth_maximum_distance,
                voxel_size=self.config.voxel_size,
            ),
        )

    def extract_textured_mesh(self, texture_observations, refine_iterations: int = 0,
                              surface_only: bool = True, level: float = 0.0,
                              camera_min_distance: Optional[float] = None,
                              camera_max_distance: Optional[float] = None,
                              texture_depth_tolerance_m: Optional[float] = None):
        if level != 0.0:
            raise NotImplementedError("portable dense mesh extraction supports only the zero TSDF level")
        vertices, faces, normals, colors = self.extract_mesh_tensors(
            level=level, surface_only=surface_only, refine_iterations=refine_iterations,
        )
        projector = self._texture_projector(texture_observations)
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

    def extract_surface_voxels(self, sdf_threshold: float = None):
        """Return ``(centers, colors, signed_distances_m)`` near the surface.

        This differs intentionally from :meth:`extract_occupied_voxels`: it
        exposes *observed* cells on both sides of the zero crossing, matching
        the source debug/export API.  Geometry-only maps return deterministic
        neutral colors because they have no RGB accumulator.
        """
        threshold = self.config.truncation_distance if sdf_threshold is None else float(sdf_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("sdf_threshold must be finite and non-negative")
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
                                texture_observations=None, camera_min_distance=None,
                                camera_max_distance=None, texture_depth_tolerance_m=None):
        subvoxel_factor = self._validate_subvoxel_factor(subvoxel_factor)
        max_points = self._validate_max_points(max_points)
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
        return self._texture_projector(texture_observations).texture_occupied_voxels(
            voxels, texture_observations, camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance,
            texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        return self.mapper.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector,
        )

    get_matching_feature_voxels = extract_matching_feature_voxels

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False):
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

    def memory_usage_mb(self):
        return self.mapper.memory_usage_mb()

    def update_static_obstacles(self, scene, env_idx: int = 0, debug: bool = False):
        del debug
        return self.mapper.update_static_obstacles(scene, env_idx)

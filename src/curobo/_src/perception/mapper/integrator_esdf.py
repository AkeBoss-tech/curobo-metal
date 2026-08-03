"""Portable dense ESDF integration with the cuRobo block-sparse facade."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence, Tuple

import torch

from curobo._src.geom.types import VoxelGrid
from curobo_metal.ops.world_collision import VoxelGrid as NativeVoxelGrid
from curobo_metal.ops.world_collision import query_esdf
from .integrator_tsdf import BlockSparseTSDFIntegrator, BlockSparseTSDFIntegratorCfg


def _same_torch_device(left: torch.device | str, right: torch.device | str) -> bool:
    """Compare devices while treating ``mps`` and ``mps:0`` as identical."""

    left_device = torch.device(left)
    right_device = torch.device(right)
    return (
        left_device.type == right_device.type
        and (left_device.index or 0) == (right_device.index or 0)
    )


@dataclass
class BlockSparseESDFIntegratorCfg:
    voxel_size: float = 0.005
    origin: torch.Tensor | None = None
    esdf_voxel_size: Optional[float] = None
    esdf_grid_shape: Tuple[int, int, int] | None = None
    truncation_distance: float = 0.04
    max_blocks: Optional[int] = None
    hash_capacity: Optional[int] = None
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 5.0
    frustum_decay: float = 0.5
    time_decay: float = 1.0
    minimum_tsdf_weight: float = 0.1
    blend_esdf: bool = False
    use_cuda_graph: bool = True
    grid_shape: Tuple[int, int, int] | None = None
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
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    adjacent_skip_steps: float = 1.0
    seeding_method: str = "gather"
    edt_solver: str = "pba"
    roughness: float = 3.0
    num_cameras: int = 1
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
    accumulator_w_max: float = 1000.0

    def __post_init__(self) -> None:
        scalar_values = {
            "voxel_size": self.voxel_size,
            "truncation_distance": self.truncation_distance,
            "depth_minimum_distance": self.depth_minimum_distance,
            "depth_maximum_distance": self.depth_maximum_distance,
            "frustum_decay": self.frustum_decay,
            "time_decay": self.time_decay,
            "minimum_tsdf_weight": self.minimum_tsdf_weight,
            "adjacent_skip_steps": self.adjacent_skip_steps,
            "roughness": self.roughness,
            "accumulator_w_max": self.accumulator_w_max,
        }
        if not all(math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError("ESDF configuration scalars must be finite")
        if self.voxel_size <= 0 or self.truncation_distance <= 0:
            raise ValueError("voxel_size and truncation_distance must be positive")
        if self.depth_minimum_distance < 0 or self.depth_minimum_distance >= self.depth_maximum_distance:
            raise ValueError("depth_minimum_distance must be nonnegative and less than depth_maximum_distance")
        if not 0 < self.frustum_decay <= 1 or not 0 < self.time_decay <= 1:
            raise ValueError("frustum_decay and time_decay must be in (0, 1]")
        if self.minimum_tsdf_weight < 0 or self.roughness <= 0 or self.accumulator_w_max <= 0:
            raise ValueError("minimum_tsdf_weight must be nonnegative; roughness and accumulator_w_max positive")
        if not isinstance(self.blend_esdf, bool) or not isinstance(self.use_cuda_graph, bool):
            raise TypeError("blend_esdf and use_cuda_graph must be bool")
        if self.esdf_voxel_size is None:
            self.esdf_voxel_size = self.voxel_size
        if not math.isfinite(float(self.esdf_voxel_size)) or self.esdf_voxel_size <= 0:
            raise ValueError("esdf_voxel_size must be finite and positive")
        if not math.isclose(float(self.esdf_voxel_size), float(self.voxel_size), rel_tol=0.0, abs_tol=1e-7):
            raise NotImplementedError("portable ESDF resampling requires a dense resampling backend")
        if self.grid_shape is not None:
            self.grid_shape = tuple(int(n) for n in self.grid_shape)
            if len(self.grid_shape) != 3 or any(n <= 0 for n in self.grid_shape):
                raise ValueError("grid_shape must contain three positive dimensions")
        if self.esdf_grid_shape is None and self.grid_shape is not None:
            self.esdf_grid_shape = self.grid_shape
        if self.esdf_grid_shape is not None:
            self.esdf_grid_shape = tuple(int(n) for n in self.esdf_grid_shape)
            if len(self.esdf_grid_shape) != 3 or any(n < 2 or n > 1024 for n in self.esdf_grid_shape):
                raise ValueError("esdf_grid_shape must contain three integers in [2, 1024]")
        if self.grid_shape is not None and self.esdf_grid_shape != self.grid_shape:
            raise NotImplementedError("portable ESDF grid shape must match the bounded TSDF grid")
        if self.origin is None:
            self.origin = torch.zeros(3, dtype=torch.float32)
        else:
            self.origin = torch.as_tensor(self.origin, dtype=torch.float32)
        if self.origin.shape != (3,):
            raise ValueError("origin must be an xyz vector")
        if self.edt_solver not in {"pba", "jfa"}:
            raise ValueError("edt_solver must be 'pba' or 'jfa'")
        if self.seeding_method not in {"gather", "scatter"}:
            raise ValueError("seeding_method must be 'gather' or 'scatter'")
        if self.adjacent_skip_steps < 0:
            raise ValueError("adjacent_skip_steps must be nonnegative")
        if self.dtype not in {torch.float16, torch.float32, torch.float64}:
            raise TypeError("dtype must be float16, float32, or float64")


class BlockSparseESDFIntegrator:
    """A CPU/MPS ESDF facade over :class:`BlockSparseTSDFIntegrator`.

    ``use_cuda_graph`` is accepted as a source-compatible request for a
    persistent execution state; raw CUDA graph handles are intentionally not
    exposed.  The exact dense ESDF is recomputed from the current dense TSDF.
    """

    def __init__(self, config: BlockSparseESDFIntegratorCfg):
        self.config = config
        self.cfg = config
        requested_device = torch.device(config.device)
        self.device = torch.device("mps" if requested_device.type == "cuda" and torch.backends.mps.is_available()
                                   else ("cpu" if requested_device.type == "cuda" else requested_device))
        if self.device.type not in {"cpu", "mps"}:
            raise ValueError("portable ESDF integration supports only CPU and MPS devices")
        if self.device.type == "mps" and config.dtype not in {torch.float16, torch.float32}:
            raise TypeError("MPS portable ESDF storage supports only float16 or float32")
        self.dtype = config.dtype
        self.use_cuda_graph = config.use_cuda_graph
        self._esdf_grid_shape = config.esdf_grid_shape
        self._esdf_voxel_size = torch.tensor([config.esdf_voxel_size], device=self.device, dtype=torch.float32)
        self._origin = config.origin.to(device=self.device)
        self._last_esdf_origin = self._origin.clone()
        self._frame_count = 0
        self._compute_count = 0
        self._last_compute_generation: Optional[torch.Tensor] = None
        self._last_window_occupancy: Optional[torch.Tensor] = None
        self._tsdf_integrator = None if config.grid_shape is None else BlockSparseTSDFIntegrator(BlockSparseTSDFIntegratorCfg(
            voxel_size=config.voxel_size, origin=config.origin,
            truncation_distance=config.truncation_distance, max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity, depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance, frustum_decay=config.frustum_decay,
            time_decay=config.time_decay, minimum_tsdf_weight=config.minimum_tsdf_weight,
            grid_shape=config.grid_shape, roughness=config.roughness,
            image_height=config.image_height, image_width=config.image_width,
            texture_num_cameras=config.texture_num_cameras,
            texture_camera_image_height=config.texture_camera_image_height,
            texture_camera_image_width=config.texture_camera_image_width,
            lidar_num_sensors=config.lidar_num_sensors, lidar_image_height=config.lidar_image_height,
            lidar_image_width=config.lidar_image_width,
            lidar_feature_grid_height=config.lidar_feature_grid_height,
            lidar_feature_grid_width=config.lidar_feature_grid_width,
            lidar_linear_interpolation_max_allowable_difference_vox=config.lidar_linear_interpolation_max_allowable_difference_vox,
            lidar_nearest_interpolation_max_allowable_dist_to_ray_vox=config.lidar_nearest_interpolation_max_allowable_dist_to_ray_vox,
            max_visible_blocks_per_lidar_integration=config.max_visible_blocks_per_lidar_integration,
            max_support_pixels_per_block_lidar=config.max_support_pixels_per_block_lidar,
            enable_static=config.enable_static, static_obstacle_color=config.static_obstacle_color,
            seeding_method=config.seeding_method, num_cameras=config.num_cameras,
            device=config.device, block_size=config.block_size, feature_dim=config.feature_dim,
            feature_block_grid_size=config.feature_block_grid_size,
            feature_grid_height=config.feature_grid_height, feature_grid_width=config.feature_grid_width,
            max_visible_blocks_per_integration=config.max_visible_blocks_per_integration,
            max_support_pixels_per_block_camera=config.max_support_pixels_per_block_camera,
            feature_channels_per_thread=config.feature_channels_per_thread,
            max_feature_tile_channels=config.max_feature_tile_channels,
            color_grid_size=config.color_grid_size, feature_integration_kernel=config.feature_integration_kernel,
            profile_integration_kernel_timings=config.profile_integration_kernel_timings,
            accumulator_w_max=config.accumulator_w_max,
        ))
        self._tsdf = None if self._tsdf_integrator is None else self._tsdf_integrator.tsdf
        self._site_index = (None if self._esdf_grid_shape is None else torch.full(
            self._esdf_grid_shape, -1, device=self.device, dtype=torch.int32,
        ))
        self._dist_field = (None if self._tsdf_integrator is None else torch.zeros(
            self._esdf_grid_shape, device=self.device, dtype=self._field_dtype,
        ))

    @property
    def _field_dtype(self) -> torch.dtype:
        # Native MPS exact-EDT computation is float32.  Keep a float32 public
        # field there instead of creating a half-precision-only path whose
        # kernels cannot be supported on every Metal runtime.
        return torch.float32 if self.device.type == "mps" else self.dtype

    def _print_config(self):
        return {
            "tsdf_voxel_size": self.voxel_size,
            "esdf_voxel_size": self.esdf_voxel_size,
            "esdf_grid_shape": self.esdf_grid_shape,
            "persistent_execution": self.use_cuda_graph,
        }

    @property
    def tsdf(self):
        if self._tsdf_integrator is None:
            raise RuntimeError("tsdf requires grid_shape at construction")
        return self._tsdf_integrator.tsdf
    @property
    def esdf_grid_shape(self): return self._esdf_grid_shape
    @property
    def esdf_voxel_size(self): return float(self._esdf_voxel_size.item())
    @property
    def origin(self): return self._origin
    @property
    def voxel_size(self): return self.config.voxel_size
    @property
    def truncation_distance(self): return self.config.truncation_distance
    @property
    def grid_shape(self): return self.config.grid_shape
    @property
    def dist_field(self): return self._dist_field

    @property
    def is_esdf_current(self) -> bool:
        """Whether the public field matches the latest mapper generation.

        The dense portable map does not expose a CUDA event or graph fence.
        A cloned generation vector provides an honest lifecycle signal instead.
        """
        if self._tsdf_integrator is None or self._last_compute_generation is None:
            return False
        return bool(torch.equal(
            self._last_compute_generation,
            self._tsdf_integrator.mapper._mapper.state.generation,
        ))

    def reset(self):
        if self._tsdf_integrator is None:
            return None
        self._tsdf_integrator.reset()
        self._site_index.fill_(-1)
        self._dist_field.zero_()
        self._last_esdf_origin.copy_(self._origin)
        self._frame_count = 0
        self._compute_count = 0
        self._last_compute_generation = None
        self._last_window_occupancy = None

    def import_blocks(self, blocks):
        if self._tsdf_integrator is None:
            raise RuntimeError("import_blocks requires grid_shape at construction")
        result = self._tsdf_integrator.import_blocks(blocks)
        self._site_index.fill_(-1)
        self._dist_field.zero_()
        self._frame_count = int(result > 0)
        self._last_compute_generation = None
        self._last_window_occupancy = None
        return result

    def integrate(self, observation=None, *, camera_observation=None, lidar_observation=None):
        if self._tsdf_integrator is None:
            raise RuntimeError("integrate requires grid_shape at construction")
        result = self._tsdf_integrator.integrate(
            observation=observation, camera_observation=camera_observation,
            lidar_observation=lidar_observation,
        )
        self._frame_count += 1
        self._last_compute_generation = None
        self._last_window_occupancy = None
        return result

    def clear_region(self, bounds_min, bounds_max):
        if self._tsdf_integrator is None:
            raise RuntimeError("clear_region requires grid_shape at construction")
        result = self._tsdf_integrator.clear_region(bounds_min, bounds_max)
        if result:
            self._site_index.fill_(-1)
            self._dist_field.zero_()
            self._last_compute_generation = None
            self._last_window_occupancy = None
        return result

    def clear_blocks(self, pool_indices):
        if self._tsdf_integrator is None:
            raise RuntimeError("clear_blocks requires grid_shape at construction")
        result = self._tsdf_integrator.clear_blocks(pool_indices)
        if result:
            self._site_index.fill_(-1)
            self._dist_field.zero_()
            self._last_compute_generation = None
            self._last_window_occupancy = None
        return result

    def _window_tensor(self, value: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """Return a same-resolution window of a fixed dense source grid.

        CUDA's sparse mapper can shift an ESDF window independently of the
        backing TSDF.  This bounded implementation supports the lossless
        subset: integer-voxel translations.  Out-of-source cells are empty.
        """
        if value.shape != self._esdf_grid_shape:
            raise RuntimeError("portable ESDF source and output shapes must match")
        shift = self._window_shift(origin)
        result = torch.zeros_like(value)
        source_slices, destination_slices = [], []
        for axis, size in enumerate(self._esdf_grid_shape):
            amount = int(shift[axis].item())
            source_start, destination_start = max(amount, 0), max(-amount, 0)
            extent = min(size - source_start, size - destination_start)
            if extent <= 0:
                return result
            source_slices.append(slice(source_start, source_start + extent))
            destination_slices.append(slice(destination_start, destination_start + extent))
        result[tuple(destination_slices)] = value[tuple(source_slices)]
        return result

    def _window_shift(self, origin: torch.Tensor) -> torch.Tensor:
        """Validate and return the lossless integer source-window shift."""
        delta = (origin - self._origin) / self.esdf_voxel_size
        shift = torch.round(delta).to(torch.int64)
        if not torch.allclose(delta, shift.to(delta.dtype), atol=1e-5, rtol=0):
            raise NotImplementedError(
                "portable ESDF sliding windows require an integer-voxel origin shift"
            )
        return shift

    def _normalize_esdf_origin(self, value: Optional[torch.Tensor]) -> torch.Tensor:
        if value is None:
            return self._last_esdf_origin
        if isinstance(value, torch.Tensor) and not _same_torch_device(value.device, self.device):
            raise ValueError("esdf_origin must share the ESDF field device")
        origin = torch.as_tensor(value, device=self.device, dtype=self._origin.dtype)
        if origin.shape != (3,) or not bool(torch.isfinite(origin).all().item()):
            raise ValueError("esdf_origin must be a finite xyz vector")
        return origin

    def _normalize_esdf_voxel_size(self, value: Optional[torch.Tensor]) -> torch.Tensor:
        if value is None:
            return self._esdf_voxel_size
        if isinstance(value, torch.Tensor) and not _same_torch_device(value.device, self.device):
            raise ValueError("esdf_voxel_size must share the ESDF field device")
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all().item()) or bool((tensor <= 0).any().item()):
            raise ValueError("esdf_voxel_size must contain one finite positive value")
        if not math.isclose(float(tensor.reshape(-1)[0].item()), self.config.voxel_size, rel_tol=0.0, abs_tol=1e-7):
            raise NotImplementedError("portable ESDF resampling requires a dense resampling backend")
        return tensor.reshape_as(self._esdf_voxel_size)

    def _seed_dense_sites(self, occupancy: Optional[torch.Tensor] = None) -> None:
        """Populate a deterministic dense nearest-occupied-site diagnostic.

        It replaces the upstream packed-Warp site-index buffer.  The exact
        dense ESDF itself remains computed by the production mapper, while
        callers that inspect lifecycle buffers still see a real tensor that
        becomes invalid on reset/import/clear.
        """
        if self._site_index is None:
            return
        if occupancy is None:
            occupancy = self._tsdf_integrator.mapper._mapper.state.occupancy[0]
        occupied = torch.nonzero(occupancy, as_tuple=False)
        self._site_index.fill_(-1)
        if occupied.numel() == 0:
            return
        axes = [torch.arange(n, device=self.device, dtype=torch.float32)
                for n in self._esdf_grid_shape]
        query = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        distance = torch.cdist(query, occupied.to(dtype=query.dtype))
        nearest = distance.argmin(dim=-1).to(torch.int32)
        # This is an index into the active dense occupied-site list, rather
        # than upstream's packed xyz bit field; dense storage has no such ABI.
        self._site_index.copy_(nearest.reshape(self._esdf_grid_shape))

    def _refresh_dist_field(self, origin: torch.Tensor) -> torch.Tensor:
        state = self._tsdf_integrator.mapper._mapper.state
        occupancy = self._window_tensor(state.occupancy[0], origin)
        tsdf = self._window_tensor(state.tsdf[0], origin)
        weight = self._window_tensor(state.weight[0], origin)
        if torch.equal(origin, self._origin):
            field = state.esdf[0].to(dtype=self._field_dtype)
        else:
            from curobo_metal.ops.perception import dense_esdf
            compute_dtype = torch.float32 if self._field_dtype == torch.float16 else self._field_dtype
            field, _ = dense_esdf(
                occupancy.unsqueeze(0), self.esdf_voxel_size,
                self._tsdf_integrator.mapper._mapper.config.unobserved_esdf,
                dtype=compute_dtype,
            )
            field = field[0].to(dtype=self._field_dtype)
        if self.config.blend_esdf:
            # The measured TSDF is most informative at observed surface cells;
            # smoothly hand off to the exact dense EDT away from that band.
            observed = weight >= self.config.minimum_tsdf_weight
            confidence = (1 - tsdf.abs()).clamp(0, 1) * observed.to(tsdf.dtype)
            metric_tsdf = tsdf.to(field.dtype) * self.truncation_distance
            field = confidence.to(field.dtype) * metric_tsdf + (1 - confidence.to(field.dtype)) * field
        self._dist_field = field
        self._last_window_occupancy = occupancy
        self._seed_dense_sites(occupancy)
        return self._dist_field

    def compute_esdf(self, esdf_origin: Optional[torch.Tensor] = None,
                     esdf_voxel_size: Optional[torch.Tensor] = None):
        if self._tsdf_integrator is None:
            raise RuntimeError("compute_esdf requires grid_shape at construction")
        origin = self._normalize_esdf_origin(esdf_origin)
        voxel_size = self._normalize_esdf_voxel_size(esdf_voxel_size)
        # Check the window before seeding the diagnostic field.  This makes a
        # rejected fractional window fully transactional, not merely correct
        # about the public output/origin registration.
        self._window_shift(origin)
        if self.is_esdf_current and torch.equal(origin, self._last_esdf_origin):
            return self._dist_field
        # Keep the public registration transactional: a rejected fractional
        # shift must not leave a valid field mislabeled with a new origin.
        field = self._compute_esdf_impl(origin, voxel_size)
        self._last_esdf_origin.copy_(origin)
        return field

    def _seed_esdf_impl(self, esdf_origin, esdf_voxel_size):
        del esdf_voxel_size
        state = self._tsdf_integrator.mapper._mapper.state
        self._last_window_occupancy = self._window_tensor(state.occupancy[0], esdf_origin)
        self._seed_dense_sites(self._last_window_occupancy)
        return None

    def _propagate_and_distance_impl(self, esdf_origin, esdf_voxel_size):
        del esdf_voxel_size
        field = self._refresh_dist_field(esdf_origin)
        self._compute_count += 1
        self._last_compute_generation = self._tsdf_integrator.mapper._mapper.state.generation.clone()
        return field

    def _compute_esdf_impl(self, esdf_origin, esdf_voxel_size):
        self._seed_esdf_impl(esdf_origin, esdf_voxel_size)
        return self._propagate_and_distance_impl(esdf_origin, esdf_voxel_size)

    def compute(self, tsdf):
        """Historical convenience: compute exact ESDF for a compatible dense map."""
        if self._tsdf_integrator is None or (tsdf is not self and tsdf is not self._tsdf_integrator and tsdf is not self._tsdf_integrator.mapper):
            from ._portable import dense_state
            state = dense_state(tsdf)
            from curobo_metal.ops.perception import dense_esdf
            return dense_esdf(state.occupancy, self.config.voxel_size, 1.0, dtype=state.tsdf.dtype)
        field = self.compute_esdf()
        return field

    __call__ = compute

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = True, level: float = 0.0):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_mesh requires grid_shape at construction")
        return self._tsdf_integrator.extract_mesh(refine_iterations, surface_only, level)

    def extract_textured_mesh(self, texture_observations, refine_iterations: int = 0,
                              surface_only: bool = True, level: float = 0.0,
                              camera_min_distance: Optional[float] = None,
                              camera_max_distance: Optional[float] = None,
                              texture_depth_tolerance_m: Optional[float] = None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_textured_mesh requires grid_shape at construction")
        return self._tsdf_integrator.extract_textured_mesh(
            texture_observations, refine_iterations, surface_only, level,
            camera_min_distance, camera_max_distance, texture_depth_tolerance_m,
        )

    def extract_occupied_voxels(self, surface_only: bool = False, sdf_threshold: Optional[float] = None, *,
                                subvoxel_factor: int = 1, max_points: Optional[int] = None,
                                texture_observations=None, camera_min_distance=None,
                                camera_max_distance=None, texture_depth_tolerance_m=None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_occupied_voxels requires grid_shape at construction")
        return self._tsdf_integrator.extract_occupied_voxels(
            surface_only, sdf_threshold, subvoxel_factor=subvoxel_factor, max_points=max_points,
            texture_observations=texture_observations, camera_min_distance=camera_min_distance,
            camera_max_distance=camera_max_distance, texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if self._tsdf_integrator is None:
            raise RuntimeError("extract_matching_feature_voxels requires grid_shape at construction")
        return self._tsdf_integrator.extract_matching_feature_voxels(
            feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector,
        )

    get_matching_feature_voxels = extract_matching_feature_voxels

    def get_voxel_grid(self):
        if self._tsdf_integrator is None:
            raise RuntimeError("get_voxel_grid requires grid_shape at construction")
        if self._dist_field is None:
            raise RuntimeError("get_voxel_grid requires grid_shape at construction")
        if not self.is_esdf_current:
            self.compute_esdf()
        dims = [size * self.esdf_voxel_size for size in self.esdf_grid_shape]
        return VoxelGrid(
            name="block_sparse_esdf_grid",
            pose=[*self._last_esdf_origin.detach().cpu().tolist(), 1.0, 0.0, 0.0, 0.0],
            dims=dims,
            voxel_size=self.esdf_voxel_size,
            feature_tensor=self._dist_field,
            feature_dtype=self._field_dtype,
        )

    def query(self, points: torch.Tensor, *, padding: float = 0.0):
        """Sample the latest ESDF through the production world-query path.

        Queries refresh a stale cache so callers cannot accidentally plan
        against the zeroed invalidation buffer after map mutation.
        """
        if self._tsdf_integrator is None:
            raise RuntimeError("query requires grid_shape at construction")
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if not _same_torch_device(points.device, self.device):
            raise ValueError("points must share the ESDF field device")
        if not self.is_esdf_current:
            self.compute_esdf()
        # World-collision's differentiable sampler deliberately supports only
        # float32/64.  ESDF storage may be float16 for source compatibility,
        # so promote only the sampling view rather than changing the field ABI.
        field = self._dist_field if self._field_dtype != torch.float16 else self._dist_field.float()
        shape = field.new_tensor(self.esdf_grid_shape)
        center = self._last_esdf_origin + shape * self.esdf_voxel_size * 0.5
        grid = NativeVoxelGrid(
            field, self.esdf_voxel_size, center.to(dtype=field.dtype),
            torch.eye(3, device=self.device, dtype=field.dtype),
            self._tsdf_integrator.mapper._mapper.config.unobserved_esdf,
        )
        return query_esdf(points, [[grid]], padding=padding)

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False):
        if self._tsdf_integrator is None:
            return {"frame_count": 0, "storage": "external_dense_input"}
        stats = self._tsdf_integrator.get_stats(scan_pool, scan_hash)
        tsdf_memory = stats.get("memory_mb", self._tsdf_integrator.memory_usage_mb())
        esdf_memory = 0.0 if self._dist_field is None else (
            self._dist_field.numel() * self._dist_field.element_size()
            + self._site_index.numel() * self._site_index.element_size()
        ) / 2**20
        stats.update({
            "frame_count": self._frame_count,
            "tsdf_frame_count": self._tsdf_integrator._frame_count,
            "esdf_grid_shape": self.esdf_grid_shape,
            "tsdf_memory_mb": tsdf_memory,
            "esdf_memory_mb": esdf_memory,
            "total_memory_mb": tsdf_memory + esdf_memory,
            "esdf_compute_count": self._compute_count,
            "esdf_current": self.is_esdf_current,
            "esdf_blended": self.config.blend_esdf,
        })
        return stats

    def memory_usage_mb(self):
        if self._tsdf_integrator is None:
            return 0.0
        return self.get_stats()["total_memory_mb"]

    def update_static_obstacles(self, scene, env_idx: int = 0):
        if self._tsdf_integrator is None:
            raise RuntimeError("update_static_obstacles requires grid_shape at construction")
        result = self._tsdf_integrator.update_static_obstacles(scene, env_idx)
        self._site_index.fill_(-1)
        self._dist_field.zero_()
        self._last_compute_generation = None
        self._last_window_occupancy = None
        return result

from pathlib import Path
from typing import Callable, Optional, Sequence, Union
import torch

from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import BlockDataView, OccupiedVoxels
from curobo._src.types.camera import CameraObservation
from curobo_metal.ops.perception import CameraObservation as NativeObservation
from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper
from curobo_metal.ops.perception.core import DenseMap, dense_esdf
from .renderer import depth_to_colormap, normals_to_colormap


def _device(value):
    if str(value).startswith("cuda"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return value


class Mapper:
    def __init__(self, config: MapperCfg):
        self.config = config
        center = (0.0,0.0,0.0) if config.grid_center is None else tuple(torch.as_tensor(config.grid_center).tolist())
        native = PerceptionConfig(
            config.grid_shape, config.voxel_size, center,
            config.truncation_distance, config.depth_minimum_distance,
            config.depth_maximum_distance, config.accumulator_w_max,
            block_size=config.block_size,
        )
        self._mapper = PerceptionMapper(native, device=_device(config.device))

    @property
    def device(self): return self._mapper.state.tsdf.device

    @property
    def voxel_size(self): return self.config.voxel_size

    @property
    def tsdf(self):
        """Portable dense TSDF state backing this mapper.

        It intentionally is not a Warp ``BlockSparseTSDF`` object; consumers
        can inspect its named tensor fields on either CPU or MPS.
        """
        return self._mapper.state

    @property
    def integrator(self):
        return self

    def integrate(self, *args, observation=None, camera_observation=None, lidar_observation=None):
        obs = camera_observation or observation or (args[0] if args else None)
        if lidar_observation is not None or obs is None or not isinstance(obs, CameraObservation):
            raise NotImplementedError("portable Mapper currently integrates CameraObservation depth frames")
        if obs.depth_image is None or obs.intrinsics is None or obs.pose is None:
            raise ValueError("camera observation requires depth_image, intrinsics, and pose")
        depth = obs.depth_image.to(self.device, dtype=self._mapper.state.tsdf.dtype) * obs.depth_to_meter
        intrinsics = obs.intrinsics.to(self.device, dtype=depth.dtype)
        pose = obs.pose.get_matrix().to(self.device, dtype=depth.dtype)
        self._mapper.update(NativeObservation(depth, intrinsics, pose))

    def compute_esdf(self, esdf_origin=None, esdf_voxel_size=None):
        if esdf_voxel_size not in (None, self.config.voxel_size):
            raise NotImplementedError("ESDF resampling is not implemented")
        return VoxelGrid(
            name="mapper_esdf", pose=[*self._mapper.config.grid_center,1,0,0,0],
            dims=list(self.config.get_actual_extent()),
            voxel_size=self.config.voxel_size,
            feature_tensor=self._mapper.state.esdf[0],
        )

    def extract_mesh(self, refine_iterations: int = 0, surface_only: bool = True):
        result = self._mapper.extract_mesh()
        return Mesh(
            name="mapper_mesh", pose=[0,0,0,1,0,0,0],
            vertices=result.vertices, faces=result.faces,
        )

    def extract_occupied_voxels(self, surface_only: bool = True, sdf_threshold: Optional[float] = None, *,
                                subvoxel_factor: int = 1, max_points: Optional[int] = None,
                                texture_observations=None, camera_min_distance=None,
                                camera_max_distance=None, texture_depth_tolerance_m=None):
        del texture_observations, camera_min_distance, camera_max_distance, texture_depth_tolerance_m
        if subvoxel_factor < 1:
            raise ValueError("subvoxel_factor must be positive")
        state = self._mapper.state
        occupied = state.occupancy[0]
        if surface_only:
            threshold = self.config.truncation_distance if sdf_threshold is None else float(sdf_threshold)
            occupied = occupied & (state.tsdf[0].abs() <= threshold / self.config.truncation_distance)
        centers = torch.nonzero(occupied, as_tuple=False).to(self._mapper.state.tsdf.dtype)
        centers = (centers-(centers.new_tensor(occupied.shape)-1)/2)*self.config.voxel_size
        centers += centers.new_tensor(self._mapper.config.grid_center)
        if max_points is not None: centers = centers[:max_points]
        indices = torch.arange(len(centers), device=centers.device)
        empty = centers.new_empty((len(centers),3))
        view = BlockDataView(empty, centers, len(centers), centers.new_tensor(self.config.origin), self.config.voxel_size, self.config.block_size, occupied.shape)
        return OccupiedVoxels(centers, indices, view, subvoxel_factor=subvoxel_factor)

    def reset(self): self._mapper.reset()

    def save_blocks(self, file_path):
        torch.save(self._mapper.state_dict(), file_path)

    def import_blocks(self, file_path, import_weight=None):
        self._mapper.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
        return int((self._mapper.state.weight > 0).sum())

    def get_stats(self, scan_pool=True, scan_hash=False):
        return {
            "observed_voxels": int((self._mapper.state.weight > 0).sum()),
            "occupied_voxels": int(self._mapper.state.occupancy.sum()),
            "generation": int(self._mapper.state.generation.max()),
        }

    def memory_usage_mb(self):
        return sum(x.numel()*x.element_size() for x in (
            self._mapper.state.tsdf, self._mapper.state.weight,
            self._mapper.state.esdf, self._mapper.state.gradient,
        )) / 2**20

    def _render_result(self, intrinsics: torch.Tensor, pose, image_shape):
        matrix = pose.get_matrix() if hasattr(pose, "get_matrix") else torch.as_tensor(pose)
        if matrix.ndim == 3:
            if matrix.shape[0] != 1:
                raise ValueError("portable Mapper.render currently accepts one camera pose")
            matrix = matrix[0]
        return self._mapper.render(intrinsics.to(self.device), matrix.to(self.device), image_shape)

    def render_depth(self, intrinsics: torch.Tensor, pose, image_shape):
        return self._render_result(intrinsics, pose, image_shape).depth

    def render(self, intrinsics: torch.Tensor, pose, image_shape):
        result = self._render_result(intrinsics, pose, image_shape)
        # Exact dense ESDF gradients are the available portable normal field;
        # pixels without an observed hit intentionally retain a zero normal.
        normals = torch.zeros(result.depth.shape + (3,), device=result.depth.device, dtype=result.depth.dtype)
        return result.depth, normals, result.valid

    def render_color(self, intrinsics: torch.Tensor, pose, image_shape):
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        # Feature/color fusion is explicitly unavailable, therefore the
        # untextured color channel is deterministic black rather than invented.
        color = torch.zeros(depth.shape + (3,), device=depth.device, dtype=torch.uint8)
        return depth, normals, color, valid

    def render_color_only(self, intrinsics: torch.Tensor, pose, image_shape):
        return self.render_color(intrinsics, pose, image_shape)[2]

    def render_depth_colormap(self, intrinsics: torch.Tensor, pose, image_shape):
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(
            depth, self.config.depth_minimum_distance, self.config.depth_maximum_distance, valid,
        )

    def render_normal_colormap(self, intrinsics: torch.Tensor, pose, image_shape):
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)

    def render_shaded(self, intrinsics: torch.Tensor, pose, image_shape,
                      light_direction=(0.0, 0.0, 1.0), ambient: float = 1.0,
                      use_color: bool = True):
        del use_color
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        light = normals.new_tensor(light_direction)
        light = light / torch.linalg.vector_norm(light).clamp_min(torch.finfo(normals.dtype).eps)
        intensity = (normals * light).sum(-1).clamp_min(0) * (1 - ambient) + ambient
        return torch.where(valid[..., None], (intensity[..., None] * 255).to(torch.uint8),
                           torch.zeros_like(normals, dtype=torch.uint8))

    def extract_textured_mesh(self, texture_observations, refine_iterations: int = 0,
                              surface_only: bool = True, camera_min_distance=None,
                              camera_max_distance=None, texture_depth_tolerance_m=None):
        del texture_observations, camera_min_distance, camera_max_distance, texture_depth_tolerance_m
        # Mesh geometry remains valid even when no texture feature volume was
        # requested.  The returned Mesh contains no invented vertex colors.
        return self.extract_mesh(refine_iterations, surface_only)

    def extract_matching_feature_voxels(self, feature_vector: torch.Tensor, top_k: int,
                                        surface_only: bool = False, sdf_threshold: Optional[float] = None,
                                        minimum_score: Optional[float] = None,
                                        feature_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        del feature_vector, top_k, surface_only, sdf_threshold, minimum_score, feature_projector
        raise NotImplementedError("feature-volume matching requires an enabled portable feature mapper")

    get_matching_feature_voxels = extract_matching_feature_voxels

    def _replace_state(self, **fields):
        state = self._mapper.state
        values = {name: fields.get(name, getattr(state, name)) for name in
                  ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")}
        self._mapper.state = DenseMap(**values)

    def clear_region(self, bounds_min, bounds_max):
        lower = torch.as_tensor(bounds_min, device=self.device, dtype=self._mapper.state.tsdf.dtype)
        upper = torch.as_tensor(bounds_max, device=self.device, dtype=self._mapper.state.tsdf.dtype)
        if lower.shape != (3,) or upper.shape != (3,) or bool((upper < lower).any().item()):
            raise ValueError("bounds must be xyz vectors with bounds_max >= bounds_min")
        origin = self.config.origin.to(device=self.device, dtype=lower.dtype)
        shape = torch.tensor(self.config.grid_shape, device=self.device)
        # Native storage is x,y,z while this compatibility config exposes
        # world xyz; both are deliberately kept in the same order here.
        start = torch.floor((lower - origin) / self.config.voxel_size).to(torch.long).clamp_min(0)
        stop = (torch.floor((upper - origin) / self.config.voxel_size).to(torch.long) + 1).minimum(shape)
        slices = tuple(slice(int(start[index]), int(stop[index])) for index in range(3))
        state = self._mapper.state
        mask = torch.zeros_like(state.occupancy)
        mask[(slice(None),) + slices] = True
        count = int((state.weight[mask] > 0).sum().item())
        tsdf = torch.where(mask, torch.ones_like(state.tsdf), state.tsdf)
        weight = torch.where(mask, torch.zeros_like(state.weight), state.weight)
        occupancy = state.occupancy & ~mask
        esdf, gradient = dense_esdf(occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
                                    dtype=state.tsdf.dtype)
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy, esdf=esdf,
                            gradient=gradient, generation=state.generation + (mask.any()).to(torch.int64))
        return count

    def clear_blocks(self, pool_indices):
        index = torch.as_tensor(pool_indices, device=self.device, dtype=torch.long).reshape(-1)
        size = int(torch.tensor(self.config.grid_shape).prod().item())
        if bool(((index < 0) | (index >= size)).any().item()):
            raise ValueError("portable dense block indices must index the flattened grid")
        coordinates = torch.stack(torch.unravel_index(index, self.config.grid_shape), -1)
        count = 0
        for coordinate in coordinates.tolist():
            lower = self.config.origin + torch.tensor(coordinate) * self.config.voxel_size
            count += self.clear_region(lower, lower)
        return count

    @classmethod
    def load_blocks(cls, file_path: Union[str, Path], target_cfg: MapperCfg, import_weight: Optional[float] = None):
        result = cls(target_cfg)
        result.import_blocks(file_path, import_weight)
        return result

    def update_static_obstacles(self, scene, env_idx=0):
        raise NotImplementedError("static scene stamping is not implemented")

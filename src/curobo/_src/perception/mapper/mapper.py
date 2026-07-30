from pathlib import Path
from typing import Optional
import torch

from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import BlockDataView, OccupiedVoxels
from curobo._src.types.camera import CameraObservation
from curobo_metal.ops.perception import CameraObservation as NativeObservation
from curobo_metal.ops.perception import PerceptionConfig, PerceptionMapper


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

    def extract_mesh(self, refine_iterations=0, surface_only=True):
        result = self._mapper.extract_mesh()
        return Mesh(
            name="mapper_mesh", pose=[0,0,0,1,0,0,0],
            vertices=result.vertices, faces=result.faces,
        )

    def extract_occupied_voxels(self, surface_only=True, sdf_threshold=None, *, subvoxel_factor=1, max_points=None, **kwargs):
        occupied = self._mapper.state.occupancy[0]
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

    def render_depth(self, intrinsics, pose, image_shape):
        return self._mapper.render(intrinsics.to(self.device), pose.get_matrix().to(self.device), image_shape).depth

    def render(self, intrinsics, pose, image_shape):
        result = self._mapper.render(intrinsics.to(self.device), pose.get_matrix().to(self.device), image_shape)
        normals = torch.zeros(result.depth.shape+(3,), device=result.depth.device, dtype=result.depth.dtype)
        return result.depth, normals, result.valid

    def clear_region(self, bounds_min, bounds_max):
        raise NotImplementedError("bounded region mutation is not implemented")

    def update_static_obstacles(self, scene, env_idx=0):
        raise NotImplementedError("static scene stamping is not implemented")

import math
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
import torch

from curobo._src.geom.types import Mesh, VoxelGrid
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.perception.mapper.storage import (
    BlockDataView,
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
    OccupiedVoxels,
)
from curobo._src.perception.mapper.checkpoint_blocks import (
    build_block_metadata,
    is_portable_dense_block_payload,
    load_block_checkpoint,
    prepare_blocks_for_import,
    save_block_checkpoint,
    validate_block_metadata_for_target,
)
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
        if config.feature_dim:
            raise NotImplementedError("MapperCfg.feature_dim requires CUDA/Warp feature-volume integration")
        if config.lidar_num_sensors:
            raise NotImplementedError("MapperCfg.lidar_num_sensors requires CUDA/Warp LiDAR integration")
        center = (0.0,0.0,0.0) if config.grid_center is None else tuple(torch.as_tensor(config.grid_center).tolist())
        native = PerceptionConfig(
            config.grid_shape, config.voxel_size, center,
            config.truncation_distance, config.depth_minimum_distance,
            config.depth_maximum_distance, config.accumulator_w_max,
            block_size=config.block_size,
        )
        self._mapper = PerceptionMapper(native, device=_device(config.device))
        self._storage = BlockSparseTSDF.from_native(BlockSparseTSDFCfg(
            max_blocks=config.max_blocks,
            hash_capacity=config.hash_capacity,
            voxel_size=config.voxel_size,
            origin=torch.as_tensor(center),
            truncation_distance=config.truncation_distance,
            device=config.device,
            grid_shape=config.grid_shape,
            enable_dynamic=True,
            enable_static=config.enable_static,
            static_obstacle_color=tuple(float(v) / 255.0 if float(v) > 1 else float(v)
                                        for v in config.static_obstacle_color),
            block_size=config.block_size,
            feature_dim=0,
            color_grid_size=config.color_grid_size,
            accumulator_w_max=config.accumulator_w_max,
        ), self._mapper)
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)

    @property
    def device(self): return self._mapper.state.tsdf.device

    @property
    def voxel_size(self): return self.config.voxel_size

    @property
    def tsdf(self):
        """Dense-backed :class:`BlockSparseTSDF` lifecycle facade.

        The returned object is intentionally not a Warp hash table.  Its
        ``state`` property is the actual dense PyTorch map on CPU or MPS.
        """
        return self._storage

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
        if esdf_origin is not None:
            expected = torch.as_tensor(self._mapper.config.grid_center, device=self.device,
                                       dtype=self._mapper.state.tsdf.dtype)
            supplied = torch.as_tensor(esdf_origin, device=self.device, dtype=expected.dtype)
            if supplied.shape != (3,):
                raise ValueError("esdf_origin must be an xyz vector")
            if not torch.allclose(supplied, expected):
                raise NotImplementedError("portable dense Mapper does not implement sliding ESDF windows")
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
        coordinates = torch.nonzero(occupied, as_tuple=False)
        centers = coordinates.to(self._mapper.state.tsdf.dtype)
        centers = (centers-(centers.new_tensor(occupied.shape)-1)/2)*self.config.voxel_size
        centers += centers.new_tensor(self._mapper.config.grid_center)
        # The source contract uses evenly spaced subvoxel centers.  Cap source
        # voxels before expansion so every retained source voxel remains
        # complete and the result is deterministic.
        source_limit = None
        if max_points is not None:
            source_limit = max_points // (subvoxel_factor ** 3)
            coordinates, centers = coordinates[:source_limit], centers[:source_limit]
        if subvoxel_factor > 1 and len(centers):
            axis = ((torch.arange(subvoxel_factor, device=centers.device, dtype=centers.dtype) + 0.5)
                    / subvoxel_factor - 0.5) * self.config.voxel_size
            offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            centers = (centers[:, None] + offsets[None]).reshape(-1, 3)
            coordinates = coordinates[:, None].expand(-1, len(offsets), -1).reshape(-1, 3)
        shape = tuple(int(v) for v in occupied.shape)
        indices = (coordinates[:, 0] * shape[1] * shape[2] + coordinates[:, 1] * shape[2] + coordinates[:, 2]).to(torch.long)
        total = int(math.prod(shape))
        coords = torch.stack(torch.meshgrid(
            *[torch.arange(v, device=centers.device, dtype=torch.int32) for v in shape], indexing="ij",
        ), -1).reshape(-1, 3)
        rgb = torch.zeros((total, 1, 4), device=centers.device, dtype=centers.dtype)
        features = centers.new_empty((total, 1, 0))
        feature_weight = centers.new_zeros((total, 1))
        view = BlockDataView(rgb, coords, total, centers.new_tensor(self._mapper.config.grid_center),
                             self.config.voxel_size, 1, shape, features=features,
                             feature_weight=feature_weight)
        return OccupiedVoxels(centers, indices, view, subvoxel_factor=subvoxel_factor)

    def reset(self):
        self._storage.reset()
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)

    def save_blocks(self, file_path):
        """Persist dense CPU/MPS state in the cuRobo checkpoint envelope."""
        blocks = {
            field: getattr(self._mapper.state, field)
            for field in ("tsdf", "weight", "occupancy", "esdf", "gradient", "generation")
        }
        save_block_checkpoint(file_path, build_block_metadata(self.tsdf), blocks)

    def import_blocks(self, file_path, import_weight=None):
        """Restore a dense checkpoint into an empty mapper.

        A V2 CUDA/Warp sparse block-pool payload is identified and validated
        by the checkpoint helpers, then rejected explicitly rather than being
        misinterpreted as portable dense state.
        """
        checkpoint = load_block_checkpoint(file_path)
        validate_block_metadata_for_target(checkpoint["block_metadata"], self.tsdf)
        blocks = prepare_blocks_for_import(
            checkpoint["blocks"], checkpoint["block_metadata"],
            import_weight=import_weight,
            minimum_tsdf_weight=self.config.minimum_tsdf_weight,
            block_empty_threshold=0.0,
        )
        if not is_portable_dense_block_payload(blocks):
            raise NotImplementedError("importing CUDA/Warp sparse block-pool payloads is unavailable on the portable dense mapper")
        if bool((self._mapper.state.weight > 0).any().item()):
            raise ValueError("block import requires an empty target mapper")
        state_checkpoint = self._mapper.state_dict()
        state_checkpoint.update(blocks)
        self._storage.import_blocks(state_checkpoint)
        self._static_mask = torch.zeros_like(self._mapper.state.occupancy)
        return int((self._mapper.state.weight > 0).sum().item())

    def get_stats(self, scan_pool=True, scan_hash=False):
        stats = self._storage.get_stats(scan_pool, scan_hash)
        stats.update({
            "occupied_voxels": int(self._mapper.state.occupancy.sum()),
            "static_voxels": int(self._static_mask.sum()),
            "generation": int(self._mapper.state.generation.max()),
        })
        return stats

    def memory_usage_mb(self):
        return self._storage.memory_usage_mb()

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
        """Stamp portable cuboids and spheres into the dense TSDF occupancy.

        This is deliberately bounded to analytic primitives.  CUDA/Warp scene
        tensors, mesh BVHs, and sparse static channels have materially
        different memory and tie semantics, so they reject explicitly.
        """
        if not self.config.enable_static:
            raise RuntimeError("MapperCfg.enable_static=True is required before stamping static obstacles")
        from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
        from curobo._src.geom.data.data_scene import SceneData

        if isinstance(scene, SceneData):
            scene = scene.scene_model
        if isinstance(scene, (list, tuple)):
            if env_idx < 0 or env_idx >= len(scene):
                raise ValueError("env_idx is outside the provided static scene list")
            scene = scene[env_idx]
        if not isinstance(scene, SceneCfg):
            raise TypeError("static stamping requires SceneCfg or SceneData carrying SceneCfg")
        if scene.mesh or scene.voxel or scene.capsule or scene.cylinder:
            raise NotImplementedError("portable static stamping supports only cuboids and spheres; mesh/voxel/Warp primitives require CUDA/Warp")
        state = self._mapper.state
        centers = torch.stack(torch.meshgrid(
            *[(torch.arange(n, device=self.device, dtype=state.tsdf.dtype) - (n - 1) / 2) * self.config.voxel_size + c
              for n, c in zip(self.config.grid_shape, self._mapper.config.grid_center)], indexing="ij",
        ), -1)
        mask = torch.zeros_like(state.occupancy[0])
        for obstacle in scene.cuboid:
            if not isinstance(obstacle, Cuboid):
                raise TypeError("scene.cuboid must contain Cuboid records")
            pose = obstacle.pose
            position = centers.new_tensor(pose[:3])
            quat = centers.new_tensor(pose[3:])
            quat = quat / torch.linalg.vector_norm(quat).clamp_min(torch.finfo(centers.dtype).eps)
            w, x, y, z = quat.unbind()
            rotation = torch.stack((
                1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
            )).reshape(3, 3)
            local = (centers - position) @ rotation
            half = centers.new_tensor(obstacle.dims) / 2
            mask |= (local.abs() <= half).all(-1)
        for obstacle in scene.sphere:
            if not isinstance(obstacle, Sphere):
                raise TypeError("scene.sphere must contain Sphere records")
            position = centers.new_tensor(obstacle.position if obstacle.position is not None else obstacle.pose[:3])
            mask |= torch.linalg.vector_norm(centers - position, dim=-1) <= obstacle.radius
        # Remove the previous static layer before writing a replacement.  The
        # dense backend has no separate static channel, hence this only claims
        # static scene mutation between depth integration calls.
        dynamic_occupancy = state.occupancy.clone()
        dynamic_occupancy[self._static_mask] = False
        occupancy = dynamic_occupancy.clone()
        occupancy[0] |= mask
        tsdf = state.tsdf.clone()
        weight = state.weight.clone()
        tsdf[0] = torch.where(mask, torch.full_like(tsdf[0], -0.5), tsdf[0])
        weight[0] = torch.where(mask, torch.ones_like(weight[0]), weight[0])
        esdf, gradient = dense_esdf(occupancy, self.config.voxel_size, self._mapper.config.unobserved_esdf,
                                    dtype=state.tsdf.dtype)
        self._replace_state(tsdf=tsdf, weight=weight, occupancy=occupancy, esdf=esdf,
                            gradient=gradient, generation=state.generation + 1)
        self._static_mask = torch.zeros_like(occupancy)
        self._static_mask[0] = mask
        return int(mask.sum().item())

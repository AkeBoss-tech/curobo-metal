"""Bounded PyTorch block-pool runtime used for large portable TSDF maps.

The production CUDA implementation combines a Warp hash table with a fixed
pool of voxel blocks.  Allocating the nominal grid densely is not viable for
the large (commonly 512 cubed) maps that motivate that API.  This module keeps
the same bounded-pool semantics in ordinary PyTorch: block lookup is handled
by a host dictionary while all observable block data remains device-resident.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from curobo._src.perception.mapper.storage import BlockSparseTSDFData


def _portable_device(value: str | torch.device) -> torch.device:
    if str(value).startswith("cuda"):
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


class PortableSparseTSDF:
    """Fixed-capacity TSDF block pool with source-shaped observable tensors."""

    _portable_sparse = True

    @property
    def block_size(self) -> int:
        return int(self.config.block_size)

    @property
    def grid_center(self) -> torch.Tensor:
        return self.data.origin

    def __init__(self, config: Any):
        self.config = config
        self.device = _portable_device(config.device)
        max_blocks = int(config.max_blocks)
        # Capacity is a limit, not a request to materialize every block.
        # Grow physical tensors only as blocks are allocated.
        initial_capacity = min(max_blocks, max(16, min(1024, int(math.ceil(max_blocks ** 0.5)))))
        self._capacity = initial_capacity
        block_voxels = int(config.block_size) ** 3
        color_voxels = int(config.color_grid_size) ** 3
        feature_voxels = int(config.feature_block_grid_size) ** 3
        empty_i32 = lambda n=1: torch.zeros(n, dtype=torch.int32, device=self.device)
        self.data = BlockSparseTSDFData(
            block_data=torch.zeros(
                (initial_capacity, block_voxels, 2), dtype=torch.float16, device=self.device
            ),
            block_grid_rgb=torch.zeros(
                (initial_capacity, color_voxels, 4), dtype=torch.float16, device=self.device
            ),
            # ``block_coords`` is part of the source-visible pool ABI: callers
            # are allowed to reshape it using ``max_blocks`` even before the
            # lazy data arrays have grown to that size.  Coordinates are tiny
            # (3 int32s per slot), so keep this one metadata array at logical
            # capacity while retaining lazy allocation for voxel, RGB, and
            # feature payloads.
            block_coords=torch.zeros(max_blocks * 3, dtype=torch.int32, device=self.device),
            block_size=int(config.block_size),
            decay_factor=torch.ones(initial_capacity, dtype=torch.float32, device=self.device),
            free_count=empty_i32(),
            free_list=empty_i32(initial_capacity),
            frustum_flags=empty_i32(initial_capacity),
            grid_shape=tuple(config.grid_shape),
            hash_capacity=int(config.hash_capacity),
            hash_table=torch.full(
                (int(config.hash_capacity),), -1, dtype=torch.int64, device=self.device
            ),
            max_blocks=max_blocks,
            new_block_count=empty_i32(),
            new_blocks=empty_i32(initial_capacity),
            num_allocated=empty_i32(),
            origin=config.origin.to(device=self.device, dtype=torch.float32),
            truncation_distance=float(config.truncation_distance),
            voxel_size=float(config.voxel_size),
            allocation_failures=empty_i32(),
            block_sums=torch.zeros(initial_capacity, dtype=torch.float32, device=self.device),
            block_to_hash_slot=torch.full(
                (initial_capacity,), -1, dtype=torch.int32, device=self.device
            ),
            recycle_count=empty_i32(),
            static_block_data=(
                torch.full(
                    (initial_capacity, block_voxels),
                    float("inf"),
                    dtype=torch.float16,
                    device=self.device,
                )
                if config.enable_static
                else torch.full((1, 1), float("inf"), dtype=torch.float16, device=self.device)
            ),
            static_block_sums=torch.zeros(initial_capacity, dtype=torch.float32, device=self.device),
            feature_dim=int(config.feature_dim),
            feature_block_grid_size=int(config.feature_block_grid_size),
            color_grid_size=int(config.color_grid_size),
            has_dynamic=True,
            has_static=bool(config.enable_static),
            has_features=bool(config.feature_dim),
            block_features=(
                torch.zeros(
                    (initial_capacity, feature_voxels, int(config.feature_dim)),
                    dtype=torch.float16,
                    device=self.device,
                )
                if config.feature_dim
                else torch.zeros((1, 1, 1), dtype=torch.float16, device=self.device)
            ),
            block_feature_weight=(
                torch.zeros(
                    (initial_capacity, feature_voxels), dtype=torch.float16, device=self.device
                )
                if config.feature_dim
                else torch.zeros((1, 1), dtype=torch.float16, device=self.device)
            ),
        )
        self._coord_to_pool: dict[tuple[int, int, int], int] = {}
        self._pool_to_coord: dict[int, tuple[int, int, int]] = {}
        self._free: list[int] = []

    def _grow(self, required: int) -> None:
        """Grow physical tensors while preserving the advertised max_blocks."""
        if required <= self._capacity:
            return
        old = self._capacity
        new = min(self.data.max_blocks, max(required, old * 2))
        if new < required:
            raise RuntimeError("sparse block pool capacity exhausted")
        def grow(value: torch.Tensor, shape: tuple[int, ...], fill: float = 0.0) -> torch.Tensor:
            result = torch.full(shape, fill, dtype=value.dtype, device=value.device)
            result[: min(value.shape[0], result.shape[0])].copy_(value[: min(value.shape[0], result.shape[0])])
            return result
        self.data.block_data = grow(self.data.block_data, (new, *self.data.block_data.shape[1:]))
        self.data.block_grid_rgb = grow(self.data.block_grid_rgb, (new, *self.data.block_grid_rgb.shape[1:]))
        # ``block_coords`` is allocated at logical capacity in ``__init__``;
        # do not couple its footprint to the lazily grown payload arrays.
        if self.data.block_coords.numel() < self.data.max_blocks * 3:
            self.data.block_coords = grow(
                self.data.block_coords, (self.data.max_blocks * 3,)
            )
        self.data.decay_factor = grow(self.data.decay_factor, (new,), 1.0)
        self.data.free_list = grow(self.data.free_list, (new,), -1)
        self.data.frustum_flags = grow(self.data.frustum_flags, (new,))
        self.data.new_blocks = grow(self.data.new_blocks, (new,), -1)
        self.data.block_sums = grow(self.data.block_sums, (new,))
        self.data.block_to_hash_slot = grow(self.data.block_to_hash_slot, (new,), -1)
        self.data.static_block_sums = grow(self.data.static_block_sums, (new,))
        if self.data.has_static:
            self.data.static_block_data = grow(self.data.static_block_data, (new, *self.data.static_block_data.shape[1:]), float("inf"))
        if self.data.has_features:
            self.data.block_features = grow(self.data.block_features, (new, *self.data.block_features.shape[1:]))
            self.data.block_feature_weight = grow(self.data.block_feature_weight, (new, *self.data.block_feature_weight.shape[1:]))
        self._capacity = new

    def _coords_in_bounds(self, coords: torch.Tensor) -> torch.Tensor:
        # ``grid_shape`` is the source (z, y, x) order while block keys are
        # world (x, y, z).  The sparse origin is the map centre, so keys are
        # signed block coordinates around the centered grid anchor.
        blocks = torch.tensor(
            [math.ceil(int(self.config.grid_shape[2]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[1]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[0]) / self.config.block_size)],
            dtype=torch.int64,
        )
        lower = -(blocks // 2)
        upper = lower + blocks
        return coords[((coords >= lower) & (coords < upper)).all(dim=1)]

    def _candidate_blocks(self, observation: Any) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
        depth = observation.depth_image.detach().to(device="cpu", dtype=torch.float32)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        intrinsics = observation.intrinsics.detach().to(device="cpu", dtype=torch.float32)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        matrices = observation.pose.get_matrix().detach().to(device="cpu", dtype=torch.float32)
        if matrices.ndim == 2:
            matrices = matrices.unsqueeze(0)
        rgb = observation.rgb_image
        if rgb is None:
            mean_rgb = torch.zeros(3, dtype=torch.float32)
        else:
            rgb_cpu = rgb.detach().to(device="cpu", dtype=torch.float32)
            valid_rgb = torch.isfinite(rgb_cpu).all(dim=-1)
            mean_rgb = (
                rgb_cpu[valid_rgb].mean(dim=0) / 255.0
                if bool(valid_rgb.any())
                else torch.zeros(3, dtype=torch.float32)
            )

        candidates: list[torch.Tensor] = []
        block_extent = float(self.config.voxel_size * self.config.block_size)
        blocks = torch.tensor(
            [math.ceil(int(self.config.grid_shape[2]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[1]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[0]) / self.config.block_size)],
            dtype=torch.int64,
        )
        for camera in range(depth.shape[0]):
            image = depth[camera]
            valid = (
                torch.isfinite(image)
                & (image >= self.config.depth_minimum_distance)
                & (image <= self.config.depth_maximum_distance)
            )
            if not bool(valid.any()):
                continue
            v, u = torch.nonzero(valid, as_tuple=True)
            z = image[v, u]
            k = intrinsics[min(camera, intrinsics.shape[0] - 1)]
            rays = torch.stack(
                ((u.float() - k[0, 2]) / k[0, 0], (v.float() - k[1, 2]) / k[1, 1], torch.ones_like(z)),
                dim=-1,
            )
            distances = torch.arange(
                float(self.config.depth_minimum_distance),
                float(z.max()) + float(self.config.truncation_distance) + block_extent,
                block_extent,
            )
            along_ray = distances[None, :] <= (
                z[:, None] + float(self.config.truncation_distance)
            )
            local = rays[:, None, :] * distances[None, :, None]
            transform = matrices[min(camera, matrices.shape[0] - 1)]
            world = torch.matmul(local, transform[:3, :3].T) + transform[:3, 3]
            # A block-sized traversal step need not land inside the narrow
            # truncation band. Include each measured endpoint and its rear
            # truncation endpoint explicitly so the allocated pool contains
            # both sides of the zero crossing, as the CUDA seeding pass does.
            endpoint_local = rays[:, None, :] * torch.stack(
                (z, z + float(self.config.truncation_distance)), dim=1
            )[..., None]
            endpoint_world = (
                torch.matmul(endpoint_local, transform[:3, :3].T) + transform[:3, 3]
            )
            selected_world = torch.cat(
                (world[along_ray], endpoint_world.reshape(-1, 3)), dim=0
            )
            grid_xyz = torch.tensor(
                [int(self.config.grid_shape[2]), int(self.config.grid_shape[1]),
                 int(self.config.grid_shape[0])], dtype=torch.float32
            )
            block_offset = -(blocks // 2).to(torch.float32)
            map_lower = self.config.origin.cpu() - grid_xyz * float(self.config.voxel_size) * 0.5
            block = torch.floor((selected_world - map_lower) / block_extent).to(torch.int64) + block_offset.to(torch.int64)
            candidates.append(block)
        if not candidates:
            return [], mean_rgb
        unique = torch.unique(torch.cat(candidates, dim=0), dim=0)
        unique = self._coords_in_bounds(unique)
        return [tuple(int(v) for v in row.tolist()) for row in unique], mean_rgb

    def _allocate(self, coord: tuple[int, int, int]) -> tuple[int | None, bool]:
        existing = self._coord_to_pool.get(coord)
        if existing is not None:
            return existing, False
        if self._free:
            pool = self._free.pop()
            self.data.free_count.fill_(len(self._free))
        else:
            high_water = int(self.data.num_allocated.item())
            if high_water >= self.data.max_blocks:
                self.data.allocation_failures.add_(1)
                return None, False
            pool = high_water
            self._grow(pool + 1)
            self.data.num_allocated.add_(1)
        self._coord_to_pool[coord] = pool
        self._pool_to_coord[pool] = coord
        self.data.block_coords.view(-1, 3)[pool] = torch.tensor(
            coord, dtype=torch.int32, device=self.device
        )
        # Portable storage does not expose Warp's packed hash ABI, but this
        # source-visible array still carries the active/inactive pool-slot
        # contract used by feature matching and checkpoint consumers.
        self.data.block_to_hash_slot[pool] = pool
        # Keep the source-visible hash arrays coherent as well.  The portable
        # dictionary remains authoritative, but consumers inspect these fields
        # after clear/reallocate and expect no stale slot.
        slot = int((((coord[0] * 73856093) ^ (coord[1] * 19349663) ^ (coord[2] * 83492791)) & 0x7FFFFFFF) % self.data.hash_capacity)
        for _ in range(self.data.hash_capacity):
            entry = int(self.data.hash_table[slot].item())
            if entry < 0:
                self.data.hash_table[slot] = pool
                self.data.block_to_hash_slot[pool] = slot
                break
            slot = (slot + 1) % self.data.hash_capacity
        return pool, True

    def _integrate_candidates(
        self,
        coords: list[tuple[int, int, int]],
        rgb: torch.Tensor | None,
        features: torch.Tensor | None = None,
        *,
        visible_capacity: int | None = None,
        depth_observation: Any | None = None,
    ) -> int:
        """Allocate candidate blocks and fuse compact geometry/appearance grids."""
        new: list[int] = []
        touched: list[int] = []
        for coord in coords:
            pool, created = self._allocate(coord)
            if pool is None:
                continue
            touched.append(pool)
            if created:
                new.append(pool)
        self.data.new_blocks.zero_()
        if new:
            self.data.new_blocks[: len(new)] = torch.tensor(new, dtype=torch.int32, device=self.device)
        self.data.new_block_count.fill_(len(new))
        self.data.frustum_flags.zero_()
        if not touched:
            return 0
        index = torch.tensor(touched, dtype=torch.long, device=self.device)
        if visible_capacity is not None and len(touched) > visible_capacity:
            self.data.block_data[index] = 0
            self.data.block_grid_rgb[index] = 0
            if self.data.has_features:
                self.data.block_features[index] = 0
                self.data.block_feature_weight[index] = 0
            raise ValueError(
                f"num_visible_blocks={len(touched)} exceeds "
                f"max_visible_blocks_per_integration={visible_capacity}"
            )
        if depth_observation is None:
            values = self.data.block_data[index]
            values[..., 0] += 0.5
            values[..., 1] += 1.0
            values.clamp_(max=float(self.config.accumulator_w_max))
            self.data.block_data[index] = values
        else:
            self._integrate_depth_values(index, depth_observation)

        if rgb is not None:
            samples = rgb.detach().to(device="cpu", dtype=torch.float32)
            if samples.ndim == 3:
                samples = samples.unsqueeze(0)
            samples = samples.reshape(samples.shape[0], -1, 3) / 255.0
            node_count = self.data.block_grid_rgb.shape[1]
            positions = torch.linspace(
                0, max(samples.shape[1] - 1, 0), node_count
            ).round().long()
            colors = samples[:, positions].sum(dim=0).to(
                device=self.device, dtype=self.data.block_grid_rgb.dtype
            )
            rgbw = self.data.block_grid_rgb[index]
            rgbw[..., :3] += colors[None]
            rgbw[..., 3] += float(samples.shape[0])
            self.data.block_grid_rgb[index] = rgbw
        if self.data.has_features and features is not None:
            samples = features.detach().to(device="cpu", dtype=torch.float32).reshape(
                -1, self.data.feature_dim
            )
            node_count = self.data.block_features.shape[1]
            positions = torch.linspace(0, max(len(samples) - 1, 0), node_count).round().long()
            feature_nodes = samples[positions].to(
                device=self.device, dtype=self.data.block_features.dtype
            )
            self.data.block_features[index] += feature_nodes[None]
            self.data.block_feature_weight[index] += 1.0
        self.data.frustum_flags[index] = 1
        self.data.block_sums[index] = self.data.block_data[index, :, 1].sum(
            dim=1, dtype=torch.float32
        )
        return len(touched)

    def _integrate_depth_values(self, pool_indices: torch.Tensor, observation: Any) -> None:
        """Fuse projective depth into each voxel of selected sparse blocks.

        ``block_data[..., 0]`` is the weighted normalized TSDF sum and
        ``block_data[..., 1]`` its observation weight, matching the CUDA block
        pool representation.  Geometry is evaluated on CPU because camera
        projection is small (only selected blocks) and this avoids unsupported
        MPS indexing corner cases; the persistent pool remains device-resident.
        """
        pools = pool_indices.detach().to("cpu", dtype=torch.long)
        block_coords = self.data.block_coords.view(-1, 3)[pools].to("cpu", torch.float32)
        block_size = int(self.config.block_size)
        voxel_size = float(self.config.voxel_size)
        local_axis = (torch.arange(block_size, dtype=torch.float32) + 0.5) * voxel_size
        local = torch.stack(
            torch.meshgrid(local_axis, local_axis, local_axis, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        blocks = torch.tensor(
            [math.ceil(int(self.config.grid_shape[2]) / block_size),
             math.ceil(int(self.config.grid_shape[1]) / block_size),
             math.ceil(int(self.config.grid_shape[0]) / block_size)], dtype=torch.float32
        )
        block_offset = torch.floor(blocks * 0.5)
        grid_xyz = torch.tensor(
            [int(self.config.grid_shape[2]), int(self.config.grid_shape[1]),
             int(self.config.grid_shape[0])], dtype=torch.float32
        )
        map_lower = self.data.origin.detach().to("cpu", torch.float32) - grid_xyz * voxel_size * 0.5
        base = map_lower + (block_coords + block_offset) * (block_size * voxel_size)
        world = base[:, None, :] + local[None, :, :]

        depth = observation.depth_image.detach().to("cpu", torch.float32)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        intrinsics = observation.intrinsics.detach().to("cpu", torch.float32)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        matrices = observation.pose.get_matrix().detach().to("cpu", torch.float32)
        if matrices.ndim == 2:
            matrices = matrices.unsqueeze(0)
        added_sum = torch.zeros(world.shape[:-1], dtype=torch.float32)
        added_weight = torch.zeros_like(added_sum)
        height, width = depth.shape[-2:]
        for camera in range(depth.shape[0]):
            matrix = matrices[min(camera, len(matrices) - 1)]
            camera_points = (world - matrix[:3, 3]) @ matrix[:3, :3]
            z = camera_points[..., 2]
            safe_z = z.clamp_min(torch.finfo(torch.float32).tiny)
            k = intrinsics[min(camera, len(intrinsics) - 1)]
            px = torch.round(k[0, 0] * camera_points[..., 0] / safe_z + k[0, 2]).long()
            py = torch.round(k[1, 1] * camera_points[..., 1] / safe_z + k[1, 2]).long()
            in_image = (z > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
            linear = py.clamp(0, height - 1) * width + px.clamp(0, width - 1)
            sampled = depth[camera].reshape(-1)[linear]
            valid_depth = (
                torch.isfinite(sampled)
                & (sampled >= float(self.config.depth_minimum_distance))
                & (sampled <= float(self.config.depth_maximum_distance))
            )
            sdf = sampled - z
            # Keep the source mapper's one-sided projective integration
            # contract: valid free-space samples in front of a surface carry
            # the saturated +1 value.  Surface extraction below removes those
            # saturated values when applying its metric threshold.
            valid = in_image & valid_depth & (sdf >= -float(self.config.truncation_distance))
            normalized = (sdf / float(self.config.truncation_distance)).clamp(-1.0, 1.0)
            added_sum += torch.where(valid, normalized, torch.zeros_like(normalized))
            added_weight += valid.to(torch.float32)

        current = self.data.block_data[pool_indices].float()
        total_sum = current[..., 0] + added_sum.to(self.device)
        total_weight = current[..., 1] + added_weight.to(self.device)
        maximum = float(self.config.accumulator_w_max)
        scale = torch.where(
            total_weight > maximum,
            maximum / total_weight.clamp_min(torch.finfo(torch.float32).tiny),
            torch.ones_like(total_weight),
        )
        current[..., 0] = total_sum * scale
        current[..., 1] = total_weight * scale
        self.data.block_data[pool_indices] = current.to(self.data.block_data.dtype)

    def integrate(self, observation: Any, *, visible_capacity: int | None = None) -> None:
        coords, _mean_rgb = self._candidate_blocks(observation)
        return self._integrate_candidates(
            coords,
            observation.rgb_image,
            getattr(observation, "feature_grid", None),
            visible_capacity=visible_capacity,
            depth_observation=observation,
        )

    def integrate_lidar(self, observation: Any, *, visible_capacity: int | None = None) -> int:
        ranges = observation.range_image.detach().to(device="cpu", dtype=torch.float32)
        if ranges.ndim == 2:
            ranges = ranges.unsqueeze(0)
        valid_range = observation.valid_range_m.detach().to(device="cpu", dtype=torch.float32)
        if valid_range.ndim == 1:
            valid_range = valid_range.unsqueeze(0)
        elevation = observation.elevation_range_rad.detach().to(device="cpu", dtype=torch.float32)
        if elevation.ndim == 1:
            elevation = elevation.unsqueeze(0)
        matrices = observation.pose.get_matrix().detach().to(device="cpu", dtype=torch.float32)
        if matrices.ndim == 2:
            matrices = matrices.unsqueeze(0)
        candidates: list[torch.Tensor] = []
        block_extent = float(self.config.voxel_size * self.config.block_size)
        steps = max(2, math.ceil(2 * self.config.truncation_distance / block_extent) + 1)
        offsets = torch.linspace(-self.config.truncation_distance, self.config.truncation_distance, steps)
        for sensor in range(ranges.shape[0]):
            image = ranges[sensor]
            valid = torch.isfinite(image) & (image >= valid_range[sensor, 0]) & (image <= valid_range[sensor, 1])
            if not bool(valid.any()):
                continue
            row, column = torch.nonzero(valid, as_tuple=True)
            distance = image[row, column]
            azimuth = -math.pi + 2.0 * math.pi * column.float() / image.shape[1]
            if image.shape[0] == 1:
                angle = torch.full_like(azimuth, float(elevation[sensor, 0]))
            else:
                angle = elevation[sensor, 0] + (
                    elevation[sensor, 1] - elevation[sensor, 0]
                ) * row.float() / (image.shape[0] - 1)
            cos_e = torch.cos(angle)
            rays = torch.stack(
                (cos_e * torch.cos(azimuth), cos_e * torch.sin(azimuth), torch.sin(angle)), dim=-1
            )
            local = rays[:, None] * (distance[:, None, None] + offsets[None, :, None])
            transform = matrices[min(sensor, matrices.shape[0] - 1)]
            world = torch.matmul(local, transform[:3, :3].T) + transform[:3, 3]
            candidates.append(
                torch.floor((world.reshape(-1, 3) - self.config.origin.cpu()) / block_extent).long()
            )
        coords: list[tuple[int, int, int]] = []
        if candidates:
            unique = self._coords_in_bounds(torch.unique(torch.cat(candidates), dim=0))
            coords = [tuple(int(v) for v in row.tolist()) for row in unique]
        return self._integrate_candidates(
            coords,
            observation.rgb_image,
            getattr(observation, "feature_grid", None),
            visible_capacity=visible_capacity,
        )

    def decay_and_recycle(self, factor: float) -> int:
        if not math.isfinite(float(factor)) or not 0.0 <= factor <= 1.0:
            raise ValueError("decay_factor must be finite and in [0, 1]")
        high_water = int(self.data.num_allocated.item())
        if high_water == 0:
            self.data.recycle_count.zero_()
            return 0
        active = sorted(self._pool_to_coord)
        if active:
            index = torch.tensor(active, dtype=torch.long, device=self.device)
            self.data.block_data[index] *= factor
            self.data.block_grid_rgb[index] *= factor
            if self.data.has_features:
                self.data.block_features[index] *= factor
                self.data.block_feature_weight[index] *= factor
            sums = self.data.block_data[index, :, 1].sum(dim=1, dtype=torch.float32)
            self.data.block_sums[index] = sums
            expired = index[sums <= 1.0e-5].to("cpu").tolist()
        else:
            expired = []
        for pool in expired:
            coord = self._pool_to_coord.pop(int(pool))
            self._coord_to_pool.pop(coord, None)
            self.data.block_data[pool].zero_()
            self.data.block_grid_rgb[pool].zero_()
            if self.data.has_static:
                self.data.static_block_data[pool].fill_(float("inf"))
                self.data.static_block_sums[pool] = 0
            if self.data.has_features:
                self.data.block_features[pool].zero_()
                self.data.block_feature_weight[pool].zero_()
            self.data.block_sums[pool] = 0
            self.data.frustum_flags[pool] = 0
            slot = int(self.data.block_to_hash_slot[pool].item())
            if 0 <= slot < self.data.hash_capacity:
                self.data.hash_table[slot] = -2
            self.data.block_to_hash_slot[pool] = -1
            self._free.append(int(pool))
        self.data.free_list.zero_()
        if self._free:
            self.data.free_list[: len(self._free)] = torch.tensor(
                self._free, dtype=torch.int32, device=self.device
            )
        self.data.free_count.fill_(len(self._free))
        self.data.recycle_count.fill_(len(expired))
        return len(expired)

    def clear_blocks(self, pool_indices: Any) -> int:
        indices = torch.as_tensor(pool_indices, dtype=torch.int64).reshape(-1).tolist()
        cleared = 0
        for raw_pool in indices:
            pool = int(raw_pool)
            coord = self._pool_to_coord.pop(pool, None)
            if coord is None:
                continue
            self._coord_to_pool.pop(coord, None)
            self.data.block_data[pool].zero_()
            self.data.block_grid_rgb[pool].zero_()
            if self.data.has_features:
                self.data.block_features[pool].zero_()
                self.data.block_feature_weight[pool].zero_()
            if self.data.has_static:
                self.data.static_block_data[pool].fill_(float("inf"))
                self.data.static_block_sums[pool] = 0
            self.data.block_sums[pool] = 0
            self.data.frustum_flags[pool] = 0
            slot = int(self.data.block_to_hash_slot[pool].item())
            if 0 <= slot < self.data.hash_capacity:
                self.data.hash_table[slot] = -2
            self.data.block_to_hash_slot[pool] = -1
            self._free.append(pool)
            cleared += 1
        self.data.free_list.zero_()
        if self._free:
            self.data.free_list[: len(self._free)] = torch.tensor(
                self._free, dtype=torch.int32, device=self.device
            )
        self.data.free_count.fill_(len(self._free))
        return cleared

    def clear_region(self, bounds_min: Any, bounds_max: Any) -> int:
        minimum = torch.as_tensor(bounds_min, dtype=torch.float32).reshape(3)
        maximum = torch.as_tensor(bounds_max, dtype=torch.float32).reshape(3)
        if bool((maximum < minimum).any()):
            raise ValueError("bounds_max must be greater than or equal to bounds_min")
        extent = float(self.config.voxel_size * self.config.block_size)
        blocks = torch.tensor(
            [math.ceil(int(self.config.grid_shape[2]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[1]) / self.config.block_size),
             math.ceil(int(self.config.grid_shape[0]) / self.config.block_size)], dtype=torch.int64
        )
        grid_xyz = torch.tensor(
            [int(self.config.grid_shape[2]), int(self.config.grid_shape[1]),
             int(self.config.grid_shape[0])], dtype=torch.float32
        )
        map_lower = self.config.origin.cpu() - grid_xyz * float(self.config.voxel_size) * 0.5
        offset = blocks // 2
        lo = torch.floor((minimum - map_lower) / extent).to(torch.int64) - offset
        hi = torch.floor((maximum - map_lower) / extent).to(torch.int64) - offset
        targets = [pool for pool, coord in self._pool_to_coord.items()
                   if all(int(lo[i]) <= coord[i] <= int(hi[i]) for i in range(3))]
        return self.clear_blocks(targets)

    def reset(self) -> None:
        self.data.block_data.zero_()
        self.data.block_grid_rgb.zero_()
        if self.data.has_features:
            self.data.block_features.zero_()
            self.data.block_feature_weight.zero_()
        self.data.block_coords.zero_()
        self.data.block_sums.zero_()
        self.data.block_to_hash_slot.fill_(-1)
        self.data.hash_table.fill_(-1)
        self.data.num_allocated.zero_()
        self.data.free_count.zero_()
        self.data.allocation_failures.zero_()
        self.data.new_block_count.zero_()
        self.data.recycle_count.zero_()
        self._coord_to_pool.clear()
        self._pool_to_coord.clear()
        self._free.clear()

    def export_blocks(self) -> dict[str, torch.Tensor]:
        """Return the active source-shaped block payload in pool order."""
        pools = sorted(self._pool_to_coord)
        index = torch.tensor(pools, dtype=torch.long, device=self.device)
        blocks: dict[str, torch.Tensor] = {
            "active_block_coords": self.data.block_coords.view(-1, 3)[index].detach().clone(),
            "block_data": self.data.block_data[index].detach().clone(),
            "block_grid_rgb": self.data.block_grid_rgb[index].detach().clone(),
        }
        if self.data.has_features:
            blocks["block_features"] = self.data.block_features[index].detach().clone()
            blocks["block_feature_weight"] = self.data.block_feature_weight[index].detach().clone()
        if self.data.has_static:
            blocks["static_block_data"] = self.data.static_block_data[index].detach().clone()
        return blocks

    def import_blocks(self, blocks: dict[str, torch.Tensor]) -> int:
        """Restore a validated compact payload into an empty block pool."""
        if self._pool_to_coord or int(self.data.num_allocated.item()):
            raise ValueError("block import requires an empty target")
        coords = blocks["active_block_coords"]
        count = int(coords.shape[0])
        if count > self.data.max_blocks:
            raise ValueError("import contains more blocks than target capacity")
        self.reset()
        self._grow(count)
        device_coords = coords.to(device=self.device, dtype=torch.int32)
        self.data.block_coords.view(-1, 3)[:count].copy_(device_coords)
        self.data.block_data[:count].copy_(blocks["block_data"].to(self.device))
        self.data.block_grid_rgb[:count].copy_(blocks["block_grid_rgb"].to(self.device))
        if self.data.has_features:
            self.data.block_features[:count].copy_(blocks["block_features"].to(self.device))
            self.data.block_feature_weight[:count].copy_(
                blocks["block_feature_weight"].to(self.device)
            )
        if self.data.has_static and "static_block_data" in blocks:
            self.data.static_block_data[:count].copy_(blocks["static_block_data"].to(self.device))
        self.data.num_allocated.fill_(count)
        self.data.block_sums[:count] = self.data.block_data[:count, :, 1].float().sum(-1)
        if count:
            self.data.block_to_hash_slot[:count] = -1
        for pool, coordinate in enumerate(coords.to("cpu", torch.int64).tolist()):
            key = tuple(int(value) for value in coordinate)
            self._coord_to_pool[key] = pool
            self._pool_to_coord[pool] = key
            slot = int((((key[0] * 73856093) ^ (key[1] * 19349663) ^ (key[2] * 83492791)) & 0x7FFFFFFF) % self.data.hash_capacity)
            for _ in range(self.data.hash_capacity):
                if int(self.data.hash_table[slot].item()) < 0:
                    self.data.hash_table[slot] = pool
                    self.data.block_to_hash_slot[pool] = slot
                    break
        return count

    def get_stats(self, scan_pool: bool = True, scan_hash: bool = False) -> dict[str, Any]:
        high_water = int(self.data.num_allocated.item())
        free = int(self.data.free_count.item())
        stats: dict[str, Any] = {
            "num_allocated": high_water,
            "free_count": free,
            "active_blocks": high_water - free,
            "holes": free,
            "recycled_last": int(self.data.recycle_count.item()),
            "allocation_failures": int(self.data.allocation_failures.item()),
            "pool_usage_pct": (high_water - free) / max(self.data.max_blocks, 1) * 100.0,
            "storage": "sparse_portable",
        }
        if not scan_pool:
            stats.pop("holes")
        if scan_hash:
            stats.update(
                {
                    "hash_empty": self.data.hash_capacity - len(self._coord_to_pool),
                    "hash_tomb": free,
                    "hash_occ": len(self._coord_to_pool),
                }
            )
        return stats

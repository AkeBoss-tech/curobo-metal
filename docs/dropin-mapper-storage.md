# Mapper storage lifecycle

`curobo._src.perception.mapper.storage.BlockSparseTSDF` is available on CPU
and MPS as a bounded dense PyTorch map.  It supports configuration validation,
real tensor state, reset, export/import of the portable dense state, memory and
occupancy statistics, and read-only `BlockDataView` / `OccupiedVoxels` queries.
`Mapper.tsdf` exposes this storage facade; its `.state` property contains the
underlying `DenseMap` (`tsdf`, `weight`, `occupancy`, `esdf`, `gradient`, and
`generation`).

`Mapper.update_static_obstacles()` stamps `SceneCfg` cuboids and spheres when
`MapperCfg.enable_static=True`.  Replacing the static scene is deterministic;
the dense implementation rebuilds static occupancy rather than retaining a
Warp hash/pool channel.  Subvoxel voxel extraction produces evenly spaced
sample centers and keeps the source voxel's dense flattened index.

Not supported: CUDA/Warp hash tables, raw `to_warp()` conversion, custom Warp
kernels, sparse block-pool import/export, learned feature/RGB control grids,
LiDAR mapping, mesh/voxel/capsule/cylinder static stamping, or sliding-window
ESDF resampling.  These methods fail with `NotImplementedError` (or a precise
configuration error) instead of presenting a dense tensor as ABI-compatible
Warp storage.

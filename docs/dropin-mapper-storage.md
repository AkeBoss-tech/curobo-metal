# Mapper storage lifecycle

`curobo._src.perception.mapper.storage.BlockSparseTSDF` is available on CPU
and MPS as a bounded dense PyTorch map.  It supports configuration validation,
real tensor state, reset, export/import of the portable dense state, memory and
occupancy statistics, per-frame new-coverage diagnostics, and read-only
`BlockDataView` / `OccupiedVoxels` queries.
`Mapper.tsdf` exposes this storage facade; its `.state` property contains the
underlying `DenseMap` (`tsdf`, `weight`, `occupancy`, `esdf`, `gradient`, and
`generation`).

`Mapper.update_static_obstacles()` stamps `SceneCfg` cuboids and spheres when
`MapperCfg.enable_static=True`.  Replacing the static scene is deterministic;
the dense implementation rebuilds static occupancy rather than retaining a
Warp hash/pool channel.  Subvoxel voxel extraction produces evenly spaced
sample centers and keeps the source voxel's dense flattened index.

`prepare_frame()` snapshots the current observed dense voxels; until the next
reset or `invalidate_cache()`, `tsdf.data.new_blocks` and
`tsdf.data.new_block_count` identify newly observed flattened dense voxel IDs.
`get_stats()` supplies the source monitoring keys in logical-block units:
`active_blocks`, `num_allocated`, and `free_count` refer to the fixed dense
grid's occupied logical blocks, while `holes`, fragmentation, and hash metrics
are zero because there is no allocation pool or hash table. Extra
`dense_*` fields describe actual dense capacity and observation coverage.

Not supported: CUDA/Warp hash tables, raw `to_warp()` conversion, custom Warp
kernels, sparse block-pool import/export, learned feature/RGB control grids,
LiDAR mapping, mesh/voxel/capsule/cylinder static stamping, or sliding-window
ESDF resampling.  These methods fail with `NotImplementedError` (or a precise
configuration error) instead of presenting a dense tensor as ABI-compatible
Warp storage.

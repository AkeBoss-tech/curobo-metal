# Portable mapper lifecycle

`MapperCfg` now follows the pinned V2 public coordinate contract: its
`grid_shape` is `(nz, ny, nx)`, `voxel_to_world(iz, iy, ix)` maps voxel
centres, and `world_to_voxel` rounds to the nearest valid voxel or returns
`(-1, -1, -1)` out of bounds. The dense CPU/MPS backend exposes
`native_grid_shape` solely for its internal `(x, y, z)` tensor layout.

The high-level `Mapper` keeps depth integration, reset, checkpoint import and
export, static-scene updates, dynamic AABB/cell clearing, ESDF caching, mesh
and voxel extraction, and rendering on the selected CPU or float32 MPS device.
Static cuboid/sphere stamps form a separate portable overlay: dynamic clear and
camera fusion preserve that overlay, and `reset()` clears both channels and the
mapper's frame/ESDF counters. This is a real dense-map lifecycle, not a Warp
block-pool emulation.

The portable ESDF is computed at `voxel_size`. Requests for a distinct ESDF
resolution, sliding windows, LiDAR integration, feature/RGB volumes, sparse
Warp checkpoint payloads, Blox, and raw CUDA/Warp rendering/integration kernels
raise explicit errors rather than silently changing map registration or data
meaning.

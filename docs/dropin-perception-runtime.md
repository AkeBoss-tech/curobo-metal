# Perception runtime compatibility

This wave makes every pinned `curobo._src.perception.mapper` module and the
pose-estimation helper modules import safely without Warp, CUDA, Blox, or
`trimesh`.

The public `Mapper`, TSDF integrator, exact dense ESDF, sparse block packing,
mesh extraction, rendering, pose refinement, coordinate conversion,
quantization, checkpoints, point-cloud alignment, and LM primitives execute
with ordinary PyTorch tensors on CPU and float32 MPS. They route to the
production `curobo_metal.ops.perception` implementation. The portable map is a
bounded dense representation with sparse checkpoint/export views; it is not
the upstream Warp hash-table ABI.

Raw Warp kernel functions and builder entry points import but raise a precise
`NotImplementedError`. Lidar integration, Blox representations, raw pointers,
CUDA launch objects, feature-volume texture fusion, and the exact Warp/PBA/JFA
scheduling are not emulated. The portable ESDF computes an exact dense distance
transform, so values follow the documented mathematical contract while
performance and tie behavior need not match those approximate CUDA schedulers.

`BlockSparseTSDFIntegrator` and `BlockSparseESDFIntegrator` now carry the V2
high-level lifecycle: camera fusion, resets/imports, AABB and explicit-cell
clearing, dense ESDF computation, mesh/voxel export, stats, and source-shaped
checkpoint metadata. The portable ESDF grid must use the TSDF grid's shape and
voxel size. Lossless sliding windows at integer-voxel offsets are supported;
fractional-window resampling remains explicit rather than silently returning
an incorrectly registered distance field. `blend_esdf=True` uses measured
observed TSDF values near surfaces and the exact dense EDT away from them. The
ESDF facade exposes an `is_esdf_current` lifecycle signal and a
`query(points, padding=...)` path that refreshes invalidated fields through
the production differentiable voxel sampler. `use_cuda_graph=True` is a
persistent portable-execution request, not a CUDA Graph object.

The integrator facades keep the V2 component and lifecycle spellings: `.tsdf`
and `._tsdf`, raw mesh export as `(vertices, triangles, normals, colors)`, a
real invalidatable dense `._site_index` diagnostic, and memory/stat accounting
for TSDF plus ESDF buffers. `time_decay` is applied to the dense map and
rebuilds its derived ESDF safely. Camera `frustum_decay` is also supported by
projecting the bounded dense voxel centers into each input camera; it preserves
the useful in-view-versus-out-of-view lifecycle without claiming the upstream
Warp per-block flag/pool ABI. TSDF surface export returns source-shaped
`(centers, uint8_colors, signed_distances_m)` tuples, and RGB-D mesh/voxel
exports use the portable visibility-tested projective texturer. Static
`SceneCfg` cuboids and spheres can be stamped when `enable_static=True`;
mesh/voxel/capsule/cylinder stamping remains explicit CUDA/Warp functionality.

Checkpoint loading uses Torch's weights-only mode and saves portable dense
tensor payloads under the V2 `curobo.mapper_blocks` metadata spelling. Native
Warp block-pool payloads, raw hash-table pointers, LiDAR frames, static Warp
stamping, and feature volumes fail explicitly. Optional external mesh packages
are not imported at module import time.

# Portable scene-data caches

`curobo._src.geom.data.data_scene`, `data_cuboid`, and `data_voxel` provide
the pinned V2 cache lifecycle using regular PyTorch tensors on CPU and Apple
Metal.  They support per-environment construction, batch replacement, named
lookup, enable/disable, pose and dimension updates, clear/reload operations,
and finite ESDF sampling.

Cuboid and voxel names must be unique within an environment; `SceneData` also
rejects duplicates across its collision layers so lookup cannot silently route
to the wrong geometry type.  Batch updates validate capacity and names before
clearing a current environment.  Voxel metadata follows V2's meaning:
`params = [nx, ny, nz, voxel_size]`, `dims` remain metric extents, and stored
inverse poses are reconstructed back to world poses by `get_voxel_grid`.

The V2 CUDA implementation may expose aliasing Warp buffers and Warp structs.
This portable implementation copies `VoxelGrid` feature data into its owned
cache, preserving deterministic CPU/MPS behavior and avoiding CUDA lifetime
assumptions.  `to_warp`, raw Warp SDF functions, and Warp kernel structs remain
explicitly unsupported.  Use the vectorized `sample_voxel_sdf` /
`sample_voxel_sdf_with_grad` helpers or the production `SceneCollision` path.

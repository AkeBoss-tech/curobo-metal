# Geometry and platform compatibility

This wave provides the pinned cuRoboV2 geometry-data, sphere-fitting, metric,
XRDF, profiling, and viewer import paths on CPU and Apple MPS.

Cuboid, mesh, and dense voxel records use ordinary PyTorch tensors and retain
multi-environment cache names, activation, mutation, pose storage, cache
replacement, and deterministic scene serialisation. `SceneData.from_scene_cfg`
and `SceneData.from_batch_scene_cfg` construct directly usable caches; callers
can add, enable, move, clear, and enumerate cuboid/mesh/ESDF layers on CPU or
Apple MPS. `SceneCfg` also converts supported analytic primitives to portable
triangle meshes or conservative OBBs, and can merge in-memory meshes without a
`trimesh` dependency. Sphere, cylinder, and arbitrary-axis capsule conversion
emit deterministic closed meshes (rather than a conservative box); flat and
nested polygon buffers triangulate consistently, point clouds can be
voxel-surface exported, mesh scales preserve tensors, and a complete scene can
be saved as a dependency-free merged OBJ. These are ordinary CPU/MPS PyTorch
value operations; discrete point-cloud topology is not differentiable.
Two-dimensional hull queries and mesh sphere fitting are deterministic. Mesh
SDF queries use the production differentiable triangle implementation rather
than Warp.

Raw Warp structs/functions remain explicit `NotImplementedError` boundaries.
Mesh data must be caller-provided vertices with triangular faces: raw Warp mesh
IDs, Warp BVH construction, external file loading, USD/Isaac scene conversion,
and texture/scene-graph export are not silently emulated. An optional
``trimesh`` scene graph is therefore not fabricated; portable OBJ export is the
supported interchange route. Voxel storage is dense; sparse map or Blox
lifecycle remains outside this data layer.
OpenUSD and Viser modules import without optional packages; constructing or
using those integrations raises a precise dependency error. When `usd-core` is
installed, basic stage and transform helpers work, while the larger
Isaac-specific robot animation surface remains explicitly unsupported.

`MeshData` now keeps immutable local triangle snapshots in a shared portable
cache, with stable cache identifiers, separate transforms/enables for each
environment, reconstruction to an in-memory `SceneCfg`, and a `query_points`
bridge to the production vectorized CPU/MPS triangle-distance operator.  It
supports shared geometry with different per-environment poses and preserves
point-query gradients, but it never exposes a pretend Warp mesh ID or BVH.
Reusing a mesh name with different geometry is rejected: changing it could
otherwise mutate a live shared cache in another environment.  Use a different
name, or clear the cache after all environments have released the geometry.

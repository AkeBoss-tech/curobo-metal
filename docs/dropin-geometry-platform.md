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
`trimesh` dependency.
Two-dimensional hull queries and mesh sphere fitting are deterministic. Mesh
SDF queries use the production differentiable triangle implementation rather
than Warp.

Raw Warp structs/functions remain explicit `NotImplementedError` boundaries.
Mesh data must be caller-provided vertices with triangular faces: raw Warp mesh
IDs, Warp BVH construction, external file loading, USD/Isaac scene conversion,
and texture/scene-graph export are not silently emulated. A capsule or cylinder
conversion produces a conservative cuboid mesh for portable collision rather
than claiming an exact Warp tessellation. Voxel storage is dense; sparse map
or Blox lifecycle remains outside this data layer.
OpenUSD and Viser modules import without optional packages; constructing or
using those integrations raises a precise dependency error. When `usd-core` is
installed, basic stage and transform helpers work, while the larger
Isaac-specific robot animation surface remains explicitly unsupported.

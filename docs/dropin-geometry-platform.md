# Geometry and platform compatibility

This wave provides the pinned cuRoboV2 geometry-data, sphere-fitting, metric,
XRDF, profiling, and viewer import paths on CPU and Apple MPS.

Cuboid, mesh, and dense voxel records use ordinary PyTorch tensors and retain
multi-environment cache names, activation, mutation, and pose storage.
Two-dimensional hull queries and mesh sphere fitting are deterministic. Mesh
SDF queries use the production differentiable triangle implementation rather
than Warp.

Raw Warp structs/functions remain explicit `NotImplementedError` boundaries.
OpenUSD and Viser modules import without optional packages; constructing or
using those integrations raises a precise dependency error. When `usd-core` is
installed, basic stage and transform helpers work, while the larger
Isaac-specific robot animation surface remains explicitly unsupported.

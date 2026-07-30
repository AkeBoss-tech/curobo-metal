# Drop-in collision and world namespace

This port targets cuRoboV2 revision `8e734f3ced1df898990bcd92de40abce475907db`.
It exposes `curobo.collision_checking`, `curobo._src.collision`, geometry obstacle
records, `CollisionBuffer`, and `SceneCollision` at their pinned import paths.

Production-backed behavior includes oriented cuboid, in-memory triangle mesh,
dense voxel/ESDF, discrete sphere, linearly swept sphere, self-sphere, named
obstacle activation, fixed-capacity caches, and environment routing. CPU and
Apple MPS use the same PyTorch operators; the wrapper creates the backend with
CPU fallback disabled. Cache overflow, invalid environment routing, duplicate
names, bad tensor layouts, negative activation distances, and unsupported
options raise rather than becoming no-ops.

## Known gaps

- Sphere, capsule, and cylinder scene objects are rejected. The production
  world backend currently has cuboid primitives only.
- File-backed meshes are rejected. Callers must supply vertices and triangular
  faces. Mesh collision is an exact vectorized triangle scan, not upstream
  Warp's per-mesh acceleration structure, so large-scene performance and
  closest-face tie breaking can differ.
- Dense ESDF grids are supported. Sparse voxel mutation and point-coordinate
  remapping are rejected; replace the dense `VoxelGrid` instead.
- The swept speed metric is rejected. Swept queries use deterministic linear
  interpolation, not upstream continuous Warp traversal.
- Robot config loading and high-level sampling/validation await the pinned
  kinematics and cost namespaces. Direct production-component construction and
  low-level world/self collision are available.
- Obstacle pose mutation currently requires rebuilding/reloading its owning
  `SceneCfg`; accepting the upstream mutation call would otherwise be a no-op.

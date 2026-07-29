# Jacobian and collision-checker compatibility

This wave exposes public Jacobian and cuRobo-style collision-checker surfaces
without introducing a second collision implementation. The wrappers route to
`ops.kinematics`, `ops.whole_body`, `ops.collision`, and
`ops.world_collision`; supported MPS queries therefore retain the existing
Metal dispatch.

## Jacobian convention

`geometric_jacobian(model, q)` returns `[B,L,6,J]`, including a batch
dimension for unbatched input. Rows `0:3` are the world-frame linear velocity
of the selected link-frame origin. Rows `3:6` are world-frame angular
velocity, computed as `vee(dR/dq Rᵀ)`. Columns use active-joint order. This is
a spatial/world geometric Jacobian, not a body Jacobian and not a Jacobian of
an arbitrary point offset from the link origin.

Serial `KinematicChain`/`SerialRobot` and tree
`WholeBodyModel`/`TreeRobot` models are supported. Link names and indices may
be selected; tree-configured end effectors and the final serial link have
convenience routing. CPU supports float32 and float64. MPS supports float32.
Jacobians remain differentiable with respect to `q`, including higher-order
derivatives supported by their underlying composed PyTorch path.

## Collision checker surface

`WorldCollisionConfig`, `RobotCollisionCheckerConfig`, and
`RobotSceneCollisionConfig` (plus upstream-style `Cfg` aliases) configure:

- fixed-capacity primitive, mesh, and voxel caches;
- multiple environments selected by per-batch `int64` indices;
- obstacle activation, replacement, removal, and clearing;
- activation distance, padding, and swept-query interpolation count;
- robot self-collision pair filtering.

Every cache mutation increments its generation. Queries materialize their
routing from the current slots and do not retain derived obstacle tensors, so
updates cannot return stale results. Capacity overflow, empty-slot activation,
invalid environment selection, and unequal ESDF slot layouts raise specific
errors.

Discrete sphere queries accept `[S,4]` or `[B,S,4]` packed as world
`(x,y,z,radius)`. Primitive, mesh, and voxel signed clearances are reduced by
minimum clearance; the public collision value is a nonnegative cuRobo-style
violation/cost and reduces across robot spheres with maximum. An empty world
returns zero cost and zero gradient. Equal-distance ties follow the underlying
operators' first serialized obstacle rule.

Swept queries accept start/end spheres or `[B,H,S,4]` trajectories and linearly
interpolate each segment, including radius. They return all sampled costs and
the maximum-over-time value and selected world-center gradient. This is sampled
swept collision, not continuous collision detection; callers must select a
step count appropriate to obstacle scale and velocity.

Voxel caches route both collision and raw ESDF queries. Multi-environment raw
ESDF extraction currently requires the same number of active grids in every
environment because the production sampler has a rectangular slot contract.

## Deliberate unsupported boundaries

- Arbitrary implicit CPU fallback is not performed. `allow_cpu_fallback` is
  policy metadata; supported CPU/MPS operators run on the query device, and
  device/dtype mismatches fail rather than silently copying tensors.
- Mesh cache entries currently use their serialized vertex frame. Per-slot
  mutable mesh poses should be baked into vertices or queried through the
  lower-level transformed mesh operator.
- Swept queries are interpolated samples, not an analytic time-of-impact API.
- The compatibility layer does not emulate upstream CUDA graph buffers,
  Warp/Blox objects, NVBlox mapper mutation, or CUDA-only acceleration
  structures.

Focused compatibility coverage lives in
`tests/compat/jacobian_collision_checker/`. Correctness and timing evidence is
recorded in `artifacts/correctness/jacobian_collision_checker.json`.

# Geometry collision and sphere-fit boundary

This package exposes the cuRoboV2 geometry collision and sphere-fit module
paths at pinned revision `8e734f3ced1df898990bcd92de40abce475907db`.

`SceneCollision`, `CollisionChecker`, `CollisionBuffer`, sphere fitting, and
`WarpMeshQuery` are usable through regular PyTorch tensors on CPU and Metal.
Mesh distance uses the production vectorized triangle-query operator.  Its
signed result requires a caller-declared watertight mesh; otherwise it is an
unsigned nearest-triangle query.

The following raw CUDA/Warp surfaces intentionally retain their callable
layouts but raise a precise `NotImplementedError`:

- `sphere_obstacle_collision_kernel`
- `swept_sphere_obstacle_collision_kernel`
- `SphereObstacleCollision` and `SweptSphereObstacleCollision`

They consume Warp obstacle descriptors, BVH mesh ids, atomic output buffers,
and CUDA stream launches.  Returning a plausible tensor from those entry points
would not preserve their synchronization, reduction, gradient, or analytic-CCD
semantics.  Use `SceneCollision.get_sphere_distance` or
`SceneCollision.get_swept_sphere_distance` instead; swept queries are sampled
at the documented fixed resolution, not an analytic continuous-collision
certificate.

`apply_speed_metric` follows V2's raw argument order and supports in-place
tensor buffers.  A four-argument result-returning order from early portable
releases remains available solely for backwards compatibility.

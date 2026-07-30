# Advanced collision and attachment compatibility

This wave completes the portable collision surfaces used by pinned cuRoboV2
robot-scene planning. `AttachmentManager` deterministically fits primitive or
vertex meshes to spheres, updates per-environment link-sphere configurations,
can disable the source world obstacles, and restores both on detach. Mutated
spheres flow through production FK rather than a display-only copy.

`SceneCollision` supports batched and multi-environment cuboids, triangle
meshes, dense ESDF voxels, spheres, capsules, and finite cylinders. Discrete,
sampled swept, ESDF, activation, obstacle pose/enable, sparse dense-grid update,
and deterministic first-tie behavior are available on CPU and MPS. Robot-scene
configuration builds packaged robots and collision checkers, joint sampling,
limit validation, world/self-collision queries, and trajectory validation.

The public compatibility modules do not import Warp. Their tensor-level
activation and speed-metric helpers execute in PyTorch; attempts to launch raw
Warp kernels fail explicitly and direct callers to `SceneCollision`. Mesh
queries use exact vectorized triangle scans rather than a Warp BVH, and swept
queries are fixed-resolution sampling rather than analytic continuous collision
detection. These differences affect large-scene performance and possible
between-sample contacts, not the documented signed-distance convention.

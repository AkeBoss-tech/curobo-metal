# Portable `RobotSceneCollision`

`curobo._src.collision.collision_robot_scene.RobotSceneCollision` composes the
portable tree kinematics, sphere world queries, self-collision pairs, joint
limits, and seeded sampling into the pinned high-level checker surface.  The
same object runs with PyTorch tensors on CPU and Apple MPS; reusable collision
diagnostic buffers are device-local and are cleared for a robot-only world.

Supported lifecycle and query surfaces include:

- setup, scene replacement/cache clearing, and multi-environment routing;
- FK-backed scene/self collision, constraints, validity masks, configuration
  and trajectory sampling, and active-joint selection;
- `tool_frames`, joint-bound checks, and differentiable point-to-robot sphere
  envelope distances.  Point distances are positive inside the envelope and
  support `[points,3]` with one robot or `[batch,points,3]` with one or matching
  robot batches.

`get_self_collision()` deliberately keeps this package's existing `[B,H]`
penalty view for compatibility with its planning stack.  Internal validity
checks normalize it before combining it with `[B,H,S]` scene clearances, so
non-square batch/horizon workloads have no accidental broadcasting.

This is not an emulation of CUDA launch buffers, Warp BVHs, or analytic
continuous collision detection.  Raw CUDA/Warp ABI entry points remain
explicit unsupported boundaries.  Portable mesh queries use vectorized
triangle scans and swept collision is sampled at the configured resolution.

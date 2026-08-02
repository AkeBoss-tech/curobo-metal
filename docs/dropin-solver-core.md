# Portable SolverCore lifecycle

`curobo._src.solver.solver_core.SolverCore` is the shared eager CPU/MPS
orchestration component used by the portable solver facades.  It owns the
robot kinematics, deterministic seed manager, goal registry, optional scene
collision object, and any caller-attached rollout/optimizer consumers.

Preparing a goal detects structural changes (solve mode, batch, environment,
goal-set, seed count, or controlled links).  Structural changes reset
shape-keyed consumers and size them to the seed-expanded problem batch.  A
value-only goal update still refreshes every attached rollout and optimizer,
which prevents stale target tensors in long-lived applications.

`update_world` accepts a `SceneCfg`, per-environment `list[SceneCfg]`,
`SceneCollisionCfg`, or an existing portable `SceneCollision`.  It replaces
the scene, propagates it to attached eager rollouts, and increments
`scene_generation` so applications can invalidate their own caches.  YAML,
USD, Warp collision objects, CUDA graph capture, streams, and graph debug
dumps remain explicit unsupported boundaries; they are never silently mapped
to an empty or CPU fallback scene.

## Seed preparation

`curobo._src.solver.manager_seed.SeedManager` keeps the V2 optimizer-facing
layouts: action seeds are returned as `[batch * seed, 1, dof]`, while
trajectory and deceleration seeds are `[batch * seed, horizon, dof]`.
Batch-major and seed-major action configuration inputs are normalized,
over-provisioned user seeds are truncated, and missing action seeds are padded
from a deterministic bound-respecting Halton buffer.  Trajectory preparation
prioritizes complete user trajectories, then interpolates user goal
configurations, then holds the current joint state. `reset_seed()` restores the
portable sample-buffer stream on CPU and MPS. CUDA graph capture, raw device
buffers, and CUDA/Warp sampling ABI are intentionally not emulated.

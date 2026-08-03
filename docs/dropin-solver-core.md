# Portable SolverCore lifecycle

`curobo._src.solver.solver_core.SolverCore` is the shared eager CPU/MPS
orchestration component used by the portable solver facades.  It owns the
robot kinematics, deterministic seed manager, goal registry, optional scene
collision object, and any caller-attached rollout/optimizer consumers.

When `SolverCoreCfg` contains typed `RobotRolloutCfg` and optimizer records,
the core also constructs its own metrics/auxiliary rollouts, optimizer stages,
`MultiStageOptimizer`, initial joint state, and attachment manager. This is
the direct V2 composition path. YAML-shaped placeholder rollout records remain
valid transport configuration for higher-level portable solvers, which own
their objective themselves; they are not misinterpreted as executable CUDA
rollouts.

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

The upstream `reset_cuda_graph()` lifecycle hook is safe to call during a
portable shape reset: it clears ordinary stage/rollout execution state and
does not claim to reset a CUDA graph. Explicit graph capture and graph debug
dump requests remain unsupported.

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

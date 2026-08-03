# Motion planner lifecycle compatibility

`MotionPlanner` and `BatchMotionPlanner` execute the high-level cuRobo
IK → optional PRM seed → TrajOpt composition on ordinary PyTorch CPU or Apple
Metal tensors.  Their reusable state is portable execution/cache state; it is
not a CUDA graph or stream capture.

## World updates

`update_world` accepts `SceneCfg`, a complete `list[SceneCfg]`,
`SceneCollisionCfg`, or a `SceneCollision`.  A scene update that fits the
current cache mutates that adapter in place, retaining references held by an
attachment manager and application code.  An explicitly supplied collision
adapter/config replaces it after validation.  Every successful update
synchronizes IK, TrajOpt, TrajOpt's pose-IK composition, PRM, and attachment
state to one world object; `world_generation` increments so applications can
invalidate their own derived caches.

For a `multi_env=True` batch planner, a world update must provide exactly
`max_batch_size` environments.  This prevents accidental routing of a
per-problem query through environment zero.  Unsupported YAML/USD scene paths
are rejected explicitly rather than treated as an empty world.

## Retry and destruction semantics

Batch planning retains the first successful row for every problem across
retries, including materialized optional trajectory payloads.  A destroyed
planner can be destroyed again safely, but rejects new planning, world, and
mutation work with `RuntimeError`.  This gives the portable lifecycle a clear
boundary without claiming CUDA graph teardown support.

The focused tests in `tests/dropin/motion_planner_lifecycle` exercise CPU and,
when Apple MPS is available, fallback-disabled device residency.

# Portable scene collision cost

`curobo._src.cost.cost_scene_collision.SceneCollisionCost` is a CPU/Metal
facade over the production `SceneCollision` checker.  It accepts the pinned
`[batch, horizon, spheres, 4]` `(x, y, z, radius)` layout and has the usual
V2 `setup_batch_tensors`, `update_num_spheres`, `reset`, `validate_input`,
`_discrete_fn`, `_sweep_fn`, and gradient-buffer lifecycle.

`SceneCollisionCostCfg` compiles its scalar activation distance onto its
`DeviceCfg` and rejects negative, non-finite, or vector activation distances
before a solve begins. It also validates non-negative sphere counts and the
configured checker protocol, then derives the checker count from native
`SceneCollision` instances. Query-compatible custom checkers remain supported
for testing and application extensions. `use_sweep_kernel` is retained as a
V2 configuration option but selects the portable swept-query route; it does
not expose Warp or CUDA kernel handles.

Signed per-sphere clearances are converted to
`0.5 * max(activation_distance - clearance, 0)^2`, then reduced with either
`sum_distance=True` or the deterministic first `max` reduction.  The cost
weight is applied once after that reduction.  A custom checker which already
returns `[batch, horizon]` is accepted as an already-aggregated loss.  Binary
mode preserves cuRobo's positive-collision offset before applying the weight.

Swept queries route to the checker’s fixed-resolution swept query and support
the portable speed metric.  They are sampled collision checks, not analytic
continuous collision detection.  Raw Warp kernel handles, Warp gradient ABI,
and CUDA graph capture remain unavailable on Metal.

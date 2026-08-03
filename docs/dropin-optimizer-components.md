# Optimizer component compatibility

`curobo._src.optim.components.ActionBounds`, `BestTracker`, and
`DebugRecorder` provide device-resident, eager-PyTorch versions of the V2
component lifecycle on CPU and float32 MPS.

`ActionBounds` exposes both V2 packed attributes (`horizon_lows`,
`horizon_highs`, `step_max`, and `horizon_step_max`) and the rank-two aliases
used by the portable solvers. Refreshing observes both a changed horizon and
changed mutable rollout limits.

`BestTracker` allocates per-problem cost/action/iteration/convergence buffers,
implements V2's strict absolute-and-relative best-cost update condition, and
supports in-place masked clearing. It deliberately uses ordinary PyTorch
tensors rather than CUDA graph buffers.

`DebugRecorder` returns V2's `{"debug": actions, "debug_cost": costs}`
mapping. Its snapshots are detached clones so debug collection does not retain
autograd graphs through long-lived MPC sessions.

These components are portable behavioral implementations, not CUDA/Warp ABI
or NVIDIA numerical-parity claims.

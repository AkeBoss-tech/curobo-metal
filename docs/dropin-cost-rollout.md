# Cost and rollout compatibility

The `curobo._src.cost` configuration and cost classes and the
`curobo._src.rollout` result, aggregation, goal-registry, robot-rollout, and
Rosenbrock interfaces are available without CUDA. Their executable paths use
ordinary differentiable PyTorch operations and therefore run on CPU and
fallback-disabled Apple MPS.

The public `curobo.rollout` module exports `RosenbrockCfg` and
`RosenbrockRollout` as pinned cuRobo V2 does. CUDA graph capture and raw Warp
kernel functions are implementation mechanics rather than portable public
semantics; this package does not claim those CUDA execution details.

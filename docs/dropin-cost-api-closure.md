# Portable cost API closure

`curobo._src.cost` has executable CPU and Apple-MPS implementations for the
pinned cuRoboV2 cost configurations and lifecycle methods. Cost instances keep
an immutable configuration weight and a separate mutable execution weight, so
`disable_cost()`, `enable_cost()`, batch setup, reset, and dt updates are safe
to use from persistent solver objects.

The portable implementations cover differentiable joint-limit/target costs,
state and torque regularization, indexed terminal/non-terminal c-space
distance, multi-tool pose-goalset selection, discrete/swept checker routing,
configured self-collision pairs, and 2-D support-polygon costs. They operate
on ordinary PyTorch tensors and preserve device residency on fallback-disabled
MPS for float32 tensors.

Derived configuration `clone()` calls retain their concrete type and clone
tensor-backed limits, targets, and criteria.  `ToolPoseCost` supports in-place
updates for a subset of configured tools, terminal/non-terminal convergence
tolerances, and goal-frame projection for axis-selective Cartesian approach
criteria.  Its public module returns the upstream interleaved
`[position_cost, rotation_cost]` channels per tool (`[B,H,2*links]`), accepts a
single-horizon goalset for an arbitrary current horizon, and supports either
`[B]` or `[B,1]` goal-batch indices.  Its diagnostic buffers are reusable
detached observations; returned costs remain native differentiable PyTorch
tensors, including gradients to selected goals.  Swept scene cost forwards
the public speed-metric and gradient-input controls to the portable collision
checker and validates the configured batch/sphere lifecycle.

`CSpaceCostCfg` additionally validates finite, non-negative component and
target weights, c-space dimensionality, and all joint-limit fields required by
the selected POSITION or STATE mode before a solve begins.  It accepts a
`RobotStateTransition`-style initializer and converts a STATE configuration to
the corresponding two-term POSITION form in teleport mode, matching the
pinned public lifecycle.  A missing `cost_type` remains a legacy
scalar-POSITION convenience; new code should set the type explicitly.

The portable C-space evaluator does not implement the CUDA/Warp rollout
retiming kernels.  Configurations requesting `retime_weights` or
`retime_regularization_weights` therefore raise a precise error at creation
instead of silently running with incorrect weights.

Raw Warp launch helpers remain deliberately unavailable. In particular, this
does not claim CUDA packed-buffer ABI compatibility, CUDA Graph capture,
Warp-specific tie layouts, or numerical identity with the NVIDIA kernels.
`ToolPoseCost` and the high-level cost classes are the supported differentiable
portable interfaces; direct raw Warp entrypoints fail with an explicit error.
The native autograd path deliberately does not reproduce Warp's
`use_grad_input=False` custom-backward override; normal PyTorch loss scaling is
preserved instead.

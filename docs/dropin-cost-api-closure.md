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

Raw Warp launch helpers remain deliberately unavailable. In particular, this
does not claim CUDA packed-buffer ABI compatibility, CUDA Graph capture,
Warp-specific tie layouts, or numerical identity with the NVIDIA kernels.
`ToolPoseCost` and the high-level cost classes are the supported differentiable
portable interfaces; direct raw Warp entrypoints fail with an explicit error.

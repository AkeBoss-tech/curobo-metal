# Portable State Filter

`curobo._src.util.state_filter` provides the pinned `FilterCfg` and
`JointStateFilter` command-filter surface using standard PyTorch tensors on
CPU and Apple MPS.  The filter owns a persistent command state, applies
per-channel low-pass coefficients, supports position, velocity,
acceleration, jerk integration, broadcastable action tensors, reset, and
explicit batch/shape validation.  Enabled filters transfer ingress states to
their `DeviceCfg`; disabled filters are exact input-object pass-throughs.

The portable implementation preserves normal first-order PyTorch autograd
through filtering and integration.  It deliberately does not emulate CUDA
graphs, streams, raw packed state buffers, or CUDA ABI kernels.

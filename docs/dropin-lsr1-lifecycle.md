# Portable L-SR1 lifecycle

`curobo._src.optim.gradient.lsr1.LSR1Opt` is a batched eager PyTorch
implementation for CPU and MPS. It uses independent, newest-first `(s, y)`
histories per problem and evaluates the L-SR1 inverse-Hessian rank-one updates
on the action tensor's device. Invalid, non-finite, zero-motion, or
ill-conditioned pairs are discarded independently, so one problem cannot
poison another batch member.

The optimizer shares the portable L-BFGS lifecycle: fixed candidate line
searches with a no-op candidate, best-action selection, action bounds,
terminal locking, warm starts, masked reinitialization, shifting, resizing,
solver-parameter validation, debug traces, and regular PyTorch autograd. An
oversized history is reduced to the useful action-dimensional limit, matching
the pinned solver's intended bounded-memory behavior.

The helper accepts either portable `[batch, history, dimension]` buffers or
the pinned kernel-shaped `[history, batch, dimension, 1]` buffers. CUDA Graph
capture, raw CUDA/Warp kernels, pointer/buffer ABI behavior, and exact NVIDIA
numerical parity are intentionally not provided.

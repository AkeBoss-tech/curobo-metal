# TrajOpt result compatibility

`curobo._src.solver.solver_trajopt_result.TrajOptSolverResult` is a portable
CPU/MPS result value.  It preserves the public batched `[batch, seed, ...]`
selection and copy lifecycle: successful candidate copies include every
materialized `JointState` channel, cloning does not alias tensor/debug state,
and interpolated plans are trimmed to their stored final timestep when every
returned trajectory has a common length.

Seed selection is deterministic: equal costs retain original seed order.  The
result carries ordinary PyTorch tensors and therefore works on `mps:0` without
CPU fallback.  Raw CUDA graph handles, packed rollout result buffers, and
CUDA-only metrics allocator/ABI operations are not emulated; stale raw metric
views are cleared after portable top-k selection.

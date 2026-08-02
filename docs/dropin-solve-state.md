# Solver-state compatibility

`curobo._src.solver.solve_state` is the portable, tensor-free description of
the active solve shape.  It is shared by the CPU and MPS implementations; it
does not own device buffers, CUDA graph capture, streams, or raw CUDA ABI
objects.

`SolveState` follows pinned cuRobo V2 seed precedence exactly: an explicit
`num_seeds` wins, otherwise IK, TrajOpt, and graph seed counts are selected in
that order.  Multi-environment mode derives from `num_envs`, and any
multi-problem request is batch mode.  An explicitly supplied `batch_mode=True`
on a one-problem request remains true, matching V2.

`get_batch_size()` is intentionally only valid after a generic seed count has
been resolved, exactly as upstream.  The IK and TrajOpt-specific accessors
instead return zero when that stage is not configured.  `clone()` preserves
all scalar metadata and the supplied `tool_frames` list reference, as the V2
implementation does.

The execution layers allocate CPU or MPS tensors from this metadata.  CUDA
graph capture and raw CUDA/Warp execution are outside the portable API.

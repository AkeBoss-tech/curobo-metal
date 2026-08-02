# Portable quasi-Newton and line-search compatibility

`curobo._src.optim.gradient.lbfgs`, `conjugate_gradient`, `lsr1`, and
`line_search_strategy` provide deterministic, batched PyTorch implementations
on CPU and Apple MPS.  L-BFGS uses the production portable L-BFGS stack;
nonlinear conjugate gradient supports Fletcher-Reeves, Polak-Ribiere, and
Dai-Yuan directions; L-SR1 keeps a guarded limited-memory inverse-Hessian
history.  Greedy, Armijo, weak/strong Wolfe, and approximate Wolfe candidates
are selected per problem with stable first-tie behavior.

The portable path executes fixed candidate line searches using ordinary tensor
operations and maintains tensors on the caller's device.  It rejects invalid
history, scale, and CUDA-graph settings early.  CUDA kernel line search,
shared-memory two-loop kernels, CUDA Graph capture, and raw Warp/NVIDIA ABI
entry points are not emulated; their flags are normalized to the portable
PyTorch path or raise at the graph-capture boundary.

This is behavioral portability, not a claim of bitwise CUDA equivalence.
Exact reductions, line-search ties, and low-level buffer layout still require
paired replay on a pinned NVIDIA environment to compare.

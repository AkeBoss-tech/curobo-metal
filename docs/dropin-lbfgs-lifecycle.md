# Portable L-BFGS lifecycle

`LBFGSOpt` uses a per-problem, device-resident limited-memory `(s, y, rho)`
history and an ordinary PyTorch two-loop recursion. Curvature pairs with no
motion, non-finite values, or insufficient positive curvature are ignored per
problem. Candidate scales are evaluated deterministically on CPU/MPS and the
best finite action is retained.

The optimizer supports direct batched seeds, masked warm-start
`reinitialize`, history reset/shift/resize, optional terminal-action locking,
bounded actions, finite convergence rules, and validated runtime updates.
CUDA graph capture, raw CUDA line-search/two-loop kernels, packed rollout
buffers, and exact CUDA `APPROX_WOLFE` execution remain explicit unsupported
boundaries; the portable path uses finite candidate selection instead.

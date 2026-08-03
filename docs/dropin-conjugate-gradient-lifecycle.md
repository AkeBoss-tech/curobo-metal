# Portable conjugate-gradient lifecycle

`curobo._src.optim.gradient.conjugate_gradient.ConjugateGradientOpt` implements
batched nonlinear conjugate gradient in ordinary PyTorch on CPU and
fallback-disabled float32 MPS.

## Supported behavior

- Fletcher-Reeves, Polak-Ribiere, and Dai-Yuan beta rules with finite,
  nonnegative beta clamping.
- Deterministic batched candidate line searches: greedy, Armijo, weak/strong
  Wolfe, and approximate Wolfe variants.  Candidate selection uses stable
  first ties; Armijo/Wolfe modes select the largest accepted configured scale.
- Box-bound projection, optional terminal-control locking, finite-cost best
  fallback, flattened `[B, H*D]` and `[B, H, D]` seeds, and execution inside a
  caller's `torch.no_grad()` scope.
- Persistent CG histories across warm solves, plus explicit reinitialize,
  reset, shape-reset, batch-resize, and MPC horizon-shift behavior.  History
  shifts flatten `[H,D]` controls before advancing by `shift_steps * D`.

## Explicit boundaries

The CUDA kernel line-search flag is accepted for configuration compatibility
but resolves to the device-resident PyTorch candidate loop. CUDA graph capture,
Warp kernels, packed rollout ABI buffers, and CUDA stream ownership are not
implemented on the portable backend. Numerical equivalence to NVIDIA CUDA is
not claimed without paired replay evidence.

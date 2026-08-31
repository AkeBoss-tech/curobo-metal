# Portable optimizer parity

Wave 9B completes two device-resident optimizer choices shared by IK and
trajectory optimization: deterministic particle evolution (`particle`, with
`es` as the compatibility alias) and limited-memory BFGS (`lbfgs`). Particle
execution supports CEM, MPPI, and random sampling, optional covariance,
temperature, and bounded debug histories. L-BFGS supports fixed-step, Armijo,
and strong-Wolfe searches plus selectable termination. Existing callers retain
projected Adam because `adam` remains the production problem default.

The observable contract is batched. Every seed has its own objective,
convergence/failure bit, iteration count, and status. Empty seed batches are
valid. Particle streams are reproducible from an explicit CPU-seeded generator
and transferred to the requested device; they do not depend on the global
PyTorch RNG. L-BFGS maintains bounded `(s, y, rho)` history, applies a two-loop
inverse-Hessian recursion, and uses the configured line search. Projection is applied
inside line search or population evaluation.

`ExecutionCache` persists entries by optimizer, exact shape, dtype, device,
event rank, and immutable options. It supports warm-started final solutions,
bounded capacity, update, and reset generations. Graph planning uses the same
lifecycle object to retain deterministic sample tensors. These objects are
ordinary portable Python/Torch caches: they are explicitly not CUDA Graph
captures, and changing data with the same shape still reevaluates objectives
and validity.

## Upstream-compatible choices and intentional mechanism differences

The pinned cuRobo main exposes particle/ES and L-BFGS optimizer selection,
per-seed results, warmup/reset lifecycles, and graph-oriented execution reuse.
This package preserves those useful option/result behaviors while differing in
mechanism:

- no CUDA kernels, CUDA Graph capture, Warp, or stream-specific graph replay;
- portable CEM/MPPI/random evolution instead of upstream CUDA particle kernels;
- pure Torch batched L-BFGS line search instead of upstream fused kernels;
- shape-keyed execution/sample reuse instead of captured graph executables;
- `es` routes to the deterministic particle implementation;
- default direct operator behavior remains projected Adam for release stability.

The compatibility compiler now accepts `LBFGS`, `PARTICLE`, and `ES` for IK and
trajectory configuration and carries graph cache options into `MotionGenConfig`.
Trajectory execution consumes the selected optimizer. IK problem callers can
select the same choices directly; joint-space MotionGen does not invoke IK.
The audit classifies these portable records as semantically equivalent after
the 2026-08-26 paired pinned replay passed their multi-seed, termination,
invalid, edge, and expanded matrix checks. It does not claim identical CUDA
RNG samples, solver iterates, or fused-kernel execution.

## Verification

`tests/ops/optim/test_optimizers.py` covers analytic quadratics, projection,
autograd, nonfinite failures, empty batches, determinism, cache lifecycle, warm
starts, and fallback-disabled MPS when available. Existing IK, trajectory,
graph-planning, and API compatibility suites guard integration behavior.
`benchmarks/optim/benchmark_optim.py` reports first-call (“compile”), warmup, and
steady-state timing without implying CUDA graph capture.

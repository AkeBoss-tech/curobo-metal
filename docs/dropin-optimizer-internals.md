# Portable optimizer internals

This compatibility layer implements the pinned cuRobo V2 optimizer component,
gradient, particle-sampling, factory, iteration-state, protocol, and
Levenberg–Marquardt module paths on CPU and Apple MPS.

Executable behavior uses ordinary PyTorch operations and the production
`curobo_metal.optim` stack. It preserves batched tensors, autograd, deterministic
CPU-seeded sampling, reusable optimizer state, reset/update methods, and
device-resident outputs. The public high-level solvers continue to use the
portable particle and L-BFGS implementations.

The CUDA-named implementation details are compatibility boundaries, not
emulations. CUDA Graph capture, raw CUDA pointer kernels, Warp LM kernels, and
NVIDIA ABI entry points raise `NotImplementedError`. The portable execution
cache is shape-keyed state and is not presented as CUDA Graph capture.

Numerical equivalence to the pinned NVIDIA implementation is not claimed
without paired replay. Floating-point reduction order, random-number streams,
line-search tie handling, and low-level buffer layout may differ even when the
portable optimizer has the same mathematical contract.

`ParticleOptCore` also owns a real portable particle lifecycle: deterministic
batched populations, sampled/negative/null action composition, bounded action
projection, callback-owned distribution updates, best/sample/mean extraction,
partial reset, warm-start shifting, batch resize, and debug traces.  It uses
ordinary PyTorch CPU/MPS tensors.  CUDA graph capture, Warp samplers, and the
packed CUDA rollout-result ABI remain deliberate unsupported boundaries.

`GradientOptCore` likewise provides a concrete eager CPU/MPS lifecycle for
owner-supplied L-BFGS, LSR1, and conjugate-gradient direction callbacks.  It
evaluates autograd costs and gradients, performs deterministic finite-candidate
per-problem line selection, tracks best finite actions and convergence, honors
partial reinitialization, resize and shift hooks, and records immutable debug
states.  CUDA Graph executors and packed CUDA rollout buffers remain explicit
unsupported boundaries.

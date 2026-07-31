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

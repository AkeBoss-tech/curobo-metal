# Low-level CUDA-named backend compatibility

This package preserves the pinned cuRoboV2 Python import surface under
`curobo._src.curobolib` while running supported tensor work through ordinary
PyTorch on CPU or Apple MPS. The names are compatibility seams; they do not
claim that Metal implements CUDA's runtime ABI.

Supported low-level behavior includes tensor validation, self-collision
distance dispatch, acceleration integration, position differentiation,
L-BFGS step buffers, line-search selection, launch/config calculations,
synchronization-correct timing, and shape-stable direct graph execution.
Higher-level FK, Jacobian, collision, dynamics, interpolation, and optimization
remain available through their public production APIs.

The following CUDA concepts deliberately raise `NotImplementedError`: runtime
CUDA compilation, raw pointer kernel launch, CUDA stream wrapping, CUDA graph
capture/debug artifacts, packed CUDA FK/RNEA autograd buffers, PBA kernels, and
Warp initialization/device streams. This fail-closed behavior prevents a
CUDA-named import from silently promising backend equivalence it cannot supply.

`GraphExecutor` executes the supplied callable directly and resets when input
shapes change. CUDA-named stream context helpers are no-op ordering contexts
because PyTorch MPS preserves stream ordering without exposing CUDA streams.
`CudaEventTimer` synchronizes MPS around host timing and reports milliseconds.

Large Warp/CUDA kernels and exact NVIDIA numerical parity remain outside this
portable seam and require paired execution on the pinned NVIDIA runtime.

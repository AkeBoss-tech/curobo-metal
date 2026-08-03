# Portable ESDF Integrator Lifecycle

`BlockSparseESDFIntegrator` provides a dense, exact-EDT implementation behind
the pinned cuRobo block-sparse ESDF public facade.  It runs on CPU and on MPS
with float16 or float32 source storage (the MPS output field is float32).

The cached field is tied to both the mapper generation and the ESDF window
origin.  `compute_esdf()` returns the cached field only when both still match;
`get_voxel_grid()` and `query()` refresh stale fields before returning data.
Static-scene updates, integration, imports, clears, and reset invalidate the
field and nearest-site diagnostic buffer.  Reset restores the initial empty
state and resets the public compute counter.

The portable window implementation supports only integer-voxel translations of
the fixed dense backing grid.  Fractional shifts and changed voxel sizes raise
`NotImplementedError` before changing cache state.  Tensor origins, voxel
sizes, and query points must reside on the integrator device; this validation
treats `mps` and `mps:0` as the same device.

This is not the CUDA/Warp sparse PBA/JFA implementation.  CUDA graph capture,
raw Warp buffers and kernels, sparse hash-table scheduling, and a resampling
backend remain explicit unsupported boundaries rather than claims of ABI or
numerical parity.

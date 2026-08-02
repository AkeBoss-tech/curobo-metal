# Portable PRM node sampling

`curobo._src.graph_planner.graph.node_sampling_strategy.NodeSamplingStrategy`
uses the portable, seeded Halton `SampleBuffer` on CPU and Apple MPS.  It
retains resettable sampling state, bounded/unbounded samples, filled low-
dimensional unit-ball samples, feasibility filtering, segment distance, and
all three pinned ellipsoid method names.

The Householder and approximate projections use regular PyTorch tensor
operations.  The upstream `svd` option is mapped to the equivalent
Householder-frame projection on MPS because `aten::linalg_svd` is not a native
MPS operation; this deliberately avoids a CPU fallback.  The output remains
bounded by the c-space limits and is filtered by the caller's feasibility
callback.

This does not expose upstream CUDA graph capture, Warp random-kernel ABI, or
their packed CUDA buffer layout.  Sampling is deterministic after `reset_seed`
on a given backend, but CUDA/Metal sequence parity is not claimed.

# State, transition, and sampling compatibility

This wave provides the pinned cuRoboV2 import paths for joint-state operations,
trajectory indexing, state filtering, position/acceleration transitions, tensor
helpers, and deterministic random/Halton/Roberts sample buffers.

All executable tensor paths use ordinary PyTorch operations and preserve CPU or
MPS device residency and first-order autograd. CUDA graph/stream management is
owned by a separate backend wave. B-spline transitions use portable linear
interpolation and do not claim numerical identity with upstream Warp kernels.

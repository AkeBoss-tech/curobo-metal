# Portable planning API surface

The planning, motion, rollout, graph, and solver package roots re-export their
portable CPU/MPS implementations.  Tensor-level roadmap helpers and metric
selection operations preserve autograd/device behavior.  CUDA graph reset and
raw CUDA/Warp execution remain explicit `NotImplementedError` boundaries; they
are not silently emulated on Metal.

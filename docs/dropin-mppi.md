# Portable MPPI compatibility

`curobo._src.optim.particle.mppi.MPPI` implements the pinned V2 MPPI
distribution lifecycle with ordinary PyTorch tensors on CPU and Apple MPS.
It keeps a batched action mean, `DIAG_A` and `SIGMA_I` covariance modes,
softmax cost weighting, the V2 sampled/negated/null particle mix,
deterministic fixed or per-iteration sample schedules, warm-start shifts,
reset/reinitialize operations, and recorded rollout/debug state.  When
`sample_per_problem=False`, all batch items receive the same reproducible
perturbation population; otherwise each receives an independent deterministic
population. Rollout actions and all distribution state stay on the requested
PyTorch device.

The following upstream backend details deliberately remain unsupported rather
than being misrepresented: CUDA graph capture/replay, Warp sampling kernels,
packed CUDA rollout/result buffers, and CUDA-specific stream/allocator timing.
`use_cuda_graph=True` and `reset_cuda_graph()` therefore raise a clear
`NotImplementedError`; portable persistent state is not a CUDA graph.

The implementation supports callable rollouts (or `objective`/`cost_fn`
objects) that return one scalar or a per-horizon cost sequence for every action
sequence.  It does not claim numerical identity with NVIDIA CUDA/Warp kernels;
that requires paired pinned-upstream CUDA replay evidence.

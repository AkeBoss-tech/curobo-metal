# MPPI lifecycle

The portable `MPPI` implementation keeps its distribution, covariance, sample
schedule, best trajectory, and debug/rollout values as ordinary PyTorch CPU or
MPS tensors.  It accepts the pinned V2 `MPPICfg` construction surface,
including factory dictionaries, MPPI-specific particle fields, enum strings,
and `ParticleSamplerCfg`.

An explicit particle-sampler seed controls its deterministic sample population;
the config seed remains the default.  Updating particle count, sampler
settings, or iteration count invalidates cached populations.  Updating
covariance/initial mean reinitializes the materialized distribution, and every
parameter update re-runs validation.

`shift()` advances both the distribution mean and the independently recorded
best trajectory.  It supports repeat, null, and deterministic random tail
policies for MPC warm starts.  CUDA graph capture, Warp sampling/kernel ABI,
and packed CUDA rollout buffers deliberately remain unsupported.

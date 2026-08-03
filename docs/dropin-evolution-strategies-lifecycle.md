# Portable evolution-strategies lifecycle

`curobo._src.optim.particle.evolution_strategies.EvolutionStrategies` is a
device-resident PyTorch implementation of the cuRobo V2 ES control loop.  It
uses z-score utilities and a natural-gradient mean update, while sharing the
portable particle-distribution lifecycle with MPPI.

## Supported portable behavior

- CPU and fallback-disabled float32 MPS execution.
- Deterministic CPU-seeded population generation, including independent or
  shared per-problem sampling according to `sample_per_problem`.
- `ParticleSamplerCfg.fixed_samples`: one cached perturbation population is
  reused at every iteration; otherwise one population per configured iteration
  is cached and consumed in order.
- Pinned particle order: stochastic samples (the final one is exactly the
  current mean), then negated-mean actions, then null actions.
- Batched seeds, persisted mean/covariance/best-action state, warm starts,
  shifts, reset/reinitialize, rollout recording, and `BEST`, `MEAN`, and
  `SAMPLE` selection.
- `SAMPLE` returns one projected action per problem and does not reset the
  distribution or advance the population-sample cursor.

## Explicit boundaries

CUDA graph capture, Warp samplers, raw packed CUDA rollout buffers, and CUDA
ABI kernels are not implemented.  These use the shared portable execution and
shape cache instead.  The implementation is deterministic for a fixed PyTorch
release and seed, but it does not claim numerical or random-stream identity
with NVIDIA CUDA/Warp execution without paired replay evidence.

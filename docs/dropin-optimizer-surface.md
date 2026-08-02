# Portable gradient and ES optimizer surface

`curobo._src.optim.gradient.gradient_descent.GradientDescentOpt` and
`curobo._src.optim.particle.evolution_strategies.EvolutionStrategies` execute
ordinary PyTorch CPU/MPS tensor operations.  Their public configuration,
lifecycle, best-result, distribution, deterministic sampling, and utility
interfaces are available to portable consumers.

The portable ES implementation exposes V2-compatible z-score utilities and
natural-gradient mean updates, while the search itself uses the project’s
deterministic particle optimizer.  Its covariance update and random stream are
not claimed numerically equivalent to NVIDIA's Warp/CUDA implementation.

`GradientOptCore` retains callback and lifecycle semantics for higher-level
optimizers.  `use_cuda_graph=True`, raw CUDA graph dumps, and CUDA/Warp
line-search ABI assumptions remain explicit unsupported boundaries; portable
execution instead maintains regular reusable shape state.

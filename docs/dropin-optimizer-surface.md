# Portable gradient and ES optimizer surface

`curobo._src.optim.gradient.gradient_descent.GradientDescentOpt` and
`curobo._src.optim.particle.evolution_strategies.EvolutionStrategies` execute
ordinary PyTorch CPU/MPS tensor operations.  Their public configuration,
lifecycle, best-result, distribution, deterministic sampling, and utility
interfaces are available to portable consumers.

The portable ES implementation runs its own deterministic, device-resident
search: z-score utilities, natural-gradient mean updates, covariance updates,
best/mean/sample return modes, rollout capture, warm starts, shifts, and
resizing are all real PyTorch CPU/MPS behavior.  Its covariance weighting and
random stream are not claimed numerically equivalent to NVIDIA's Warp/CUDA
implementation; raw CUDA graph capture and packed rollout ABIs remain outside
this portable surface.

`GradientOptCore` retains callback and lifecycle semantics for higher-level
optimizers.  `use_cuda_graph=True`, raw CUDA graph dumps, and CUDA/Warp
line-search ABI assumptions remain explicit unsupported boundaries; portable
execution instead maintains regular reusable shape state.

The gradient-descent facade additionally accepts V2's flattened and
``[problem, horizon, action_dim]`` seeds, runs under an enclosing
``torch.no_grad()`` scope, applies rollout-published hard action bounds,
maintains finite best-action fallback and per-problem non-fixed convergence,
and exposes reset/shift/metric/timing lifecycle methods.  It uses the V2
``gradient_descent_step_scale`` exclusively for the GD update; ``step_scale``
remains a rollout-bound configuration value rather than an alternate learning
rate.  CUDA graph capture and packed rollout-buffer ABI entry points are not
implemented on CPU/MPS.

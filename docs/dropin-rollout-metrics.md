# Portable rollout metrics

`curobo._src.rollout.metrics` provides the V2 rollout value models on CPU and
Apple Metal.  Cost, constraint, hybrid, and convergence terms are regular
differentiable PyTorch tensors with `[batch, horizon, component]` and
`[batch, seed, horizon, component]` layouts.  Aggregation preserves leading
batch/seed dimensions, reduces component values, and optionally reduces the
final horizon axis.

`CostCollection`, `CostsAndConstraints`, `RolloutResult`, and
`RolloutMetrics` support cloning, ordinary indexing, batch/seed selection, and
in-place selected-buffer updates.  Every populated rollout channel (actions,
state, feasibility, cost/constraint collections, and convergence) participates
in these lifecycle operations.  Weight metadata added through `add` stays
aligned with its named term; missing squared weights use an all-ones portable
weight when constraint weights are requested.

`CostCollectionSum` remains as the narrow explicit-VJP compatibility helper
for optimizer internals.  It is implemented using `torch.autograd.Function`;
normal aggregation uses PyTorch autograd directly.  CUDA graph capture,
streams, and packed CUDA buffer ABI are intentionally not represented.

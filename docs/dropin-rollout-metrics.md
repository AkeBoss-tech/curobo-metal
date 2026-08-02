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

All four value models also provide portable `to(...)` and `detach()` methods.
They recursively move or detach tensors in cost terms and ordinary nested debug
payloads, while a `DeviceCfg` preserves boolean/integer rollout metadata rather
than casting it to the configured floating dtype.  `JointState` retains its own
device configuration through the same operation.  An unpopulated
`RolloutResult` is an empty Python container (`len(result) == 0`), not the
invalid negative-length sentinel from the upstream implementation.

`CostCollectionSum` remains as the narrow explicit-VJP compatibility helper
for optimizer internals.  It is implemented using `torch.autograd.Function`;
normal aggregation uses PyTorch autograd directly.  CUDA graph capture,
streams, and packed CUDA buffer ABI are intentionally not represented.

# Rollout metrics lifecycle

Portable `CostCollection`, `CostsAndConstraints`, `RolloutResult`, and
`RolloutMetrics` retain batch, optional seed, horizon, and component axes on
CPU and MPS. Metric terms use `[..., horizon, component]`; a legacy
batch-scalar `[batch]` term is treated as one horizon and one component, so
horizon aggregation cannot collapse its batch axis. Collections require a
shared batch/seed and horizon layout and fail clearly for ambiguous mixtures.

Result indexing and `get_only_batch_seed_indices` apply the same selection to
nested debug tensors, while scalar/string debug metadata is kept unchanged.
The objects remain differentiable ordinary PyTorch values and can be cloned,
detached, copied, and moved to MPS without CPU fallback. CUDA packed metric
buffers, graph capture, and raw Warp/CUDA ABI behavior remain intentionally
outside this portable contract.

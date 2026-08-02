# IK solver result compatibility

`curobo._src.solver.solver_ik_result.IKSolverResult` retains the pinned V2
dataclass field layout and adds portable lifecycle helpers for results produced
on CPU or Apple MPS.

`clone()` deep-copies tensors, `JointState` channels, and nested diagnostic
metadata. `get_topk_seeds()`, `best_seed()`, and `select_seed_indices()` retain
the public `[batch, seed, ...]` rank contract; `select_batch()` preserves a
leading batch axis. `process_metrics_and_rank_seeds()` uses a stable cost order
and respects an available feasibility mask. `to(DeviceCfg(...))` moves floating
values to the requested dtype/device while retaining boolean success flags and
integer seed/goal indices.

These helpers do not emulate CUDA graph result buffers, packed pointer views,
or NVIDIA numerical parity. They operate exclusively on materialized portable
PyTorch tensors and `JointState` values.

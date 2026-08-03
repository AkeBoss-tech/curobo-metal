# TrajOpt result lifecycle

`TrajOptSolverResult` is a regular PyTorch CPU/MPS value object.  It preserves
the public `[batch, seed, ...]` result convention for cloning, device moves,
batch selection, seed selection, successful-result merging, trajectory timing,
and nested tensor debug metadata.

Portable TrajOpt may materialise only `return_seeds` trajectories while the
optimizer initially retains a larger attempted-seed cost table.  Calling
`process_metrics_and_rank_seeds()` normalizes that boundary: costs, optimized
seeds, and per-seed debug tensors are narrowed to the returned trajectories;
`seed_rank` retains the original optimizer IDs for traceability.  Later
`best_seed()` and `get_topk_seeds()` rank by the local returned costs, so these
original IDs are never mistaken for local tensor indices.

`motion_time()` accepts scalar, batch, and batch/seed `dt` values.  When `dt`
has the trajectory prefix (one value per waypoint), it sums the `horizon - 1`
actual intervals.  `interpolated_last_tstep == 0` preserves V2's full-horizon
sentinel; unequal per-result interpolation lengths remain explicitly ragged
and cannot be represented as one `JointState`.

The compatibility layer does not reproduce CUDA graph handles, raw packed
rollout buffers, Warp ABI calls, or CUDA allocator-backed metric views.

# Portable PRM graph behavior

`PRMGraphPlanner` preserves the useful stateful behavior of the pinned V2
roadmap API on CPU and Apple Metal. Calls to
`extend_roadmap_with_random_samples` and
`extend_roadmap_with_ellipsoidal_samples` retain a deterministic, bounded
vertex buffer. This is real planning state: the next `find_path` passes those
vertices to the production graph-planning operation, which rechecks their
feasibility for every query. `n_nodes` reports retained vertices,
`reset_buffer` clears them, and `reset_seed` rewinds random sampling without
discarding an explicitly built roadmap.

A batched query with an infeasible start or goal follows V2's
batch-invalid result behavior. PRM waypoint interpolation supports `LINEAR`,
`CUBIC`, and `QUINTIC`. The PRM-specific `LINEAR_CUDA` and
`BSPLINE_KNOTS_CUDA` modes are rejected rather than silently claiming original
CUDA-kernel semantics. `warmup` performs ordinary CPU/MPS planning and leaves
a clean buffer; `reset_cuda_graph` remains an explicit unavailable boundary.

Roadmap edge checking is sampled and deterministic, not analytic continuous
collision detection. CUDA graph capture, Warp steering kernels, and CUDA
SVD/Householder ellipsoid kernels are intentionally not represented as drop-in
portable ABIs.

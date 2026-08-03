# PRM indexed-roadmap lifecycle

`curobo._src.graph_planner.graph_planner_prm.PRMGraphPlanner` now keeps an
inspectable deterministic graph for roadmap nodes explicitly added through
`extend_roadmap_with_random_samples` or
`extend_roadmap_with_ellipsoidal_samples`.  The private V2 hooks
`_find_path_for_index_pairs` and `_check_paths_exist` therefore work with
persistent node indices, including batched pairs, partial connectivity, stable
weighted lengths, reset, and seed lifecycle.

Candidate neighbour ordering and shortest-path traversal are CPU control-plane
work.  Roadmap vertices and feasibility/edge checks remain CPU or MPS tensors,
and use the same callback as normal production planning.  The persistent
inspection graph is intentionally bounded by the configured PRM capacity and
does not represent CUDA graph capture, Warp neighbour kernels, or analytic
continuous collision detection.  Those raw CUDA/Warp surfaces remain explicit
unsupported boundaries; `find_path` continues to route actual terminal-query
planning through the production portable graph operator.

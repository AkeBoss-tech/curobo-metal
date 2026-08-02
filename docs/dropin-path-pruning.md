# Portable PRM path pruning

`curobo._src.graph_planner.search.path_pruner.PathPruner` preserves the V2
shortcut lifecycle on CPU and Apple Metal.  Install its dependencies from the
active PRM graph, pass the original integer node paths, and it will batch every
ordered forward pair (including existing/self pairs), delegate collision-aware
edge registration, and rerun the supplied shortest-path query.

The candidate edge tensor has shape `[E, 2, action_dim + 1]` and remains on
the configured CPU or MPS device until the graph callback's normal
control-plane boundary. Empty input path batches are a no-op. Disconnected
results are deliberately returned exactly as the graph finder represents them;
pruning never substitutes an old path for a fresh failure.

This is discrete graph shortcutting. Warp/CUDA graph capture, raw CUDA edge
kernels, and analytic continuous collision detection are not implemented.

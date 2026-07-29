# Wave 4C graph-planning scope

Wave 4C adds the backend-neutral contract and slow deterministic CPU oracle for
joint-space geometric planning. Its purpose is correctness, replay, and a
stable seam for a future device implementation.

Included:

- batched start/goal queries with a backend-neutral validity callback;
- deterministic seeded uniform sampling;
- k-nearest and radius-limited undirected roadmaps;
- discretized swept edge validation;
- deterministic A* and Dijkstra extraction;
- greedy shortcutting and bounded-step interpolation;
- explicit endpoint, disconnection, and search-limit outcomes;
- replay fixtures for direct motion, a 2-link joint-space detour, a narrow
  passage, and a disconnected query;
- Panda-class dimensional coverage and composition with existing NumPy
  FK/collision references.

Deferred:

- production Torch/MPS planning and custom Metal kernels;
- trajectory optimization, smoothing costs, timing, velocity/acceleration
  limits, dynamics, and execution;
- CUDA graphs, CUDA planners, Isaac/Omniverse, and upstream runtime imports;
- continuous-collision proofs, mesh/voxel worlds, experience graphs,
  multi-query roadmap caching, and asymmetric/directed costs.

The checked-in graph oracle intentionally favors explicit ordering and
auditable metrics over speed. `edge_step` is a collision discretization
parameter, not a guarantee between samples. A production backend may use a
different graph or algorithm if it satisfies the observable contract.

# Production geometric graph planning

`curobo_metal.ops.graph_planning` implements the Wave 5B joint-space roadmap
planner on CPU and Apple MPS. Graph topology, deterministic search, and seeded
PCG64 sampling remain CPU-controlled. Configuration validity—including
production forward kinematics, self collision, and cuboid-world collision—is
evaluated in device-resident batches. On MPS those batches select the existing
fused FK and Metal collision paths without enabling PyTorch CPU fallback.

The planner preserves the reference contract: each batch has an independent
`SeedSequence([seed, batch_index])`; nodes retain generated order; neighbor
choices use distance/index ordering; candidate edges are swept at
`edge_step`; adjacency, A*/Dijkstra tie handling, shortcutting, interpolation,
statuses, and metrics are deterministic. All candidate edge samples are
concatenated into one validity batch per roadmap. This discretized swept check
is not a continuous-collision proof.

Pass either a device-resident boolean `validity(q)` callback or a
`KinematicChain` and `CollisionModel`. The callback seam supports declarative
oracle fixtures and additional constraints. With no callback, validity
composes production FK and `robot_collision_cost`; clearance greater than or
equal to `-collision_tolerance` is valid. A shared collision model or one model
per query is supported.

`paths_to_trajectory_seeds` converts successful geometric paths to fixed-knot
`[B,1,T,J]` seeds using joint-space arc length. `plan_and_optimize` performs
that handoff directly to `optimize_trajectory`; it does not invoke trajectory
optimization if any graph query failed.

Run:

```bash
uv run --extra test pytest -q tests/ops/graph_planning
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --extra test pytest -q tests/ops/graph_planning
uv run python benchmarks/graph_planning/benchmark_graph_planning.py --device cpu
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/graph_planning/benchmark_graph_planning.py --device mps
```

Benchmark reports separate first-call warmup from synchronized repeated
latency and always includes status, path cost, accepted samples, edge checks,
and validity-query count.

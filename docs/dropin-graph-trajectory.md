# Graph and trajectory drop-in compatibility

This wave provides the pinned cuRoboV2 graph-planner, trajectory utility, and
trajectory-optimizer import paths without importing CUDA, Warp, or NetworkX.
`PRMGraphPlanner` routes batched feasibility and deterministic roadmap planning
through `curobo_metal.ops.graph_planning`; `TrajOptSolver.solve_cspace` routes
seeded trajectory optimization through `curobo_metal.ops.trajectory`.

The PRM facade now has a terminal-first persistent roadmap lifecycle: an empty
roadmap tries the direct edge, then deterministically grows feasible
ellipsoidal samples for an unresolved query up to the configured node and
iteration limits.  Added samples and their effective neighbor policy persist
across calls, `reset_buffer` restores the configured policy, and
`_find_path_impl` is available for the historical un-interpolated planning
entrypoint.  Near-identical terminals yield a zero-length two-knot plan under
the configured c-space similarity threshold.

The public enum values, result/config field names, primary method signatures,
seed generation, interpolation, mutable graph lifecycle, and command-buffer
lifecycle follow pinned revision
`8e734f3ced1df898990bcd92de40abce475907db`.

The historical names `LINEAR_CUDA` and
`get_cuda_linear_interpolation` execute portable PyTorch operations on CPU or
MPS. `TrajOptSolver` supports deterministic seed preparation/ranking, active
joint-name reduction, bounded c-space sampling, runtime tool-pose tracking,
finite-difference retiming, dense interpolation limits, serializable debug
state, c-space solves, and pose solves composed through the portable IK solver.
Calls requesting more returned plans than supplied seeds increase the number of
optimized seeds and results are ranked by actual trajectory objective.

The portable TrajOpt lifecycle also runs the requested initial, time-optimal,
and later finetune passes against the production trajectory operator. Each
accepted pass has a shorter shared knot duration and must remain feasible before
it replaces a seed. `SolveState`, result timing, goal-set indices, seed ranking,
and cache statistics are populated for every solve. The cache is ordinary
shape-keyed CPU/MPS optimizer state and resets on a structural solve-shape
change; it is not a CUDA graph. Robot-file retract states are materialized on
the configured device, so fallback-disabled MPS callers retain MPS tensors.

Raw Warp kernels, CUDA graph capture/reset, independent per-seed variable dt, and CUDA
B-spline-knot kernels remain explicit unavailable boundaries. The historical
`BSPLINE_KNOTS_CUDA` configuration falls back to endpoint-preserving linear
portable interpolation; it is not claimed to reproduce the CUDA spline kernel.
Sampled edge checks are deterministic but are not a continuous collision
certificate.

`GraphNodeManager` maintains the pinned action-plus-index node rows on CPU or
MPS: initial vertices have stable indices, exact/similarity duplicates map back
to the first compatible roadmap vertex, candidate batches preserve that mapping,
and registered paired edges retain their weighted connection records.  Its
materialized `ConnectedGraph` contains action-only nodes, action endpoint
pairs, connectivity triples, optional rollout states, and shortest-path debug
distances.  The raw CUDA graph/rollout buffers, Warp neighbour kernels, and
analytic CCD remain unavailable; graph connectivity uses the portable path
finder and its sampled checks.

`GraphConstructor` now composes those pieces into the pinned terminal lifecycle:
it validates device-resident batched inputs, installs and caches a feasible
default posture once per graph generation, assigns exact stable terminal
indices, creates bidirectional terminal/default edges, and expands each new
candidate to a deterministic nearest-neighbor batch.  Action-only connector
outputs are normalized back to action-plus-index rows before registration.  A
constructor reset intentionally clears only the default-node cache; it does not
silently discard the caller-owned roadmap.  CUDA graph capture, Warp steering,
and analytic CCD remain unavailable, so edge validity is whatever the supplied
portable connector and feasibility callback establish.

### Linear connector

`curobo._src.graph_planner.graph.connector_linear.LinearConnector` performs
endpoint-inclusive, weighted C-space interpolation in a single batched tensor
operation on CPU or MPS.  It asks the configured feasibility callback for every
sample and deterministically returns the sample immediately before the first
infeasible one (or the endpoint when all samples are feasible).  Connector rows
retain the pinned `[action..., graph_index]` layout; graph-index padding is
metadata and is not interpolated.  Resolution is
`cspace_similarity_threshold`, bounded explicitly by `steer_buffer_size`.

This is discrete swept validation, not analytic continuous collision detection.
Warp kernels, CUDA graph-owned buffers, and CUDA-only raw connector APIs remain
unsupported; existing production collision/rollout functions supply feasibility
on the same CPU/MPS device.

`GraphPlannerResult` retains the pinned variable-length per-query path list
and adds ordinary CPU/MPS result lifecycle helpers: cloning, detaching,
device/dtype movement, deterministic batch selection, success summaries, and
device-resident padded path tensors with validity masks. These helpers do not
expose or emulate raw CUDA graph/path buffers.

`NetworkXPathFinder` now preserves the pinned buffered roadmap lifecycle on
CPU and Apple Metal callers: Python, CPU-tensor, and MPS-tensor node/edge
identifiers are staged and materialized at a query/update boundary; repeated
undirected edges update their weight; graph reset also clears pending records;
and dense goal-distance slots retain `-1.0` for unreachable vertices. Equal
cost paths are resolved with a stable lexicographic node sequence, making PRM
results independent of edge insertion order. This search remains intentionally
CPU control-plane work, just as upstream NetworkX is; it is not a CUDA/Warp
kernel and accepts MPS roadmap tensors only at the scalar graph boundary.

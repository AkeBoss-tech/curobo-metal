# Backend-neutral geometric graph-planning contract

Status: Wave 4C executable reference, version 1.

## Scope and intent

`curobo_metal.reference.graph_planning` is a deterministic float64 NumPy oracle
for batched geometric seed generation in joint space. It is compatible in
intent with cuRobo's graph-planning stage: it finds a collision-valid geometric
path that can seed later optimization. It is not API or bitwise compatibility
with a particular upstream planner.

The oracle imports no PyTorch, MPS, CUDA, Warp, cuRobo, or Isaac runtime. It may
compose the existing NumPy FK and collision references through
`robot_collision_validity`. Production Torch/MPS planners, trajectory
optimization, time parameterization, dynamics, and CUDA/Isaac integration are
out of scope.

## Query, batching, and validity callback

Starts and goals have matching shape `[B,J]`; finite lower and upper limits have
shape `[J]`. Every batch item is planned independently in ascending batch
order. Returned arrays retain all `B` entries, including failures. Limits are
inclusive, and `lower < upper` is required in every joint.

The collision-validity callback receives a finite float64 array `[N,J]` and
must return a NumPy boolean array `[N]`, where true means the configuration is
valid. Ordering is preserved and callbacks must be pure: their answer may not
depend on call grouping, order, device state, or mutable global state. Invalid
callback shape or dtype is an error. The callback can represent self/world
collision, user constraints, or their conjunction.

## Sampling and roadmap construction

Each batch uses NumPy `PCG64(SeedSequence([seed,batch_index]))`. Exactly
`sample_count` uniform limit-box samples are generated in one call; invalid
samples are discarded without replacement or resampling. Node order is start,
goal, then accepted samples in generated order. Thus another batch does not
perturb an existing batch's stream.

For every node, candidate neighbors are sorted by `(Euclidean distance,
serialized node index)`. Radius filtering is applied first when configured,
then the first `k_neighbors` are retained when configured. The undirected
candidate graph is the union of per-node choices. At least one rule is
required. Candidate pairs are swept-validated in lexicographic node order.

## Swept validation, search, and post-processing

An edge from `a` to `b` is interpolated at
`ceil(max(abs(b-a))/edge_step)` equal intervals, with both endpoints included.
Every sampled configuration must be valid. This is a deterministic discretized
swept contract, not a proof of continuous collision freedom; callers choose
`edge_step` small enough for their geometry and safety margin.

Edge weight is joint-space Euclidean length. Dijkstra uses zero heuristic; A*
uses Euclidean distance to goal, which is admissible. Heap ordering, strict
distance improvement, serialized adjacency order, and first-parent retention
define exact-tie behavior. `max_search_expansions`, when set, bounds closed
non-goal nodes.

Shortcutting greedily tests the farthest remaining waypoint first and retains
the first valid edge. Output interpolation uses the smaller of
`interpolation_step` and `edge_step`, so every returned waypoint is among the
same discretized swept samples. Path cost is the sum of shortcut segment
Euclidean lengths; interpolation does not change it.

## Results, failures, limits, and metrics

Success status is `direct_success` when shortcut output contains only start and
goal, otherwise `success`. Failures are:

- `invalid_start` or `invalid_goal`;
- `disconnected` when exhaustive search cannot reach the goal;
- `search_limit` when the expansion bound stops search.

Invalid inputs and callback violations raise exceptions; an infeasible valid
query is data, not an exception. A failed path has shape `[0,J]`, cost
`+infinity`, and remains in its batch position.

Per-batch metrics report requested/accepted samples, configuration validity
queries, candidate edge checks, valid edges, expanded nodes, and final path
cost. Counts include shortcut checks. Metrics describe deterministic oracle
work, not backend performance.

## Canonical replay and comparison

Replay is canonical UTF-8 JSON with
`format="curobo-metal-graph-planning-case"`, `version=1`, sorted keys, compact
separators, finite input numbers, and one trailing newline. Version 1 replay
uses a declarative union of closed forbidden joint-space boxes. Failed expected
cost is JSON `null`, which reconstructs the normative infinite result.

Exact replay status, success, deterministic roadmap, metrics, and cost are
oracle evidence. Backend conformance does not require the same sampled path:
independently verify endpoints, limits, swept validity at the declared step,
success/failure category, and reported cost. Float64 reference cost uses
absolute tolerance `1e-12`.

# Production batched IK implementation

Wave 4A adds a portable PyTorch solver in `curobo_metal.ops.ik` and cost
composition in `curobo_metal.ops.costs`. It implements the stable semantics in
`contracts/ik.md`; it does not copy the reference solver's finite differences
or claim cuRobo CUDA LBFGS parity.

## Tensor and batching model

All input tensors, compiled kinematic metadata, and collision metadata must
already share a device and dtype. CPU accepts float32 and float64. MPS accepts
float32 only. No operator calls `.cpu()`, `.numpy()`, or changes device; an MPS
run with `PYTORCH_ENABLE_MPS_FALLBACK=0` exercises the complete fixed suite.

Targets accept `[3]`/`[4]` or `[G,3]`/`[G,4]`. Seeds accept `[J]` or `[N,J]`.
The solver evaluates the Cartesian product and returns `[G,N,J]`; a single goal
omits only the goal dimension, retaining the seed dimension and all failures.
Per-goal selection chooses the successful seed with minimum objective and uses
the first exact tie.

## Costs and gradients

Pose error is world-frame translation followed by the shortest target-frame
rotation vector. Pose, squared-hinge joint-limit, first/second-difference
smoothness, and squared-hinge collision costs are ordinary PyTorch expressions
and support autograd. Robot collision composition is production FK, production
sphere transform, and production sphere/sphere or sphere/cuboid distance.

The solver uses deterministic projected Adam with one optimizer state per
goal/seed pair. Projection applies inclusive joint limits after every step;
optional revolute wrapping precedes projection. Successful candidates are
frozen while remaining candidates continue. `differentiable=True` retains the
unrolled update graph for compact learning uses; the default releases optimizer
history between iterations.

## Outcomes

Success requires position, rotation, and collision tolerances simultaneously.
Final observable collision and limit constraints take precedence over generic
optimizer outcomes. Remaining failures are `infeasible_or_stationary` or
`max_iterations`. Results include solutions, booleans, stable statuses,
iteration counts, residuals, objective, collision cost/freedom, and selected
seed. Invalid metadata is an exception; an unsolved target is data.

## Verification and performance

`tests/ops/ik/test_solver.py` checks NumPy-oracle cost values, PyTorch gradients,
all four replay categories, goal/seed batching, empty and unbatched inputs,
deterministic replay, CPU, and fallback-disabled MPS. The standalone fixed-suite
runner is `benchmarks/ik/benchmark_ik.py`.

Benchmark JSON reports metadata construction, first/warmup solve, and
synchronized steady-state solve latency separately. It also records residuals,
success, status, collision cost, and collision freedom. The checked-in CPU and
MPS samples are under `benchmarks/ik/results/`; these portable eager results are
a correctness baseline, not a claim that eager MPS beats CPU for small serial
chains.

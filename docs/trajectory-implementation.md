# Production trajectory optimization

The production trajectory path is a portable, differentiable PyTorch
implementation for CPU float32/float64 and MPS float32. It composes the
production FK and primitive collision operations, including their fused Metal
paths when those operators select them. No trajectory-specific kernel was
added: profiling does not yet show that optimizer bookkeeping, rather than FK
or collision, is the limiting operation.

`TrajectoryProblem` accepts one problem (`[J]` endpoints and `[N,T,J]` seeds)
or a problem batch (`[B,J]` endpoints and `[B,N,T,J]` seeds). A shared
`CollisionModel`, or one model per problem, supports seed and environment
batching within the existing primitive collision API. Every seed stays
independent and exact objective ties select the first successful seed.

Costs follow the reference contract: hard endpoints plus differentiable
endpoint and inclusive-limit penalties, and physical-time velocity,
acceleration, and jerk integrals. Collision costs include knots and linearly
interpolated swept samples. Projected Adam keeps all knots in bounds and resets
the endpoints after each step. Observable endpoint and collision failures have
deterministic precedence over optimizer termination.

`generate_motion` solves a joint goal directly or consumes successful IK output
as a joint goal. `retime_trajectory`, `interpolate_trajectory`, and
`trajectory_metrics` report duration, path length, derivative maxima,
clearance, and limit violation. Sampled clearance is not a continuous
collision certificate.

Run:

```bash
uv run pytest -q tests/ops/trajectory
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run pytest -q tests/ops/trajectory
uv run python examples/motion_generation.py --device cpu
uv run python benchmarks/trajectory/benchmark_trajectory.py --device cpu
```

Benchmark warmup/compilation is measured separately from synchronized repeated
latency. Reports always include success, status, endpoint residual, sampled
clearance, and maximum limit violation.

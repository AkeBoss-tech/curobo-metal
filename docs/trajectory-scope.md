# Wave 4B trajectory reference scope

Wave 4B adds an independent motion-generation oracle without changing the
project's production-backend commitment. It answers a correctness question:
given a compact serial robot, endpoint request, limits, primitive world, and
fixed time grid, can candidate backends be judged against deterministic,
portable evidence?

The deliverable is deliberately a NumPy reference:

- `[T,J]` uniform-time trajectories and deterministic `[N,T,J]` seed batches;
- minimum-jerk initialization and projected finite-difference optimization;
- endpoint, joint-limit, velocity, acceleration, and jerk objectives;
- collision checks at knots and linearly interpolated swept samples using the
  existing sphere references;
- canonical cases and a machine-readable correctness summary.

The reference treats the start and goal as hard projected knots. A goal outside
the inclusive limits is reported as `endpoint_infeasible`; sampled collision
failure is `collision_constrained`. Success means endpoint tolerance and signed
clearance tolerance pass together. Smoothness affects ranking, not feasibility.
Because interpolated sampling is finite, success does not claim continuous
collision freedom between samples.

This wave does not add a PyTorch/MPS solver, a cuRobo compatibility facade,
Metal kernels, graph planning, trajectory timing optimization, dynamics,
meshes, voxels, or Isaac integration. The oracle is intentionally unsuitable
for latency claims: collision gradients nest central differences, and projected
descent prioritizes transparent deterministic behavior over convergence speed.

Regenerate evidence from a clean checkout with:

```bash
PYTHONPATH=src python tests/fixtures/trajectory/generate.py
PYTHONPATH=src pytest -q tests/contracts/test_trajectory.py
```

The first command rewrites the five canonical files under
`tests/fixtures/trajectory/` and
`artifacts/correctness/trajectory_reference.json`. Byte equality is tested.
The normative details and tolerances are in
`contracts/trajectory_optimization.md`; FK, collision signs/ties, and pose/joint
conventions remain owned by their existing contracts.

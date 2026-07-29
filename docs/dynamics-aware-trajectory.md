# Dynamics-aware trajectory optimization

Wave 8B adds a production, differentiable B-spline trajectory optimizer that
composes the existing whole-body recursive Newton–Euler dynamics, joint-space
collision model, and deterministic multi-seed selection. It runs with ordinary
PyTorch operations on CPU and MPS; MPS evidence is collected only with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.

## Public surface

`bspline_matrices` constructs clamped-uniform basis matrices for position,
velocity, acceleration, and jerk. Derivatives are expressed in physical time.
`sample_bspline` applies those matrices to any tensor ending in `[C, J]`.

`DynamicsAwareProblem` accepts serial or branched `WholeBodyModel` metadata,
batched start/goal queries, batched control-point seeds, per-joint velocity,
acceleration, jerk and torque limits, duration bounds, and optional collision
composition through a matching serial `KinematicChain`. `optimize_dynamics_aware`
jointly optimizes interior control points and log-duration. Fixed-path bisection
is available through `retime_dynamics_aware`.

The result retains every seed and reports `SUCCESS`, `INVALID_ENDPOINT`,
`COLLISION`, `CONSTRAINT_VIOLATION`, or `NUMERICAL_FAILURE` in stable batch/seed
order. Selection is the lowest-cost feasible seed, with the lowest index winning
ties.

MotionGen opts in with `MotionGenConfig(dynamics_aware=True,
dynamics_model=model)`. `dynamics_aware_options` forwards fields such as
`control_points`, dynamic limits, duration bounds, and weights. The default
MotionGen path is unchanged. When graph fallback is needed, sampled graph paths
are deterministically resampled into B-spline control-point seeds.

## Costs and constraints

The rollout samples `(q, q̇, q̈, q⃛)` from control points, evaluates production
tree RNEA at every sample, and composes:

- integrated squared jerk and generalized effort;
- duration regularization;
- squared hinge penalties for position, velocity, acceleration, jerk, and
  torque-limit violations;
- the existing continuous-subdivision robot collision cost.

Endpoint control points are projected exactly to the requested start and goal.
All other control points are projected into joint bounds after every Adam step.
Duration is represented in log space and clamped to configured positive bounds.

## cuRoboV2 parity and intentional mechanical divergence

Pinned cuRoboV2 publicly describes dynamics-aware optimization using a B-spline
representation, smoothness and torque limits, parallel seeds, collision-aware
rollouts, and whole-body motion generation. This implementation preserves those
observable capabilities and tensor batching semantics.

The mechanics differ only where CUDA-specific implementation details are not
portable: cuRoboV2 uses fused CUDA rollout kernels and GPU-oriented solver
machinery, while this backend composes autograd-visible PyTorch operators and a
deterministic Adam update. It does not emulate CUDA graphs, warp-level kernels,
or kernel-specific line-search ordering. These differences do not introduce a
CPU fallback on MPS.

## Verification

`tests/ops/trajectory/test_dynamics_aware.py` covers basis derivatives, RNEA
gradients, feasible and torque-violating trajectories, collision composition,
batched deterministic seeds, retiming, and fallback-disabled MPS execution.

Run the evidence benchmark with:

```bash
python benchmarks/trajectory/dynamics_aware.py --device cpu \
  --output artifacts/correctness/dynamics_aware_trajectory.json
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/trajectory/dynamics_aware.py \
  --device mps --output benchmarks/trajectory/results/dynamics-aware-mps.json
```

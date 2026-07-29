# Whole-body production implementation

Wave 6A adds differentiable fixed-base tree kinematics and rigid-body dynamics
in `curobo_metal.ops.whole_body`. The production API accepts the validated
`TreeRobot` reference model directly or compiles it once into a
device-resident `WholeBodyModel`. Inputs and outputs remain on the selected
CPU or MPS device; there are no implicit device or dtype transfers.

## Supported model and API

The implementation follows `contracts/whole_body_dynamics.md`: topologically
ordered parent-index trees, fixed/revolute/prismatic joints, multiple end
effectors, direct active-joint mimic mappings, fixed bases, local COM inertia,
and world gravity. `WholeBodyKinematicsResult` contains all-link transforms,
exact transform Jacobians, geometric Jacobians, and convenient end-effector
views. `InverseDynamicsResult`, `DynamicsCostConfig`, and
`DynamicsCostResult` make batching and cost composition explicit.

`inverse_dynamics` is a batched world-frame recursive Newton-Euler algorithm.
Reverse tree accumulation projects each joint wrench into active coordinates;
mimic contributions use the configured multiplier. `mass_matrix`,
`bias_torque`, and `gravity_torque` are defined from RNEA exactly as in the
contract. The mass matrix is symmetrized only to remove numerical roundoff.
Quadratic effort and effort-limit hinge costs use ordinary PyTorch operations,
including the specified zero subgradient at the limit.

## Differentiation and backend policy

The implementation is composed eager PyTorch. It intentionally uses functional
tensor updates in the reverse recursion so gradients propagate through
positions, velocities, accelerations, and costs. CPU supports float32 and
float64. MPS supports float32 only and uses the same composed operator path.
Runs on MPS are required to set `PYTORCH_ENABLE_MPS_FALLBACK=0` when collecting
evidence. No custom Metal kernel was added: the workload is a sequence of
small, dependency-heavy tree operations, and this wave had no profile showing
a stable fusible hot path.

## Verification and evidence

`tests/ops/whole_body` covers both checked-in NumPy fixtures, sibling Jacobian
zeros, multiple end effectors, negative-ratio mimic projection, RNEA
decomposition, symmetric positive-definite mass matrices, CPU float64 replay,
CPU/MPS float32 tolerances, non-contiguous and empty inputs, validation,
autograd VJPs, finite differences, and limit-boundary subgradients.

`benchmarks/whole_body/benchmark_whole_body.py` first replays the independent
reference artifact, then records synchronized full-tree FK, inverse-dynamics
forward/backward, and mass-matrix timings. CPU correctness and benchmark
evidence is checked in at `artifacts/correctness/whole_body_ops.json`; an MPS
run can be recorded separately with fallback disabled.

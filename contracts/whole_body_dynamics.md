# Whole-body tree kinematics and dynamics contract

Status: Wave 5D independent CPU correctness contract, version 1.

## Model and frames

The model is a fixed-base rooted tree in topological link order. Link zero is
the unique root and has parent `-1`; every later parent index is smaller than
its child index. A link owns the joint connecting it to its parent, its
parent-to-joint fixed origin, and its inertial data. Fixed, revolute, and
prismatic joints are supported. Multiple named end effectors are indices into
the same full-link result and may lie on different branches.

Coordinates are right-handed SI and homogeneous matrices act on column vectors.
As in the serial FK contract,
`T_parent_link = T_origin(xyz,rpy) @ T_motion(axis,q_link)`, fixed-axis RPY is
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`, and the normalized axis is expressed after
`T_origin`. Geometric twists are world-frame `[linear_xyz, angular_xyz]` at the
link origin.

The root is stationary. Spatial velocity recursions use world coordinates and
moments are taken about each link origin. Inertia `[ixx,iyy,izz,ixy,ixz,iyz]`
is the symmetric rotational inertia about the local center of mass, expressed
in the link frame. `com` is the local link-origin-to-COM vector. Gravity is a
world acceleration (default `[0,0,-9.81]`); inverse dynamics computes the force
required to produce acceleration against gravity, so a stationary mass has
force `-mass * gravity`.

## Active and mimic joints

Input columns contain only independently actuated movable joints, in link
appearance order. A mimic joint names one active source and evaluates
`q_m = multiplier*q_source + offset`,
`qd_m = multiplier*qd_source`, and
`qdd_m = multiplier*qdd_source`. Its Jacobian column and generalized effort
are multiplied by `multiplier`; offset contributes only to position. Mimic
chains and mimic references to other mimic joints are rejected.

This matches the pinned cuRobo V2 distinction between active joint state and
its generated mimic state. The reference keeps standard URDF multiplier and
offset semantics explicitly because pinned V2 stores mimic metadata and joint
offsets but does not expose rigid-body dynamics.

## Kinematics

`tree_forward_kinematics(robot,q)` accepts float32/float64 `[J]` or `[B,J]`,
including strided and empty batches, and returns float64:

| value | shape |
|---|---:|
| `transforms` | `[B,L,4,4]` |
| `transform_jacobian` | `[B,L,4,4,J]` |
| `geometric_jacobian` | `[B,L,6,J]` |

The exact transform derivative and VJP rules, invalid-value policy, and
float32/float64 tolerances are inherited from `forward_kinematics.md`.
Non-ancestors have exactly zero Jacobian columns. End-effector selection does
not remove intermediate or sibling links.

## Dynamics

`inverse_dynamics(robot,q,qd,qdd)` implements deterministic recursive
Newton-Euler dynamics for the full tree and returns torque/force `[B,J]`.
Children are accumulated in reverse link order. It excludes friction, damping,
rotor/gear inertia, flexible bodies, floating bases, contacts, external
wrenches, and actuator models.

The following identities define related operators:

```text
tau(q,qd,qdd) = M(q) qdd + b(q,qd)
b(q,qd)       = inverse_dynamics(q,qd,0; configured gravity)
g(q)          = inverse_dynamics(q,0,0; configured gravity)
M[:,j]        = inverse_dynamics(q,0,e_j; zero gravity)
```

`M` is symmetrized only to remove roundoff and must be symmetric positive
definite for a non-degenerate actuated fixture. The effort cost is
`w_e*sum(tau**2)`. The torque-limit cost is
`w_l*sum(max(abs(tau)-limit,0)**2)`. At `abs(tau)==limit` the selected
subgradient is zero.

## Differentiation and acceptance

The NumPy oracle exposes analytical transform Jacobians and deterministic
central-finite-difference torque Jacobians with respect to `q`, `qd`, and
`qdd`. Analytical dynamics derivatives are a future production-backend
responsibility. Reference differences use step `1e-6`, `atol=2e-7`,
`rtol=2e-6` for torque and
potential-energy gradients; cost gradients away from zero/limit kinks use
`atol=5e-7`, `rtol=5e-6`. Float64 replay tolerances are `2e-11` for transforms
and Jacobians, `2e-10` for torque/gravity/bias, and `5e-10` for mass matrices
and identity checks. A future float32 backend uses `2e-4`, `5e-4`, and `8e-4`
respectively.

Inputs must be finite real floating arrays of identical batch shape. Metadata
must have valid topology, normalized nonzero movable axes, nonnegative mass,
positive-semidefinite COM inertia, and positive effort limits. Outputs own
their storage.

## Replay and required coverage

Cases use canonical JSON format `curobo-metal-whole-body-case`, version 1.
`tests/fixtures/whole_body` contains a branched revolute/prismatic/fixed robot
with three end effectors and a negative-ratio mimic joint, plus a seven-DOF
Panda-class inertial model. `artifacts/correctness/whole_body_reference.json`
records source fixture hashes and float64 outputs. Replay never needs CUDA,
Isaac, Torch, or cuRobo runtime code.

Required checks cover batch/unbatched/strided/empty input, sibling Jacobian
zeros, mimic affine transforms and effort projection, finite differences,
RNEA decomposition, mass symmetry/positive definiteness, gravity as potential
gradient, effort limits/costs, deterministic regeneration, and invalid models.

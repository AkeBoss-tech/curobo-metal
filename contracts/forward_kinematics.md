# Forward kinematics contract

Status: sprint-1 correctness contract, version 1.

## Scope and conventions

The operator evaluates a rooted **serial** chain. Each emitted link corresponds
to one ordered joint record; fixed terminal joints are allowed and emit links
but consume no configuration coordinate. Branched trees, mimic joints, joint
limits, dynamics, and collision geometry are outside this contract.

- Coordinates use a right-handed Cartesian frame and SI units (metres,
  radians).
- Matrices use column vectors. `T_world_link @ [p_link, 1]` maps a point from
  link coordinates into world coordinates.
- A local joint transform is
  `T_parent_link(q) = T_origin(xyz, rpy) @ T_motion(axis, q)`.
- `rpy` follows the URDF fixed-axis convention:
  `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
- Revolute motion uses the right-hand rule. Prismatic motion translates along
  the normalized axis. An axis is expressed in the joint frame after
  `T_origin`.
- Joint coordinates are ordered by movable-joint appearance. Fixed joints have
  no coordinate.
- The geometric Jacobian is `[linear_xyz, angular_xyz]`. It describes the
  world-frame velocity of the link origin. Columns for joints downstream of a
  link are exactly zero.

## Inputs and outputs

`q` is floating-point and shaped `[J]` or `[B, J]`, where `J` is the number of
movable joints. The unbatched form is logically promoted to `B=1`; outputs
always retain the batch dimension. Strided/non-contiguous inputs are valid.
Empty batches `[0, J]` are valid. NaN, infinity, complex, integer, Boolean, and
wrong-rank/width inputs are errors. Angles are not wrapped or clamped.

For `L` emitted links:

| Value | Shape | Meaning |
|---|---:|---|
| `transforms` | `[B, L, 4, 4]` | `T_world_link` homogeneous transforms |
| `transform_jacobian` | `[B, L, 4, 4, J]` | exact `dT_world_link / dq` |
| `geometric_jacobian` | `[B, L, 6, J]` | link-origin spatial velocity Jacobian |

The independent CPU oracle accepts float32 or float64 input and evaluates and
returns float64. Device implementations should preserve their public input
dtype and are compared with the dtype-specific tolerances below. Outputs must
not alias mutable input storage.

## Differentiation

`transform_jacobian[..., j]` is the derivative of every homogeneous-transform
entry with respect to `q[..., j]`; derivatives of the constant bottom row are
zero. A backend autograd implementation returning `transforms` must therefore
obey the vector-Jacobian product

`grad_q[b,j] = sum(l,r,c, grad_T[b,l,r,c] * transform_jacobian[b,l,r,c,j])`.

The geometric Jacobian is checked separately and is not the derivative of a
particular pose-coordinate parameterization. At singular configurations it may
lose rank, but must remain finite and agree with transform derivatives. No
gradient is defined for robot metadata.

## Numerical acceptance

Comparisons are elementwise against checked-in float64 CPU golden data.
Absolute error is dominant near zero; both conditions use
`abs(actual - expected) <= atol + rtol * abs(expected)`.

| Quantity | float64 `atol`, `rtol` | float32 `atol`, `rtol` |
|---|---:|---:|
| transform entries | `2e-12`, `2e-12` | `2e-5`, `2e-5` |
| transform/geometric Jacobian | `5e-11`, `5e-11` | `8e-5`, `8e-5` |
| autograd VJP | `1e-10`, `1e-10` | `2e-4`, `2e-4` |

Central finite differences are only a secondary derivative check: step `1e-6`
for float64 with `atol=2e-9, rtol=2e-7`, and step `3e-3` for float32 with
`atol=3e-4, rtol=3e-3`. Exact agreement is not required across devices because
trigonometric implementations and contraction order differ.

## Replay format

Fixtures are UTF-8 JSON with `format="curobo-metal-fk-case"` and `version=1`.
They contain complete robot metadata, a two-dimensional `inputs.q`, and
float64 `expected` arrays plus link names. JSON numbers are finite; NaN and
infinity are forbidden. Canonical serialization sorts object keys and uses
compact separators, making cases byte-stable and easy to checksum.

Every backend must load the same file rather than regenerate random input.
Checked-in expected arrays come only from
`src/curobo_metal/reference/forward_kinematics.py`, which imports no backend
implementation. The planar fixture additionally has hand-derived exact
assertions, and both fixtures validate analytical derivatives with central
differences.

## Required edge coverage

Contract suites must include batch size one and greater than one, unbatched
input, fixed terminal links, zero angles, mixed-sign angles, large unwrapped
angles, non-contiguous input, and invalid values/shapes. A prismatic-joint case
must be included before a backend claims prismatic support. Empty batches are
part of the interface contract and must return correctly shaped empty outputs.

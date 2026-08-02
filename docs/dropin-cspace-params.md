# Portable C-space parameters

`curobo._src.robot.types.CSpaceParams` implements the pinned cuRoboV2 record
used by kinematics, graph planning, IK, and trajectory optimization.  Its
public field order and methods (`clone`, `copy_`, `inplace_reindex`,
`scale_joint_limits`, and `load_from_joint_limits`) match the pinned source.

## Portable guarantees

All materialized planning values are rank-one, per-DOF tensors on `device_cfg`.
Scalars for acceleration, jerk, and scale limits expand to the robot DOF.  A
batched or horizon-shaped tensor is rejected: those dimensions belong to solver
inputs such as `JointState`, not immutable robot configuration.  Values are
finite; acceleration and jerk maxima are positive, while scale, weight, and
safety-margin values are non-negative.

`inplace_reindex` permits a unique subset of the configured joints, which is
needed when reducing a kinematic tree.  It validates all requested names before
mutating state.  `copy_` retains existing compatible tensor buffers, matching
the caller-owned-buffer lifecycle used by warm planner configurations.
`scale_joint_limits` returns a clone, never mutates its input, and applies
per-DOF velocity/acceleration/jerk scales and an optional position safety
margin.  A mismatched joint order, device configuration, or safety margin that
collapses a position range fails explicitly.

## Explicit boundary

This is ordinary PyTorch CPU/MPS tensor state.  It does not expose CUDA struct
ABI buffers, CUDA graph capture, or raw Warp kernel state from the NVIDIA-only
backend.

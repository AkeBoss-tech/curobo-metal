# Drop-in kinematics namespace

The pinned cuRoboV2 public imports are available from `curobo.kinematics` and
their canonical implementation paths under `curobo._src.robot.kinematics`.
`KinematicsCfg.from_robot_yaml_file("franka.yml")` resolves the packaged asset
without requiring a caller-managed content path.

`Kinematics.compute_kinematics` accepts `JointState.position` with `[dof]`,
`[batch, dof]`, or `[batch, horizon, dof]` shape. Results preserve explicit
batch and horizon axes:

- tool position and wxyz quaternion: `[batch, horizon, tools, 3/4]`
- optional geometric Jacobian: `[batch, horizon, tools, 6, dof]`
- robot spheres: `[batch, horizon, spheres, 4]`
- optional center of mass: `[batch, horizon, 4]`

The facade compiles YAML/URDF data into the production differentiable
whole-body FK backend. CPU supports float32 and float64. MPS uses float32 and
the same backend without CPU fallback. Positions, quaternions, Jacobians,
spheres, and center of mass remain connected to PyTorch autograd.

For the packaged Franka model, configured finger locks reduce the nine URDF
joints to seven active joints, `panda_hand` is the sole tool frame, and 61
collision spheres are emitted in YAML declaration order.

`KinematicsState` uses the canonical `ToolPose` record: link lookups,
`to_dict()`, link reordering, and goal conversion all keep the FK
`[batch, horizon, links, ...]` layout. State `clone()`, `detach()`,
`contiguous()`, and `to(DeviceCfg(...))` cover tool poses, Jacobians, spheres,
CoM, and collision-geometry indices on both CPU and MPS. Integer state
indexing deliberately matches pinned cuRobo: `ToolPose` retains a singleton
batch axis while the raw tensors use normal PyTorch integer-index views.

Deliberate gaps are explicit: USD/Isaac parsing remains unsupported by the
portable config loader; nonzero locked revolute joints currently raise
`NotImplementedError`; mesh-returning helpers and CUDA multi-environment sphere
configuration mutation are not implemented. Only sphere environment index zero
is accepted, and mismatched joint names, ranks, degrees of freedom, devices, or
dtypes raise rather than silently converting.

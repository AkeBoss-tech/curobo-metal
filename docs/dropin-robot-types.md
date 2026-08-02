# Robot Link and Joint Type Compatibility

`curobo._src.robot.types.joint_types.JointType` preserves every pinned V2
integer encoding (`FIXED=-1` through `Z_ROT_NEG=11`). Portable URDF parsing
uses the same axis-aligned convention for fixed, prismatic, and revolute
joints. Arbitrary-axis joints remain an explicit unsupported boundary rather
than being approximated by a different axis.

`curobo._src.robot.types.link_params.LinkParams` is the portable static link
metadata record. `LinkParams.create` accepts pinned YAML pose transforms in
`[x, y, z, qw, qx, qy, qz]` order and materializes the expected NumPy `[3, 4]`
affine transform. It also accepts that native `[3, 4]` representation for
loader round trips. The input dictionary is not mutated.

The module retains the upstream `Pose`, `DeviceCfg`, and `log_and_raise`
imports as public re-exports. Pose conversion works with the CPU and Apple
Metal `DeviceCfg` paths; `LinkParams` itself stays host-side NumPy metadata,
as it is compiled into tensors by the kinematics loader rather than executed
as a CUDA/Warp kernel.

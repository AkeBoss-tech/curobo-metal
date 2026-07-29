# Robot and configuration compatibility

Wave 7A adds a portable configuration and value-model seam for the pinned
cuRoboV2 revision `8e734f3ced1df898990bcd92de40abce475907db`. It does not
import cuRobo, CUDA, Warp, Isaac, or simulator modules.

## Public models

`curobo_metal.types` provides `DeviceCfg` (`TensorDeviceType` is an alias),
`JointState`, `Pose`, planning result records, `MotionGenResult`, and
`MotionGenStatus`. `JointState` carries position and optional velocity,
acceleration, jerk, time, knot, and joint-name metadata. Its tensor operations
preserve all present fields. `Pose` uses cuRobo's `wxyz` quaternion order,
normalizes only when requested, and provides matrix/list conversion. Both types
support explicit device and dtype movement; MPS configurations reject float64.

These classes intentionally cover the stable data seam rather than every
deprecated upstream helper. They can be passed through an explicit adapter
where an application insists on the upstream class identity.

## Robot loading

`RobotCfg.create(path_or_mapping)` accepts:

- cuRobo-shaped YAML mappings under `robot_cfg.kinematics`;
- ordinary URDF files through `RobotCfg.from_basic` or `load_robot_config`;
- XRDF collision geometry referenced by `xrdf_path`;
- already parsed mappings, avoiding any YAML dependency.

PyYAML is used when installed. A built-in conservative parser covers the YAML
mapping/list/scalar subset used by robot and XRDF configurations, so YAML robot
loading remains available in a base installation.

URDF parsing preserves XML link and joint order in the public records, then
builds a parent-before-child order for production tree models. It preserves
fixed joints, independent joint names, mimic source/multiplier/offset, axes,
origins, position/velocity/effort limits, masses, centers of mass, and symmetric
inertia tensors. XRDF and inline YAML collision spheres preserve link order,
centers, radii, and negative disabled-sphere radii.

Paths are absolute as supplied or resolved relative to the YAML file and then
its optional `asset_root_path`. A serialized `RobotCfg` writes its resolved
URDF path, making round trips relocatable.

## Production adapters

- `to_tree_robot()` and `to_whole_body_model()` retain branches and mimic
  joints.
- `to_serial_robot()` and `to_kinematic_chain()` accept only an actual serial
  chain without mimic offsets. They raise `UnsupportedConfigError` instead of
  silently dropping structure.
- `to_collision_inputs()` returns link-local `[x,y,z,r]` values and int64 link
  indices in the production FK link order.
- `load_world_config()` accepts primitive cuboids and spheres.

## Exact unsupported boundary

USD/USDA/USDC and Isaac-specific fields raise `UnsupportedConfigError` with a
request to provide URDF. Planar, floating, and other non-fixed/non-revolute/
non-prismatic URDF joints raise the same error. Mesh, voxel, and ESDF worlds are
not converted by this layer. A branched or mimic robot cannot be converted to
the serial FK representation; use the tree/whole-body adapter. These errors are
raised at load or conversion time, never after silently removing data.

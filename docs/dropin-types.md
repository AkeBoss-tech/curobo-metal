# Foundational drop-in types

This slice targets NVlabs/cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`.

## Namespace contract

At that revision, the public API is the single module `curobo/types.py`, not a
`curobo.types` package. Its complete pinned export list is implemented:
`JointState`, `RobotState`, `Pose`, `ToolPose`, `GoalToolPose`,
`ToolPoseCriteria`, `CameraObservation`, `LidarObservation`, `ContentPath`,
and `DeviceCfg`.

The matching implementation paths are also available:
`curobo._src.types.device_cfg`, `.pose`, `.math`, `.base`, `.control_space`,
`.tensor`, `.robot`, `.camera`, `.lidar`, `.content_path`, and `.tool_pose`,
plus `curobo._src.state.state_joint`, `.state_robot`, and
`curobo._src.cost.tool_pose_criteria`.
`curobo._src.types.math` and `.base` are compatibility paths deprecated by the
pinned upstream source. No `curobo.types.math`, `.state`, `.base`, or `.robot`
submodule is invented.

## Device policy

Importing the slice has no CUDA, Warp, Isaac, or Omniverse dependency.
`DeviceCfg()` deliberately defaults to CPU in curobo-metal, the one documented
deviation from the pinned CUDA default. An explicitly requested CUDA or MPS
device is passed to PyTorch unchanged; an unavailable device raises instead of
falling back to CPU.

## Implemented behavior

`Pose` preserves the pinned constructor field order, wxyz convention, batch
shapes, list/NumPy/matrix/Euler construction, clone/index behavior, in-place
`to`, matrix and caller-owned output buffers, inverse, composition, distance,
point transformation, and repeat helpers. `JointState` preserves its field order, derivative tensor layout,
zero/from-position/from-NumPy/from-state-tensor constructors, clone, in-place
detach, conversion, indexing/assignment, shape helpers, reordering, and state
tensor concatenation, stack/cat, and seed repeat. `ControlSpace` preserves all
member names and values.

The observation values preserve pinned field order, clone/copy/device mutation,
shape errors, RGB-D ray projection, and LiDAR metadata behavior. Tool poses
enforce the pinned 4D/5D layouts and provide link extraction, ordering,
cloning, mutation, dictionary conversion, and current-to-goal conversion.
`RobotState` preserves joint/torque storage, indexing, cloning, link access,
and portable copy operations. `ContentPath` resolves relative packaged paths
and rejects ambiguous inputs. Criteria factories and tensors are CPU-default.

`RobotCfg` preserves the pinned constructor, `create`, `from_basic`, `cspace`,
and `write_config` entry points. Dictionary and URDF construction are backed by
the existing portable curobo-metal loader.

## Precise remaining gaps

All public `curobo.types` names are implemented. Remaining gaps are deprecated
or specialized internal operations: `JointState` blend, kernel application,
time scaling, augmentation/append, trajectory gather/copy/trim, DOF indexing,
and pinned finite-difference operator parity. Pose gradient scratch-buffer
arguments are accepted for signature compatibility, while portable Torch
autograd allocates its own intermediates. Camera projection uses portable
Torch rather than fused CUDA. `RobotState.copy_at_batch_seed_indices` copies
joints and torques but not backend-specific kinematics buffers.

`RobotCfg` does not reproduce upstream `KinematicsCfg` or `DynamicsCfg`
classes. `write_config` is supported only when the facade wraps a
curobo-metal robot configuration; arbitrary upstream-like kinematics objects
raise `NotImplementedError` instead of being serialized incompletely.
The root `curobo.__version__` reports the curobo-metal distribution version,
not an upstream cuRobo release identifier.

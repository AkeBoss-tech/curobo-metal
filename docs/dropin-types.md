# Foundational drop-in types

This slice targets NVlabs/cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`.

## Namespace contract

At that revision, the public API is the single module `curobo/types.py`, not a
`curobo.types` package. The implemented public imports are:

- `curobo.types.DeviceCfg`
- `curobo.types.JointState`
- `curobo.types.Pose`

The matching implementation paths are also available:
`curobo._src.types.device_cfg`, `.pose`, `.math`, `.base`, `.control_space`,
`.tensor`, and `.robot`, plus `curobo._src.state.state_joint`.
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
shapes, list/NumPy/matrix construction, clone/index behavior, in-place `to`,
matrix generation, inverse, composition, point transformation, and repeat
helpers. `JointState` preserves its field order, derivative tensor layout,
zero/from-position/from-NumPy/from-state-tensor constructors, clone, in-place
detach, conversion, indexing/assignment, shape helpers, reordering, and state
tensor concatenation. `ControlSpace` preserves all member names and values.

`RobotCfg` preserves the pinned constructor, `create`, `from_basic`, `cspace`,
and `write_config` entry points. Dictionary and URDF construction are backed by
the existing portable curobo-metal loader.

## Precise remaining gaps

The rest of pinned `curobo.types` is not implemented: `RobotState`,
`CameraObservation`, `LidarObservation`, `ContentPath`, `ToolPose`,
`GoalToolPose`, and `ToolPoseCriteria`. Internal pose buffer-output variants,
batched transform kernels, distance helpers, and copy-in-place are also not yet
provided. Most deprecated `JointState` trajectory-operation methods are absent,
including blend, stack/cat, seed repeat, augmentation, trajectory gather/trim,
and finite-difference operator parity.

`RobotCfg` does not reproduce upstream `KinematicsCfg` or `DynamicsCfg`
classes. `write_config` is supported only when the facade wraps a
curobo-metal robot configuration; arbitrary upstream-like kinematics objects
raise `NotImplementedError` instead of being serialized incompletely.
The root `curobo.__version__` reports the curobo-metal distribution version,
not an upstream cuRobo release identifier.

# Portable JointLimits

`curobo._src.robot.types.JointLimits` keeps the pinned V2 tensor layout:
position, velocity, acceleration, jerk, and optional effort bounds each have
shape `[2, dof]`, with lower values in row zero and upper values in row one.

The portable implementation materializes all inputs on its `DeviceCfg`,
preserves caller-owned buffers during `copy_`, and provides `clone`, `to`,
`reindex`, `inplace_reindex`, and conflict-safe `merge` lifecycle helpers.
Name-based selection is all-or-nothing: unknown or duplicate requested names
leave an existing record unchanged. `merge` refuses contradictory values for a
shared joint unless `overwrite=True` is explicit.

CPU tensors and float32 MPS tensors are supported. Limits may be infinite for
unbounded joints, but NaN or inverted intervals are rejected. The CUDA/Warp
packed joint-limit ABI is not emulated; portable planners consume these regular
PyTorch tensors directly.

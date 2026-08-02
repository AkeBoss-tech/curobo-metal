# Portable state and foundational types

The `curobo._src.state`, `curobo._src.transition`, and `curobo._src.types`
paths expose tensor-native portable values on CPU and MPS. `JointState`,
`RobotState`, `Pose`, tool poses, filters, `DeviceCfg`, and state-transition
models preserve PyTorch autograd and device placement.

`JointState` reexports the standard state-operation and trajectory-indexing
helpers from its pinned internal path. The pose module also exposes the common
matrix, quaternion, compose, inverse, and point-transform helpers using the
same `wxyz` convention as the value type.

`DeviceCfg` provides float, int8/int32/int64, and boolean conversions and
rejects MPS dtypes other than float32 before allocation. This avoids a hidden
CPU fallback; it does not make CUDA dtypes available on MPS.

CUDA JIT state kernels, CUDA streams/graphs, and packed-buffer ABI calls are
not implemented. The corresponding portable transition operations run as
ordinary differentiable PyTorch tensor expressions instead.

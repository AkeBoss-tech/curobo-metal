# Portable `Pose` lifecycle

`curobo._src.types.pose.Pose` is a CPU/MPS PyTorch value type using cuRobo's
`wxyz` quaternion convention. Its public construction, Euler and matrix
conversion, indexing, composition, inverse, distance, output-buffer, cloning,
and device-transfer APIs work without CUDA, Warp, or a CUDA graph.

The portable transform behavior deliberately preserves the V2 launch layouts:

- `transform_points` transforms a flattened `[N, 3]` point set with either one
  pose or one pose per point. It never produces a broadcast `[N, N, 3]`
  cross-product.
- `batch_transform_points` and its inverse transform `[B, ..., 3]` point
  clouds with a matching flattened pose batch and retain the cloud layout.
- A matching trajectory prefix is also supported: pose `[B, H, 3]` with points
  `[B, H, N, 3]` retains all four dimensions and is differentiable on CPU/MPS.
- All transform, inverse, Euler/matrix conversion, and composition operations
  use regular differentiable PyTorch operations on CPU and float32 MPS.

`clone`, `detach`, `contiguous`, `repeat_seeds`, and indexing retain the pose
name and an explicitly materialized rotation matrix. Mutable `copy_` and
`__setitem__` refresh that matrix cache when their source only supplies a
quaternion, preventing stale transforms after buffer reuse. `repeat`,
`repeat_seeds`, and `stack` preserve all dimensions after the leading batch
axis, including a trajectory/horizon dimension. Seed expansion uses the V2
flattened `batch × seed` order. `Pose.to` retains the pinned mutating contract
and moves/converts position, quaternion, and any cached rotation together
through `DeviceCfg`.

The module does not provide CUDA profiler ranges, custom CUDA adjoint buffers,
or Warp/CUDA transform kernels. The corresponding optional arguments remain
accepted for call compatibility; portable execution obtains its gradients from
PyTorch autograd instead.

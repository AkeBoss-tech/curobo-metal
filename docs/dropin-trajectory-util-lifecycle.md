# Portable trajectory utility lifecycle

`curobo._src.util.trajectory`, `trajectory_seed_generator`, and
`trajectory_execution_manager` run with normal PyTorch tensors on CPU and
Apple MPS. They retain differentiable positions, explicit batch/seed layouts,
deterministic endpoint handling, and state/action queue lifecycle methods.

## Spline-knot interpolation

The public `BSPLINE_KNOTS_CUDA` enum member is accepted for a `JointState`
whose `knot`, `knot_dt`, and `control_space` (`BSPLINE_3`, `BSPLINE_4`, or
`BSPLINE_5`) are populated. The portable implementation materialises a
clamped-uniform B-spline using ordinary PyTorch operations, including
per-batch output lengths and optional table-indexed start/implicit-goal
boundary states. It returns regular differentiable position, velocity,
acceleration, and jerk tensors on the caller's device.

This is a useful CPU/MPS compatibility implementation, but it is not claimed
to be byte-for-byte equivalent to cuRobo's CUDA spline kernel: the original
uses a packed CUDA launch layout and kernel-specific implicit-boundary rules.
Raw CUDA graph capture, packed buffers, and Warp execution remain outside this
portable module.

## Queue and seed behavior

`TrajectorySeedGenerator` produces constant, endpoint-interpolated, and
deceleration seeds while preserving autograd and device residency.
`TrajectoryExecutionManager` exposes offset command windows one state at a
time, supports non-consuming preview/reset/clear operations, and never reads
beyond a short state horizon. Its intentionally misspelled pinned method
`get_shifteaction_dim_buffer` remains available for warm starts.

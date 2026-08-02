# Motion retargeter compatibility

`curobo.motion_retargeter` provides the pinned V2 stateful retargeting
surface on PyTorch CPU and Apple MPS.  A retargeter uses global multi-seed IK
for its first frame, then uses velocity-aware warm-started local IK.  With
`use_mpc=True`, the first global solution initializes a portable
receding-horizon MPC instance and following frames emit the executed endpoint
stream as `RetargetResult.trajectory`.

The configuration preserves ordered tool criteria, environment batch capacity,
collision-sphere loading invariants, step/time tolerances, and reset behavior.
`solve_sequence` resets first and returns states arranged as
`[environment, frame, dof]`.

`MotionRetargeterCfg` is also the device-compilation boundary: criteria built
with the default CPU factory are copied to its declared CPU or MPS
`DeviceCfg`, without mutating a caller-owned criteria dictionary.  Its
`batch_shape` reports the fixed `[environment, tracked-tool]` capacity, and
`with_device(...)` recompiles the same declarative request for CPU or MPS.
Malformed capacities, empty optimizer lists, device descriptions, and an MPC
cold-start iteration count below its warm-start count fail while constructing
the config rather than during a solve.  CUDA device requests are rejected by
the shared portable device resolver; no CUDA graph, Warp, or Isaac runtime is
created by configuration compilation.

Both IK and MPC honor the fixed `num_envs` capacity.  In particular, a batched
MPC retargeter executes the requested number of endpoint steps independently
for every environment and returns an endpoint trajectory with shape
`[environment, steps_per_target, dof]`.  `solve_frame` deliberately accepts a
single target horizon; a time-major `SequenceGoalToolPose` belongs to
`solve_sequence`, which makes the state-reset boundary explicit.

The portable MPC retargeting adapter currently supports one tracked tool link;
multi-link clips work through warm-started IK.  Its Cartesian target is
resolved by the production portable IK solver before trajectory optimization.
That is functional CPU/MPS planning, not a claim of identical CUDA rollout
control, raw CUDA graph capture, Warp kernels, or Isaac humanoid integration.

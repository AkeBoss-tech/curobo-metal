# Portable retargeting result lifecycle

`curobo._src.motion.motion_retargeter_result.RetargetResult` retains the
pinned V2 layout: `joint_state` plus optional `trajectory`.  A frame result
uses `[environment, dof]`; sequence results use `[environment, frame, dof]`.
MPC trajectories are always `[environment, intermediate-frame, dof]`, while
IK retargeting returns `trajectory=None`.

The Metal implementation additionally makes the materialised CPU/MPS result
safe to consume beyond the solver call:

- `batch_size`, `num_dof`, `num_frames`, `num_trajectory_frames`,
  `is_sequence`, and `is_mpc` describe the stable public tensor layout.
- `clone()`, `detach()`, and `to(DeviceCfg | device)` retain every joint
  derivative/timing channel, preserve the autograd graph until detached, and
  never use a CUDA result buffer.
- `select_batch(...)` (and `result[...]`) preserves a leading environment
  axis even for integer selection, so a selected result can be passed to a
  batch-oriented CPU or MPS solver directly.

Construction validates matching batch size, DOF, device, dtype, and declared
joint order between final state and MPC endpoint stream.  CUDA graph handles,
Warp-owned trajectory buffers, and NVIDIA-specific result aliases are not
represented or emulated.

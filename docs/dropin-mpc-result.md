# MPC solver result compatibility

`curobo._src.solver.solver_mpc_result.MPCSolverResult` preserves the pinned
V2 dataclass fields and clone behavior. The portable implementation adds
materialized-result helpers for CPU and Apple MPS:

- `clone()` deep-copies tensors, every `JointState` derivative channel,
  `RobotState` torque data, and nested diagnostic mappings.
- `to(DeviceCfg(...))` moves floating result data while retaining bool and
  integer dtypes; `select_batch()`, indexing, and `successful()` retain the
  leading batch axis.
- `action_at()` returns an independent command value from the current action
  buffer or sequence without advancing solver state, and `validate_action_layout()`
  checks the public `[batch, horizon, dof]` command convention.

This is a portable value-object lifecycle. CUDA graph-resident command/result
buffers, packed CUDA result ABI, cursor advancement, and moving an opaque CUDA
robot-model state are explicitly unsupported rather than simulated.

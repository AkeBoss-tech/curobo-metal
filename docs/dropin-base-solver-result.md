# Base solver result compatibility

`curobo._src.solver.solver_base_result.BaseSolverResult` is the portable
materialised value layer shared by IK, TrajOpt, and MPC.  It preserves V2's
`[batch, seed, ...]` result layout and has deterministic in-place merge
operations for successful candidates or whole batch entries.  Merges include
solution tensors, all available `JointState` channels, robot-state values, and
the public rollout-metric copy protocol.

`clone()` deep-copies tensor-bearing nested diagnostics and metric/state
objects. `to(DeviceCfg)` moves regular PyTorch result values to CPU or MPS and
changes floating tensors to the requested dtype while retaining bool and
integer tensor dtypes. `refresh_batch_metadata()` derives V2's optional
`batch_size` and `num_seeds` fields from materialised output; `validate()`
checks result layouts before a merge.

The class deliberately does not emulate CUDA graph handles, packed result
buffers, or `RobotState` instances carrying a CUDA model-state buffer. Moving
the latter raises `NotImplementedError` rather than returning a mixed-device
object.

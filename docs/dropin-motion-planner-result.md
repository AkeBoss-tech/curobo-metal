# Motion-planner result compatibility

`curobo._src.motion.motion_planner_result` preserves the pinned V2 dataclass
layout for `MotionPlannerResult` and `GraspPlanResult`.  Results remain plain
PyTorch and `JointState` values on CPU and Apple MPS; they do not retain a
CUDA graph handle, raw result-buffer pointer, or packed CUDA ABI object.

The portable lifecycle adds operations that are useful after a batched plan:

- `validate()` checks the materialised success, stage, trajectory, timing,
  goal-index, device, and planning-time invariants without rejecting a
  deliberately partial failure result.
- `success_per_problem`, `num_success`, `num_failures`, `failure_mask`,
  `success_ratio`, `any_success()`, and `all_success()` collapse the normal
  `[problem, seed]` success layout deterministically.
- `clone()`, `detach()`, and `to(DeviceCfg | device)` retain all declared and
  dynamically attached materialised payloads.  Floating data adopts the
  requested dtype while bool success flags and integer goal indices retain
  their dtypes.
- `select_batch(...)` retains a leading batch axis, including for integer
  selection, and `successful()` filters to all successful problem rows.  The
  existing `result[...]` operator intentionally follows regular tensor
  indexing and may rank-reduce an integer selection.

`GraspPlanResult.stage_success()` and `stage_trajectory()` continue to expose
approach, grasp, and lift values directly.  CUDA graph-owned result buffers,
private packed memory aliases, and NVIDIA numerical/ABI equivalence are
explicitly outside this portable value-object contract.

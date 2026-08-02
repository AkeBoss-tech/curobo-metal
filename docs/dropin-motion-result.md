# Portable motion-planning result lifecycle

`curobo._src.motion.motion_planner_result` retains the pinned V2 dataclass
fields for `MotionPlannerResult` and `GraspPlanResult`.  The Metal port also
provides portable result lifecycle helpers that are useful in real batched
planning applications:

- `batch_size`, `success_per_problem`, `num_success`, `success_ratio`,
  `any_success()`, and `all_success()` interpret the conventional
  `[problem, seed]` success layout.
- `clone()`, `detach()`, `to(DeviceCfg)`, and batch indexing copy tensor,
  `JointState`, nested dictionary, and dynamically-attached stage-result
  payloads without a CUDA result-buffer ABI.
- `GraspPlanResult.stage_success()` and `stage_trajectory()` provide stable
  access to the approach, grasp, and lift outputs.

Raw CUDA result buffers, CUDA graph ownership, and upstream private packed
memory aliases are intentionally not emulated.  Tensor/trajectory outputs
remain ordinary CPU or MPS PyTorch objects and preserve autograd until
`detach()` is requested.

# Graph planner lifecycle compatibility

`curobo._src.graph_planner.PRMGraphPlanner` runs the production deterministic
PyTorch roadmap planner on CPU and MPS.  It retains V2's persistent roadmap,
sampling seed, endpoint validation, interpolation, reset, warmup, and result
semantics.  A supplied `SceneCollisionCfg` is now owned by the planner when no
checker is supplied, and a compiled `RobotRolloutCfg` contributes its portable
constraints to sample feasibility.

`GraphPlannerResult.valid_query` means the start/end inputs were valid.  It is
therefore still true for a valid query that has no connected roadmap path;
per-problem success belongs in `success`.  Results provide clone/detach/device
movement, padded variable-length paths, batch selection, and device-resident
success/failure indices.

CUDA graph capture, Warp graph buffers/steering, and analytic CCD are not
emulated. `use_cuda_graph_for_rollout` remains accepted configuration and runs
ordinary persistent eager CPU/MPS state; `reset_cuda_graph()` raises a precise
unsupported-feature error.

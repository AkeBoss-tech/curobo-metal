# Rollout protocol compatibility

`curobo._src.rollout.rollout_protocol.Rollout` is the pinned V2 structural
typing boundary used by the portable solver and optimizer stack.  It exports
the canonical `RolloutResult`, `RolloutMetrics`, `CostsAndConstraints`, and
`CostCollection` value types from `curobo._src.rollout.metrics`.

Portable CPU and MPS rollout implementations provide the full action-bounds,
action/state metrics, parameter update, batch resizing, timestep update, and
reset/seed lifecycle.  The normal eager PyTorch implementations preserve
device residency and autograd; `use_cuda_graph` is represented by the existing
shape-stable executor lifecycle where supported, rather than a raw CUDA Graph.

`RobotRolloutCfg.create_with_component_types` compiles mapping-shaped
transition and cost-manager configurations without mutating the caller's
mapping.  It validates declared component types before execution and retains
an explicit `object` sentinel for solver assembly that already owns compiled
portable records.

Raw CUDA Graph capture, CUDA streams, and Warp packed-buffer ABI calls are not
part of this portable protocol and remain explicit unsupported low-level
boundaries.

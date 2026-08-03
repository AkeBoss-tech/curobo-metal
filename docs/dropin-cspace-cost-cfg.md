# Portable C-space cost configuration lifecycle

`curobo._src.cost.cost_cspace_cfg.CSpaceCostCfg` and
`curobo._src.cost.cost_cspace_dist_cfg.CSpaceDistCostCfg` provide the pinned
V2 terminal/running C-space configuration flow on CPU and Apple Metal. They
materialize configuration tensors on `DeviceCfg.device`, validate DOF and
limit layouts before a solve, clone accepted limit records, and preserve
stable weight buffers across same-shape updates.

`BaseCSpaceCost` validates the allocated `[batch, horizon, dof]` state,
torque, goal, and goal-index layouts. Its mutable target-weight buffer starts
disabled, can be toggled without altering the reusable configuration, and is a
regular CPU/MPS tensor rather than a CUDA launch buffer.

When a state transition uses teleport mode, a state-bound configuration is
converted to position mode before bounds are validated. This deliberately
allows a valid position-only limit record for that mode. C-space distance
configurations select either `cspace_distance_weight` or `null_space_weight`
from a transition and zero running weights when `only_terminal_cost=True`.

The CUDA/Warp retiming kernels and their graph-captured workspaces remain
explicitly unavailable. Setting `retime_weights` or
`retime_regularization_weights` raises `UnsupportedCostFeature`; composed
PyTorch costs remain differentiable on the supported CPU/MPS paths.

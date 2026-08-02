# Portable c-space cost bridge

`curobo._src.cost.wp_cspace_position.PositionCSpaceFunction` and
`curobo._src.cost.wp_cspace_state.StateCSpaceFunction` now support their
pinned V2 tensor call signatures on CPU and Apple Metal.  They preserve
`[batch, horizon, dof]` output rank, indexed targets, activated joint bounds,
caller-owned output/gradient buffers, V2 physical-time regularization, and
first-order gradients for their dynamic input channels.

The position bridge additionally implements velocity-tightened position
limits for a positive current-state time offset.  The state bridge implements
position/velocity/acceleration/jerk/effort bounds, target terminal weighting,
energy regularization, and its two tensor retiming switches.

These are composed PyTorch implementations, not aliases for the CUDA/Warp
packed-buffer kernels.  The raw `forward_cspace_position_warp` and
`forward_cspace_state_warp` kernel names deliberately raise
`NotImplementedError`; code requiring Warp launch objects, CUDA graph capture,
or the NVIDIA memory ABI must use upstream cuRobo on CUDA.

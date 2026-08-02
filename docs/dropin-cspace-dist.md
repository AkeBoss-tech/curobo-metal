# Portable C-space distance cost

`curobo._src.cost.cost_cspace_dist.CSpaceDistCost` evaluates the V2 squared
joint-space target term with ordinary PyTorch on CPU and Apple Metal.  After
`setup_batch_tensors(batch, horizon)`, the normal rollout interface accepts
`current_vec` shaped `[batch, horizon, dof]`, a `[goals, dof]` table, and one
integer goal index per batch element.  It returns `[batch, horizon]`, applies
non-terminal and terminal per-DOF weights, retains reusable per-DOF cost and
gradient diagnostics, and is differentiable with respect to current and goal
tensors.

For backwards compatibility, an unallocated direct invocation retains the
old portable per-DOF output.  New planning code should allocate the rollout
shape first and consume the V2-shaped reduced output.

The raw `forward_l2_warp` kernel and CUDA graph/stream ABI are intentionally
not emulated.  This implementation is an eager CPU/MPS PyTorch equivalent,
not a Warp or CUDA binary compatibility layer.

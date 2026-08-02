# Portable self-collision cost

`curobo._src.cost.cost_self_collision.SelfCollisionCost` is a CPU/MPS
implementation of the pinned V2 cost lifecycle.  It accepts batched robot
spheres shaped `[batch, horizon, spheres, 4]`, evaluates configured pairs, and
returns the largest positive squared overlap:

`0.5 * weight * max(0, (r_i + p_i + r_j + p_j)^2 - ||x_i - x_j||^2)`.

Pair order provides deterministic tie resolution.  `store_pair_distance=True`
retains unweighted, signed squared-overlap diagnostics in `_pair_distance`;
`get_gradient_buffer()` exposes the selected-pair derivative workspace.
`setup_batch_tensors`, subset `reset`, enable/disable, and autograd work on
CPU and fallback-disabled float32 MPS.

The raw CUDA/Warp launch ABI, CUDA graph capture, and NVIDIA numerical-parity
claim remain unavailable.  The implementation uses the production portable
sphere-pair operator rather than emulating CUDA kernels.

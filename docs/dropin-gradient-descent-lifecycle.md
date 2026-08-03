# Portable gradient-descent lifecycle

`GradientDescentOpt` runs real eager PyTorch gradient descent on CPU and MPS.
It accepts cuRobo's flattened or `[problem, horizon, dof]` seed layouts,
projects published action bounds, tracks a finite best action per problem, and
handles constant/no-gradient objectives as zero-gradient updates.

`reinitialize(action, mask=...)` now retains a masked warm start for the next
batched solve; only selected problems are replaced. Resizing problems clears
stale optimizer state and notifies rollout instances through
`update_batch_size`. Configuration and runtime parameter updates validate
finite scales, convergence settings, and known field names atomically.

CUDA graph capture, CUDA timing, packed rollout buffers, and Warp/CUDA
gradient kernels remain explicit non-portable boundaries. Portable execution
uses ordinary autograd and optional MPS synchronization only.

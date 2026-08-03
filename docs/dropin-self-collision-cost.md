# Portable self-collision cost

`curobo._src.cost.cost_self_collision.SelfCollisionCost` evaluates the pinned
V2 maximum squared sphere-overlap objective on ordinary PyTorch CPU or Apple
Metal tensors.  `SelfCollisionCostCfg` is the matching factory record and its
`class_type` points to this concrete implementation.

The input is `[batch, horizon, spheres, 4]`, with `xyzw-radius` values.  The
configured pair order is deterministic: the first maximum wins.  The cost
keeps V2-style distance, gradient, sparse-index, optional pair-distance, and
block-reduction workspaces.  Repeating `setup_batch_tensors` with the same
shape retains these buffers; `reset(problem_ids)` clears every relevant
workspace for only those batch entries.  Returned costs remain differentiable
with respect to sphere centers and radii on CPU and fallback-disabled MPS.

The implementation uses the production CPU/MPS sphere-pair operation rather
than the CUDA `SelfCollisionDistance` launch ABI.  Raw CUDA graph capture,
Warp kernels, and their packed launch buffers are intentionally not exposed.

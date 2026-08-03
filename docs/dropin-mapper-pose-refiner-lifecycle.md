# Portable mapper pose-refiner lifecycle

`BlockSparseRaycastPoseRefiner` is a CPU/MPS dense-map adapter for the public
raycast-refiner surface. It uses the production dense mapper renderer and local
PyTorch pose update, not Warp block-sparse raycasting.

## Supported portable behavior

- Configured depth validity gates, iteration count, and learning rate.
- Single input contract: `[H,W]` depth returns `(Pose, float, int)`.
- Batched input contract: `[B,H,W]` depth with broadcastable `[3,3]` or
  `[B,3,3]` intrinsics and one/batched estimated transform returns
  `(Pose(B), loss[B], iterations[B])`.
- Explicit `env_indices` routes batch entries to unique dense-map environments;
  omitted indices select environments in order.
- Device-resident accepted state (`last_state`) records pose, loss, valid-depth
  counts, selected environments, and map generation; it is cloneable, resettable,
  and works on fallback-disabled float32 MPS.

## Explicit boundaries

Raw sparse-raycast sample count, tile size, LM trust-region/damping, and TSDF
weight controls rely on CUDA/Warp kernels and raise when changed from their
portable defaults. CUDA graph refinement, raw Warp raycasters, and sparse block
ABI state are not implemented; exact NVIDIA numerical parity is not claimed.

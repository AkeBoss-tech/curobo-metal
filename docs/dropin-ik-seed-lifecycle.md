# Portable IK and seeded-IK lifecycle

`IKSolver` and `SeedIKSolver` execute their real batched tensor objectives on
CPU and Apple Metal.  Their results retain cuRoboV2's `[batch, seed, dof]`
layouts, deterministic goalset ranking, typed goal/world updates, seed reset,
and finite-difference velocity support.

For a configured `max_batch_size`, a smaller IK request is internally padded
by repeating its first request and the public result is sliced back to the
original batch.  This preserves the V2 goal/seed lifecycle for callers that
reuse solver state; it does not imply CUDA Graph capture.  Raw CUDA graph
reset/capture and Warp packed LM-step APIs remain explicit `NotImplemented`
boundaries.

When `self_collision_check=True`, direct IK evaluates the configured
self-collision sphere pairs as a real CPU/MPS feasibility constraint and
optimization penalty.  A collision therefore cannot report success merely
because its tool pose converged.  World collision accepts `SceneCfg`, a list
of `SceneCfg`, a `SceneCollision`, and `SceneCollisionCfg`; YAML/USD/Warp
world asset loading remains unsupported.

The seeded LM budget follows V2's multi-link and high-main-seed policy, but
uses `torch.linalg.solve` (with a deterministic pseudo-inverse fallback) in
place of the CUDA/Warp packed ABI.  Exact CUDA stream, graph, and numeric
parity are intentionally not claimed.

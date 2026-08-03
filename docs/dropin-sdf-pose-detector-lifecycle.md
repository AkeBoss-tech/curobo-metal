# Portable SDF Pose Detector Lifecycle

`SDFPoseDetector` is a deterministic local rigid pose-refinement facade for
CPU and Apple MPS.  It consumes one `[N, 3]` observed point cloud and one
initial `Pose`; it deliberately rejects batched observations or initial-pose
batches rather than mixing independent registration problems.

The detector canonicalizes observations and the initial pose to float32 on its
configured CPU/MPS device.  Initial positions and quaternions must be finite,
and quaternions must be nonzero.  Optional articulated joint configuration is
also tensor- and finite-value validated before it is applied to `RobotMesh`.

Refinement follows the V2 outer/inner iteration lifecycle while preserving the
portable `max_iterations` budget, including a final partial group when the
budget is not divisible by `inner_iterations`.  `last_refinement_state` returns
a deep snapshot after a completed detection run; `reset()` discards that
snapshot and resets the completed-run count without mutating the mesh.

The implementation uses deterministic surface samples, nearest-point
correspondence, and eager PyTorch LM normal equations.  It is not a replacement
for the CUDA/Warp mesh-SDF BVH, mesh IDs, CUDA graph capture, raw alignment
kernels, or their numerical/performance characteristics.  Those remain
explicit unsupported boundaries.

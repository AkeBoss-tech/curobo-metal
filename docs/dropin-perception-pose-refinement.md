# Portable pose refinement

`SDFPoseDetector` and `BlockSparseRaycastPoseRefiner` are executable on CPU
and float32 MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`.

`SDFPoseDetector` accepts a `RobotMesh`, segmented depth/points, and a required
local `Pose` estimate.  It deterministically samples the portable mesh and runs
nearest-surface rigid registration with monotonic accepted updates.  It returns
the pinned `DetectionResult` fields (`pose`, configuration, confidence,
alignment error, iteration count, and elapsed time).  It intentionally does not
perform V2's global CUDA/Warp rotation search, so an initial estimate is part of
the portable contract.

The detector also exposes V2's local refinement lifecycle:
`_setup_refinement`, `_evaluate_at_pose`, `_refine_iteration`, and
`_refine_inner_iterations`.  `SDFRefinementState` accepts its pinned named
normal-equation buffers as well as the original compact portable constructor;
`clone` and `copy_` retain deep-copy/in-place semantics.  The normal equations
use robust Huber-weighted point-to-point residuals from deterministic mesh
surface samples, so they are useful for bounded local alignment on CPU/MPS but
are not a raw signed-distance or Warp-BVH substitute.

`BlockSparseRaycastPoseRefiner` accepts a portable dense `Mapper` or its TSDF
integrator facade.  It validates depth and uses the actual dense renderer to
refine a camera transform.  Its public call returns `(Pose, alignment_error,
n_iterations)`, matching the V2 high-level return shape.

The following remain explicit boundaries: Warp mesh IDs/BVH traversal, sparse
hash-table raycasting, raw two-pass SDF kernels, CUDA graph capture, Blox and
LiDAR mapping.  The dense renderer currently refines camera translation along
the viewing axis; it is real map-backed tensor work but not a replacement for
the raw Warp Levenberg-Marquardt kernel or a claim of NVIDIA numerical parity.

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

`BlockSparseRaycastPoseRefiner` accepts a portable dense `Mapper` or its TSDF
integrator facade.  It validates depth and uses the actual dense renderer to
refine a camera transform.  Its public call returns `(Pose, alignment_error,
n_iterations)`, matching the V2 high-level return shape.

The following remain explicit boundaries: Warp mesh IDs/BVH traversal, sparse
hash-table raycasting, raw two-pass SDF kernels, CUDA graph capture, Blox and
LiDAR mapping.  The dense renderer currently refines camera translation along
the viewing axis; it is real map-backed tensor work but not a replacement for
the raw Warp Levenberg-Marquardt kernel or a claim of NVIDIA numerical parity.

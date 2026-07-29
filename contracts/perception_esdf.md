# Depth-fused voxel/ESDF perception contract

This contract defines Wave 8A's bounded dense mapping seam for CPU and Apple
MPS. It targets the public concepts at pinned cuRoboV2 revision
`8e734f3ced1df898990bcd92de40abce475907db`, not binary or numerical identity
with its CUDA/Warp block-sparse mapper.

## Coordinates, cameras, and depth

World coordinates are right-handed metres. A camera looks along positive
camera `z`; `x` points right and `y` down. `camera_to_world` is a proper
homogeneous transform:

`p_world = R_camera_to_world p_camera + t_camera_to_world`.

Pinhole intrinsics are `[[fx,0,cx],[0,fy,cy],[0,0,1]]`, with positive focal
lengths and pixel coordinates starting at zero. Depth is axial camera `z` in
metres (not range along a unit ray). A value is valid exactly when it is
finite and `depth_min <= depth <= depth_max`. Zero, negative, NaN, infinity,
and values outside the inclusive range contribute no observation.

Depth accepts `[H,W]`, multiple cameras `[C,H,W]`, and environment-batched
`[E,C,H,W]`. Intrinsics and poses accept corresponding singleton broadcast
forms. All tensors share dtype and device. CPU supports float32/float64; MPS
supports float32. Camera batches fuse in serialized camera order, although
unit-weight arithmetic is order independent before weight saturation.

## Dense voxel layout

`shape=(nx,ny,nz)` and tensors use `[environment,x,y,z]`. Samples are at voxel
centres:

`center(i) = grid_center + (i - (shape - 1)/2) * voxel_size`.

This is the same centre convention consumed by
`curobo_metal.ops.world_collision.VoxelGrid`. Requested extents therefore
span voxel centres; their outer bounds lie half a voxel farther out.

## Projective TSDF and occupancy fusion

Every voxel centre is transformed into each camera, projected with the
pinhole model, and sampled at the nearest pixel using PyTorch/NumPy
round-to-nearest-even. Voxels behind the camera or outside the image are
ignored. For an accepted depth `d` and voxel axial coordinate `z`:

`sdf = d - z`, `tsdf_observation = clamp(sdf / truncation_distance, -1, 1)`.

Samples farther than one truncation distance behind the measured surface
(`sdf < -truncation_distance`) are ignored. Samples in front of the surface
are integrated, including the saturated free-space band. Fusion is a
unit-weight running average. At `max_weight`, accumulated value and weight
are scaled together; this bounds storage without changing the current
average. TSDF is normalized and dimensionless. An observed voxel is occupied
when `tsdf <= occupancy_threshold` (zero is occupied).

This projective rule deliberately omits upstream block allocation, adaptive
weights, color/features, decay, raycasting, and mesh extraction.

## Dense ESDF sign, distance, and gradient

The ESDF is positive in free space and negative in occupied space. For each
cell, exact brute-force Euclidean centre distance is computed to the nearest
cell of the opposite class. The represented surface lies halfway between
the two cell centres, so magnitude is
`max(center_distance - voxel_size/2, voxel_size/2)`. Exact ties select the
lowest flattened xyz index.

The stored gradient is the signed unit direction from the winning opposite
cell toward the query cell. It is zero for empty/all-occupied maps and at
undefined ties. With no occupied cells, ESDF is the positive configured
`unobserved_esdf`; with no free cells it is the negative value. ESDF
construction is discrete and is not differentiable through occupancy.
TSDF values remain differentiable with respect to valid sampled depths away
from projection, validity, truncation, and occupancy boundaries.

## State, updates, reset, and queries

`PerceptionMapper` owns a fixed number of environments. `update` replaces
state functionally and increments an environment generation only if at least
one voxel was integrated. A supplied unique `env_indices` vector updates
only those environments. `reset()` restores every map to TSDF 1, weight 0,
no occupancy, positive unobserved ESDF, zero gradient, and generation 0;
indexed reset affects only selected environments.

`voxel_grids()` exposes one identity-oriented ESDF grid per environment.
`query()` delegates to the existing differentiable trilinear world-collision
query, including environment routing and padding. Queries require the full
eight-sample stencil to be in bounds.

No CPU fallback is permitted on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=0`).

## Errors and empty inputs

Invalid shapes, transforms, focal lengths, ranges, devices, dtypes, duplicate
or out-of-range environment indices raise before mutation. A camera frame
containing no valid contributing depth is a successful no-op. Zero cameras
are not a valid observation.

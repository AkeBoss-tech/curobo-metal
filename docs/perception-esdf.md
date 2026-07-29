# Perception ESDF implementation

Wave 8A adds a self-contained depth-to-TSDF-to-ESDF path in
`curobo_metal.ops.perception`, plus an independent loop-based NumPy oracle.
The production implementation uses ordinary PyTorch tensor operators, stays
on CPU or MPS for the entire pipeline, and feeds the existing voxel world
collision query without conversion through host memory.

## Pinned upstream semantics inspected

The pinned cuRoboV2 source was inspected at revision
`8e734f3ced1df898990bcd92de40abce475907db`:

- `curobo/_src/types/camera.py` defines metric conversion, pinhole
  intrinsics, camera poses, and batched camera observations.
- `curobo/_src/perception/mapper/mapper_cfg.py` defines physical xyz extent,
  centre-aligned voxels, TSDF truncation, depth min/max, camera count,
  separate TSDF/ESDF resolution, update weights, and reset-oriented mapper
  state.
- `integrator_tsdf.py`, `integrator_esdf.py`, and the associated Warp kernels
  establish projective TSDF fusion followed by signed Euclidean distance
  construction.
- `test_multi_camera.py`, `test_block_integrate.py`, and
  `test_integrator.py` exercise batched cameras, repeat integration, and map
  lifecycle behavior.

The portable implementation preserves those public ideas while defining its
own deterministic dense numerical contract. It does not claim parity with
upstream block hashing, gather/scatter seed dilation, PBA/JFA selection,
visibility allocation, decay, feature/color grids, static stamping, or
CUDA-graph behavior. The authoritative portable details are in
`contracts/perception_esdf.md`.

## Architecture and parity boundary

`integrate_depth` is functional: state and an observation produce a new
`DenseMap`. `PerceptionMapper` adds selective environment mutation and reset.
The oracle separately loops over cameras and voxels with NumPy. Both use xyz
storage so the ESDF tensor can be installed directly in the pre-existing
`VoxelGrid`.

TSDF sampling is differentiable with respect to accepted depth pixels. Pixel
rounding, validity masks, occupancy classification, nearest-site ESDF, and
winner selection are intentionally discrete. ESDF query position gradients
come from the existing trilinear sampler and agree with the interpolating
polynomial; the mapper's diagnostic centre gradients are nearest-site unit
vectors and are not used to override autograd.

Dense ESDF currently forms a pairwise distance matrix. This is exact and
useful for bounded manipulation volumes and correctness fixtures, but memory
is quadratic in voxel count. Profiling did not justify a custom Metal kernel
for this initial bounded scope: projection/gather/fusion and `torch.cdist`
already execute natively on MPS, and a fused kernel would duplicate the exact
EDT work still needed. Larger maps need a separable EDT or sparse/block
implementation in a later wave.

## Verification

Synthetic tests cover centre coordinates, planar depth, camera translation,
invalid depth, repeated and multi-camera fusion, two environments, selective
update/reset, NumPy parity, depth autograd, ESDF sign, collision-query
integration, empty state, and fallback-disabled MPS execution. Fixtures are
generated analytically in tests and do not copy upstream data.

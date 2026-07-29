# Wave 4D world-collision scope and cuRoboV2 compatibility

This wave establishes independent CPU truth for triangle meshes, voxel SDFs,
and ESDF queries. It does not add a torch/Metal production path. The immutable
upstream comparison is NVlabs/curobo
`8e734f3ced1df898990bcd92de40abce475907db` (2026-07-22).

## Compatible concepts

- Mesh and voxel obstacles are stored in per-environment slots and queried by a
  batch environment index.
- Poses are applied world-to-local for SDF evaluation and gradients are rotated
  local-to-world.
- Mesh SDF is negative inside and positive outside.
- Voxel storage is `[nx,ny,nz]`, C-order, center aligned with the
  `-0.5` continuous-index shift, and uses analytic trilinear gradients.
- Sphere collision uses radius plus activation distance, a strict positive
  penetration test, and the same C1 quadratic/linear activation profile.

## Deliberate gaps and clarified differences

- Pinned V2 uses Warp BVHs (`wp.mesh_query_point`) and accepts triangle soups.
  Warp supplies a sign even when topology is unsuitable for a reliable solid.
  This reference guarantees signed values only for caller-declared watertight
  meshes and otherwise exposes unsigned distance explicitly.
- V2 mesh query failure returns a bounding-box-derived maximum distance. The
  brute-force finite reference never has a BVH query failure.
- V2 stores voxel features as float16 and evaluates kernels in float32. The
  oracle retains float64 source samples so quantization/backend error can be
  measured separately.
- Pinned V2's boundary interpolation renormalizes valid corners in its slow
  path. Wave 4D instead marks an incomplete eight-corner stencil invalid and
  returns the grid's OOB sentinel. Device compatibility should expose the mode
  rather than silently conflate these policies.
- V2's generic collision kernel accumulates obstacle costs and gradients with
  float atomics. `query_esdf` is a deterministic minimum-field query; it is not
  an emulation of V2 atomic sum ordering across overlapping obstacles.
- V2's mesh/voxel helper gradient is stored as `-grad(sdf)` for its custom
  collision/backward convention (the pinned mesh sign fix is specifically
  documented in source). Wave 4D functions return conventional `grad(sdf)`;
  `sphere_world_collision` returns the ordinary derivative of cost.
- V2 public query tensors are `[batch,horizon,spheres,4]`; the reference uses
  `[B,Q,3]` points plus a separate activation helper. Callers may flatten
  `(horizon,spheres)` into `Q`.
- V2 includes swept kernels, reusable buffers, cache mutation, voxel/ESDF
  construction, and Warp/Isaac integration. Those remain outside this wave.

## Fixture coverage

`tests/fixtures/world_collision/` contains an outward-oriented watertight box, a
watertight tetrahedron, an open triangle mesh, an analytic affine voxel field,
and two environments with a rigidly transformed grid. The correctness artifact
is generated entirely from these deterministic inputs.

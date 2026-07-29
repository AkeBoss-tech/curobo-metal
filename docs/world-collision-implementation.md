# Production world-collision implementation

Wave 5C implements the Wave 4D contract in
`curobo_metal.ops.world_collision` using differentiable PyTorch operations on
CPU and MPS. It has no NumPy, cuRobo, Warp, or fallback dependency.

## Semantics

- Mesh queries transform world points into each obstacle frame, evaluate every
  serialized triangle, and reduce with PyTorch's first-index `min`. Signed
  queries require every mesh to be declared watertight; open meshes remain
  explicitly unsigned. Odd/even classification is discrete, while distance
  gradients use the selected closest feature. Exact and region-boundary
  subgradients are installed deterministically.
- Voxel grids use center-aligned continuous indices and ordinary eight-corner
  trilinear interpolation. The last center uses the final cell at fraction one
  with its positive-cell one-sided derivative. An incomplete stencil returns
  the finite OOB sentinel, zero gradient, and invalid status.
- ESDF reduction ignores inactive and invalid candidates, subtracts padding
  after selecting the minimum, and preserves lowest-grid exact ties. Empty
  reductions return `+inf`, zero gradient, `-1`, and invalid.
- Sphere activation implements the contract's C1 quadratic/linear profile.
  Autograd through distance agrees with the returned conventional cost
  gradient.

The environment list is setup metadata. Grid tensors, transforms, points,
masks, and indices stay on one device. CPU accepts float32/float64; MPS accepts
float32 because PyTorch MPS does not provide float64.

## Optimization decision

The checked-in benchmark records representative triangle and 32-cube ESDF
queries. Wave 5C retains the portable implementation as the production path:
the current query loops primarily dispatch substantial tensor indexing and
arithmetic, while a compile-shader implementation would require a second custom
backward and substantially expand the verification surface. The benchmark
provides a stable threshold for a later fused kernel: optimize when these
queries are shown to dominate an end-to-end planning profile, preserving this
implementation as the explicit oracle-backed path. No silent fallback exists.

## Verification

`tests/ops/world_collision/test_core.py` compares values, validity, winners,
closest-feature gradients, transforms, masks, OOB behavior, ties, and
environment selection against the independent float64 reference. It also
checks autograd and runs an MPS float32 smoke query with fallback disabled.

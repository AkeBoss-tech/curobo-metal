# World-collision reference contract

## Scope and independence

`curobo_metal.reference.world_collision` is the float64, NumPy-only executable
specification for Wave 4D. It covers point/triangle-mesh distance, transformed
voxel SDF sampling, ESDF reduction, and sphere-to-SDF collision activation. It
does not import torch, Metal, cuRobo, Warp, Isaac, `trimesh`, or any production
operator. Swept collision, mesh construction/repair, voxelization, ESDF
generation, and pose gradients are out of scope.

## Common layout, frames, and reduction

World points are `[Q,3]` or `[B,Q,3]`; results retain `[B,...]`. Obstacles occupy
fixed slots in environments. `env_indices[B]` selects one environment per batch
row and defaults to environment zero. Transforms are proper local-to-world
rotations and translations:

```text
world = R @ local + t
local = R.T @ (world - t).
```

Inactive/OOB candidates are neutral: distance `+inf`, zero gradient, winner
`-1`. Exact reduction ties select the lowest serialized obstacle/face/grid
index. No tolerance manufactures ties. Inputs are finite floating arrays
(integer topology/indices); calculations and outputs are float64.

## Triangle meshes

Mesh faces are nondegenerate indexed triangles. Unsigned distance is the
Euclidean distance to the closest point on any triangle. Its gradient with
respect to the world query point is the unit vector from the closest surface
point to the query. At a surface point, where that derivative is not unique,
the deterministic subgradient is the selected face's normalized vertex-order
normal transformed to world space.

Signed distance uses the clearance convention: negative inside, zero on the
surface, positive outside. It is supported **only** when the caller explicitly
declares every mesh watertight. Inside/outside is an odd/even ray classification
and the unsigned gradient is multiplied by the sign. A `watertight=True`
declaration is a caller guarantee; the small oracle deliberately does not
perform topology repair or global manifold validation.

Non-watertight meshes and triangle soups have only an unsigned guarantee.
Requesting signed distance for one is an error. This avoids presenting
orientation- or ray-dependent pseudo-signs as geometry. At edges, vertices,
medial sets, and equal closest triangles, values remain exact while the first
serialized face supplies the documented subgradient.

## Voxel SDF and trilinear gradient

`VoxelGrid.values[nx,ny,nz]` uses C order (`z` fastest when flattened) and
stores samples at voxel centers. For local point `p`, continuous index is:

```text
v = p / voxel_size + [nx,ny,nz] / 2 - 0.5.
```

The floor index and its `+1` neighbor form the eight-corner stencil.
Interior values use ordinary trilinear interpolation. Gradients analytically
differentiate the same polynomial, divide by `voxel_size`, then rotate to the
world frame. On an integer sample plane, `floor` selects the cell on its
positive side; the value is continuous but a non-C1 field may have this
one-sided gradient. Every grid dimension is at least two.

The reference boundary rule is intentionally strict: if any stencil corner is
out of bounds, return the finite grid `out_of_bounds` value, zero gradient, and
`valid=False`. Thus the domain is the closed region between the first and last
sample centers. This is safer for collision reduction than partially
renormalizing an incomplete stencil and makes OOB distinct from an observed
large ESDF sample.

## ESDF query and sphere output

`query_esdf` takes the minimum valid, active grid value in the selected
environment, subtracts nonnegative `padding`, and returns distance, its
world-point gradient, winning grid, and validity. ESDF sign is negative in
occupied/interior space and positive in free space. No Euclidean-validity claim
is made for arbitrary supplied grids; interpolation reproduces their samples.

`sphere_world_collision` consumes any point SDF and gradient. With sphere radius
`r`, collision padding `p`, activation distance `eta`, and weight `w`:

```text
x = r + p + eta - sdf
cost = 0                              when x <= 0
cost = w * 0.5*x*x/eta                when 0 < x <= eta, eta > 0
cost = w * (x - 0.5*eta)              when x > eta
```

For `eta=0`, positive `x` uses the linear branch. Returned gradient is the
mathematical derivative of cost with respect to the sphere center:
`-w * branch_scale * grad(sdf)`. Radius derivatives and transform/grid-value
derivatives are not returned. `padding` expands the sphere independently of
activation. Boundary gradients at `x=0` are zero; at `x=eta` both branches
agree.

## Replay and tolerances

Canonical JSON has `format="curobo-metal-world-collision-case"`, `version=1`,
sorted keys, compact separators, no non-finite JSON numbers, and one newline.
Checked-in oracle values use `rtol=0, atol=1e-13`; finite differences use step
`1e-6`, `rtol=3e-6, atol=3e-8`. Future float64 device implementations use
`2e-12` absolute/relative tolerance; float32 uses `rtol=3e-5, atol=3e-6`.
Nondifferentiable boundary/tie tests assert the selected subgradient directly.

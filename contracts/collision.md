# Collision reference contract

## Scope and independence

`curobo_metal.reference.collision` is the float64, NumPy-only executable
specification for Wave 2B. It does not import PyTorch, cuRobo, Warp, CUDA, or an
MPS/Metal implementation. It covers link-local sphere transforms, configured
sphere-pair self collision, and discrete sphere-to-oriented-cuboid collision.
Meshes, voxel/ESDF worlds, swept collision, continuous collision, backend
dispatch, and cuRobo cost shaping are outside this contract.

## Common conventions

A sphere is `(x, y, z, radius)`. Radius is finite and nonnegative. Signed
distance is **clearance**: positive when shapes are separated, zero at contact,
and negative when interiors overlap. Nonnegative `padding` is subtracted from
clearance. All calculations and returned arrays use IEEE float64 regardless of
the floating input dtype. Integer indices are returned as int64 and masks must
be boolean.

Unbatched sphere input is `[S,4]`; batched input is `[B,S,4]`. Results always
retain the batch dimension, including `B=0`. A result records whether its input
was unbatched. Inputs must be real, floating, finite arrays. Invalid shapes,
indices, rotations, negative radii/extents/padding, and non-boolean masks are
errors. Non-contiguous inputs are accepted.

Inactive entries are padding, not zero-radius geometry. Their per-item distance
is `+inf`, gradient is zero, and winner is `-1` when no active candidate exists.
This makes a minimum reduction mathematically neutral. Replay JSON cannot encode
infinity, so checked-in expected arrays store only finite active values plus
masks/winners; a consumer reconstructs inactive values from this rule.

Minimum reductions traverse tables in serialized order. `numpy.argmin` semantics
select the first index on an exact tie. No tolerance is used to manufacture
ties.

## Sphere transforms

`transform_spheres(transforms, local_spheres, link_indices, active)` accepts
rigid or affine homogeneous transforms `[L,4,4]` or `[B,L,4,4]`, local spheres
`[S,4]`, indices `[S]`, and optional mask `[S]`. For active sphere `s`,

```text
world_center[b,s] = transforms[b,link[s]] @ [local_center[s], 1]
world_radius[b,s] = local_radius[s].
```

Output is `[B,S,4]`. An inactive output and Jacobian are exactly zero. The
provided center Jacobian `[B,S,3,4,4]` is with respect to the selected transform:
`d center[k] / d T[k,j] = homogeneous_local[j]`; all other entries are zero.
It is a derivative of the affine expression and does not parameterize the rigid
transform manifold.

## Sphere-sphere self collision

`pairs[P,2]` is an explicit ordered pair table. Duplicate/reversed pairs are
permitted and remain distinct reduction candidates; self-pairs and out-of-range
indices are errors. A pair is active only if its pair mask and both sphere masks
are true. For pair `(i,j)`:

```text
d = ||center_i - center_j|| - radius_i - radius_j - padding.
```

Per-pair distances are `[B,P]`; gradients with respect to all sphere components
are `[B,P,S,4]`. Center gradients are `+unit(center_i-center_j)` for `i` and its
negative for `j`; both radius gradients are `-1`. At coincident centers the norm
is nondifferentiable: the selected subgradient is `(+x for i, -x for j)`. The
reduced result is the minimum pair distance with `[B]` values, `[B,S,4]`
gradients, and `[B]` winning pair indices.

## Sphere-to-cuboid collision

Cuboids use centers `[C,3]`, proper orthonormal local-to-world rotations
`[C,3,3]`, and nonnegative half extents `[C,3]`. For world sphere center `p`,
local point `u=R.T@(p-center)`, `q=abs(u)-half`, and box signed distance:

```text
box_sdf = ||max(q,0)|| + min(max(q),0)
d = box_sdf - sphere_radius - padding.
```

Per-pair output is `[B,S,C]`; sphere gradients are `[B,S,C,4]`. The world-center
gradient is `R` times the standard box-SDF local gradient and the radius
gradient is `-1`. Reduction is independently over cuboids for every sphere,
yielding `[B,S]`, `[B,S,4]`, and winning cuboid `[B,S]`.

Outside an edge/corner, the gradient is the normalized vector from the nearest
box point. Inside or on the box, the gradient uses the axis with greatest `q`
(nearest face); exact axis ties select lowest axis. `sign(0)` is defined as
`+1`. These deterministic choices are valid subgradients and cover the box
center, medial planes, edges, corners, and surface boundaries.

## Replay and tolerances

Collision fixtures use canonical UTF-8 JSON:
`format="curobo-metal-collision-case"`, `version=1`, sorted object keys, compact
separators, no NaN/infinity, and one trailing newline. Numbers and arrays have
identical logical layouts on CPU, MPS, and CUDA; implementations must not
regenerate random inputs per backend. Each case carries its operation, inputs,
masks, expected active values/gradients/winners, and tolerance.

Reference-vs-checked-in float64 values use `rtol=0, atol=1e-13`. A future
float64 device implementation uses `rtol=2e-12, atol=2e-12`; float32 uses
`rtol=2e-5, atol=2e-6`. Analytic gradients use the same value tolerances away
from nondifferentiable sets. Central differences use step `1e-6`,
`rtol=3e-6, atol=3e-8`. Boundary tests assert the exact documented value and
selected subgradient rather than finite differences.

## Portable checker aggregation

The compatibility checker consumes an explicit pair table plus an optional
sphere-active mask. Its default self-collision output is the maximum
nonnegative penetration magnitude over active pairs. With
`sum_distance=True`, it instead sums those magnitudes. Empty or fully filtered
pair tables return zero. These checker-level reductions do not change the
signed-clearance convention of the primitive operation above.

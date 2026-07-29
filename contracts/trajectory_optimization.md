# Trajectory optimization and motion generation reference contract

Status: Wave 4B executable reference, version 1.

## Scope and independence

`curobo_metal.reference.trajectory` is a deterministic float64 NumPy
specification. It composes the existing serial FK and primitive collision
references and imports no PyTorch, cuRobo, Warp, CUDA, Isaac, MPS, graph
planner, production solver, or Metal kernel. Its projected-gradient optimizer
is intentionally slow and is normative only for checked-in replay generation.

In scope are a joint-space trajectory representation, endpoint and inclusive
joint-limit constraints, velocity/acceleration/jerk costs, discrete and
linearly interpolated collision samples, deterministic seed batches,
minimum-jerk initialization, outcome metrics, and failure categorization.
Graph/search planning, dynamics, torque limits, time optimization, mesh/ESDF
collision, and production motion-generation policy are outside this contract.

## Representation, time, and batching

A trajectory is `[T,J]`, in movable-joint order, with `T >= 2`. Knot `i` is at
physical time `i*dt`, where finite `dt > 0` is uniform and measured in seconds.
Angles are radians and prismatic coordinates are metres. Inputs must be real,
floating, finite arrays; the oracle evaluates and returns float64 and accepts
non-contiguous float32/float64 inputs.

A problem has either seeds `[N,T,J]` or no seeds. Seeds are solved independently
in serialized order and every result array retains `N`. With no seeds, `N=1`
and the initial trajectory is

```text
s = i/(T-1)
b(s) = 10 s^3 - 15 s^4 + 6 s^5
q[i] = start + b(s) (goal-start).
```

This quintic has zero continuous-time velocity and acceleration at both
endpoints. No randomness is used. Exact objective ties select the first
successful seed.

## Costs and gradients

Endpoint cost is the squared residual of knots zero and `T-1`:
`0.5*w_e*(||q[0]-start||^2 + ||q[-1]-goal||^2)`.
Joint-limit cost is the zero-margin squared hinge from `contracts/ik.md`,
summed over all knots.

For finite difference order `n` in `{1,2,3}`, define
`D_n q = diff(q,n)` and:

```text
C_n = 0.5 * w_n * dt^(1-2n) * sum((D_n q)^2).
```

Thus these are rectangular-rule integrals of squared velocity, acceleration,
and jerk samples. The exact gradient is
`w_n*dt^(1-2n)*D_n.T@D_n q`. Terms with `T <= n` have value and gradient zero.
All weights are finite and nonnegative.

Collision cost evaluates the existing squared activation hinge at sampled robot
states. Its reference state gradient uses central differences with default
`h=1e-6`; interpolation distributes each state gradient to its two bounding
knots by the exact linear weights. Endpoint, limit, and smoothness gradients
are analytic.

## Swept sampling

`collision_subdivisions=K >= 1` samples coordinates
`0, 1/K, ..., T-1`. Each segment includes its left knot and `K-1` interior
linear interpolants; only the last segment includes its right knot, so no state
is duplicated. `K=1` is discrete knot collision. These are sampled swept
states, not a continuous-collision guarantee. Sphere/cuboid signs, padding,
activation, masks, tie rules, and subgradients are inherited unchanged from
`contracts/collision.md`.

## Oracle and projection

Every initial and trial trajectory is clipped to inclusive joint limits.
Endpoints are then set to the clipped requested endpoints; feasible endpoints
therefore remain exact. Projected steepest descent zeros endpoint gradient,
uses deterministic Armijo backtracking (at most 30 halvings), and never wraps
joint coordinates. The implementation reports attempted iterations and the
final objective but is not a performance baseline or required backend
algorithm.

## Success, metrics, and failure categories

A seed succeeds when both:

- maximum start/goal Euclidean error is `<= endpoint_tolerance`;
- minimum signed clearance over every configured discrete/swept sample is
  `>= -collision_tolerance`.

No collision geometry gives minimum clearance `+inf` in memory and JSON `null`.
The result reports trajectories, per-seed success, status, attempted
iterations, objective, endpoint error, minimum clearance, and selected seed.
Statuses are:

- `success`;
- `endpoint_infeasible` (a requested endpoint lies outside a joint bound);
- `collision_constrained` (final sampled clearance violates tolerance);
- `stationary`;
- `line_search_failed`;
- `max_iterations`.

Observable endpoint/collision failures take precedence over generic optimizer
termination. Failure is result data; malformed inputs are exceptions.

## Replay, tolerances, and required cases

Replay JSON uses `format="curobo-metal-trajectory-case"`, `version=1`, sorted
keys, compact separators, finite numbers, and one trailing newline. It contains
the complete robot, world, start/goal, limits, `T`, `dt`, optional seeds,
weights, sampling/solver options, and expected outputs. Regeneration is
byte-deterministic through `tests/fixtures/trajectory/generate.py`.

Checked-in float64 costs, trajectories, and metrics use `rtol=0, atol=2e-12`.
Analytic/finite-difference gradients away from hinges and collision
nondifferentiabilities use `h=1e-6`, `rtol=3e-6, atol=3e-8`; composed collision
gradients allow `rtol=2e-4, atol=2e-5` because they nest finite differences.
A backend candidate need not reproduce oracle knots: it passes by satisfying
the serialized success category and independently recomputed endpoint,
joint-limit, and sampled-clearance tolerances.

Required fixtures are obstacle-free and obstacle-detour two-link motions, a
Panda-class obstacle-free motion, an infeasible endpoint case, and a
limit-constrained seed. Tests also cover jerk gradients, physical `dt`,
interpolation, seed ties, non-contiguous inputs, invalid metadata, canonical
serialization, and dependency independence.

# Backend-neutral IK and cost reference contract

Status: Wave 3B executable reference, version 1.

## Scope and independence

`curobo_metal.reference.costs` and `curobo_metal.reference.ik` are deterministic
float64 NumPy specifications. They may compose the independent FK and collision
references, but import no PyTorch, cuRobo, Warp, CUDA, Isaac, MPS, or production
optimizer. The solver is intentionally slow: it is an oracle for compact serial
chains and replay cases, not a production algorithm or performance baseline.

The problem/result dataclasses are optimizer-independent. A conforming solver
may use any algorithm, but it must consume the same targets, seed batch, limits,
weights, collision model, and stopping tolerances and report per-seed outcomes.

## Pose convention and costs

Frames, matrices, units, and joint ordering follow `forward_kinematics.md`.
Quaternions are scalar-first `[w,x,y,z]`, active, local-to-parent rotations.
Inputs are normalized; zero and nonfinite quaternions are errors. Since `q` and
`-q` encode one rotation, output uses the hemisphere `w >= 0`, then makes the
first nonzero component positive when `w == 0`.

For current transform `(R,p)` and target `(Rt,pt)`, pose error is

```text
e = [p - pt, Log(Rt.T @ R)]
```

Translation is in world coordinates. Rotation is the shortest rotation vector
in target coordinates with norm in `[0,pi]`. At exactly `pi`, quaternion
canonicalization selects the deterministic axis sign. Pose cost is
`0.5 * sum(weights * e**2)` and its derivative with respect to `e` is
`weights * e`.

Joint-limit cost uses a squared hinge inside an optional margin:
`0.5*w*(max(lo+margin-q,0)^2 + max(q-(hi-margin),0)^2)`. Its derivative at the
hinge is zero. Trajectory smoothness is the sum of squared first and optional
second differences over `[T,J]`; the reference returns its exact gradient.

Collision cost accepts signed clearances from the existing references and is
`0.5*w*sum(max(activation-clearance,0)^2)`. Its clearance derivative is
`-w*max(activation-clearance,0)`, including zero at activation. Robot collision
composition is FK -> local sphere transform -> configured self/world distances
-> collision cost. Candidate order and boundary subgradients remain those of
`contracts/collision.md`.

## IK inputs, batching, and limits

One `IKProblem` holds one target and seeds `[N,J]`. Each seed is solved
independently in serialized order; result arrays retain `N`, including failures.
The reference clips each initial/trial point to inclusive finite joint limits.
It never silently changes target or weights. Optional wrapping maps revolute
joints to `[-pi,pi)` before clipping; prismatic joints are never wrapped.
Wrapping is disabled by default because multi-turn limits can be meaningful.

The oracle uses central finite differences (`h=1e-6`), projected steepest
descent, and deterministic Armijo backtracking. This algorithm is normative
only for checked-in reference replays, not for device implementations.

## Success, selection, and failures

A seed succeeds only when all enabled conditions hold simultaneously:

- translation norm <= `position_tolerance`;
- rotation-vector norm <= `rotation_tolerance`;
- composed collision cost <= `collision_tolerance`.

The selected seed is the successful seed with least final objective; exact ties
select the first. If none succeeds it is null. Iterations count attempted
optimization iterations. Per-seed status is one of:

- `success`;
- `max_iterations`;
- `infeasible_or_stationary` (finite-difference gradient too small);
- `line_search_failed`;
- `limit_constrained` (failure ends on a joint bound);
- `collision_constrained` (failure retains collision cost above tolerance).

The last two observable constraints take precedence over generic optimizer
failures. Failure is data, not an exception. Invalid shapes, values, limits, or
metadata are exceptions.

## Derivatives, tolerances, and replay

Analytic scalar-cost gradients are checked with central differences using
`h=1e-6`, `rtol=3e-6`, `atol=3e-8`, away from hinge and rotation-log
nondifferentiabilities. Pose values and checked-in float64 replay values use
`rtol=0`, `atol=1e-12`. IK success is tolerance-defined rather than requiring
identical joint coordinates because redundant chains admit multiple solutions.
A backend candidate passes when it reports the expected success category and
its independently recomputed residuals satisfy the serialized tolerances.

Replay is canonical UTF-8 JSON with
`format="curobo-metal-ik-case"`, `version=1`, sorted keys, compact separators,
finite JSON numbers, and one trailing newline. It contains complete robot and
collision metadata, target, seed batch, limits, weights, solver tolerances, and
expected per-seed status/residual evidence. It never stores Python objects,
device state, or regenerated random input.

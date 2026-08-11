# Motion generation compatibility

> Historical scope note: this page documents the package-owned
> `curobo_metal.motion_gen` JSON facade. It does not describe every newer
> `curobo` high-level compatibility loader. The release-wide contract is
> `docs/compatibility.md`.

The portable facade targets the motion-planning concepts in pinned cuRoboV2
revision `8e734f3ced1df898990bcd92de40abce475907db`. At that revision upstream's
new public name is `MotionPlanner`/`MotionPlannerCfg`; `MotionGen` remains the
widely used integration vocabulary. `curobo_metal.motion_gen` provides a
stable, Mac-capable `MotionGen` surface while using only functionality that is
actually implemented in this repository.

## Supported surface

- `MotionGenConfig.load_from_robot_config` and `from_dict` load the portable
  serial-robot JSON schema. Existing trajectory replay fixtures are accepted.
- `MotionGen.warmup`, `plan_single_js`/`plan_single_joint_space`,
  `plan_batch_js`/`plan_batch_joint_space`, `plan_single`/`plan_single_pose`,
  and `plan_batch`/`plan_batch_pose` are stable entry points.
- `JointState`, `Pose`, `MotionGenResult`, `MotionGenStatus`, and
  `MotionGenMetrics` are package-owned types. Results expose optimized and
  interpolated plans, derivatives, solve time, attempt count, graph use,
  metrics/debug records, and `get_interpolated_plan()`.
- Joint-space planning performs direct trajectory optimization and
  deterministically retries with graph-derived seeds when enabled. Pose
  planning performs seeded IK before the same joint-space pipeline.
- Robot spheres, self-pairs, and primitive cuboid worlds compose the
  production FK/collision paths. CPU uses float32 or float64; MPS uses float32.
  Requested devices are validated and never silently replaced.
- Runtime primitive worlds can be replaced or cleared. Link-local collision
  spheres can be attached and detached; both operations invalidate optimizer
  and roadmap state. Retract IK seeds, multiple trajectory seeds, retry/graph
  attempt policies, and linear/cubic/B-spline interpolation are supported.

The portable config requires `robot`, `lower`, and `upper`. Solver fields such
as `steps`, `dt`, `interpolation_dt`, seed counts, tolerances, graph parameters,
and trajectory weights may be supplied under `motion_gen` or as loader keyword
arguments. A replay fixture's `inputs` and `options` are translated directly.

## Determinism and failure behavior

IK and graph sampling use the configured integer graph seed. Graph planning
uses its existing PCG64 replay contract and stable tie breaking. Result status
distinguishes invalid starts/goals, IK failure, graph failure, and trajectory
failure. Malformed tensor shapes, dtypes, devices, and configuration fields
raise immediately; planning infeasibility is returned as a result.

Batch calls preserve input order. A batch containing any failed item returns
per-item success/status without fabricating a partial stacked trajectory.

## Explicitly unsupported

The loader does not parse upstream YAML, URDF, USD, or XRDF. Generate or export
the documented JSON representation outside this package. Mesh/BVH, voxel,
ESDF, depth-camera, continuous-collision, dynamics, Isaac Sim, Omniverse,
CUDA Graph, grasp geometry beyond link-local spheres, and non-primitive runtime
worlds are not impersonated. There is no CPU fallback for requested MPS
execution. The portable MotionGen audit record is evidence-blocked pending
paired pinned CUDA replay; local test success is not a CUDA-equivalence claim.

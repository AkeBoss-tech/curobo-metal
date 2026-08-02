# Portable robot builder

`curobo.robot_builder.RobotBuilder` preserves the cuRobo V2 builder workflow
for URDF robot descriptions on CPU and MPS.  A builder can load a configuration,
collect supported link collision spheres, construct neighbour-link ignores,
save its configuration, and load that generated YAML again.

## Supported portable geometry

URDF `<sphere>` visual or collision geometry is exact: `fit_collision_spheres`
collects it unchanged (apart from an optional clip plane).  This makes the
result directly usable by the portable kinematics and collision checker.  The
method accepts the V2 fitting arguments so callers do not need a separate
platform branch; values that do not alter an exact primitive are validated but
otherwise have no effect.

## Explicit boundaries

Fitting arbitrary mesh, box, capsule, or cylinder link geometry is not silently
approximated.  It raises `NotImplementedError` and requires the optional mesh
sphere-fit backend.  Sampled self-collision pruning has no portable broad phase;
`compute_collision_matrix` therefore returns neighbouring-link ignores and
emits a warning.  XRDF export, USD/Isaac assets, and Viser visualization remain
explicitly unavailable.

This is a portable implementation boundary, not a CUDA numerical-equivalence
claim for Warp mesh fitting or collision-pair pruning.

# Release compatibility contract

This is the authoritative human-readable compatibility contract for
`curobo-metal` 0.1.0a1. Machine-readable evidence lives in
`artifacts/parity/capabilities.json`, `artifacts/api_compat/upstream-api.json`,
and `artifacts/parity/replay/`. Older `dropin-*` and wave documents describe
individual implementation slices; when they conflict with this page, this page
and the checked release artifacts control.

## Release status

The alpha is a bounded Apple Silicon/MPS preview, not a complete source-
unchanged replacement for NVIDIA cuRobo. Compatibility targets exactly cuRobo
commit `8e734f3ced1df898990bcd92de40abce475907db`.

The distribution exposes both `curobo_metal` and a compatibility-oriented
`curobo` namespace. NVIDIA cuRobo and `curobo-metal` must not be installed in
the same environment because both own `curobo`.

## Audited alpha claims

- All 361 pinned upstream runtime module paths import from the installed wheel.
- The documented non-`_src` public facades have no known static export or
  callable-shape differences in the current inventory audit.
- CPU and Apple MPS production paths exist for the documented configuration,
  types, kinematics, collision, cost, IK, trajectory, graph-planning,
  perception, dynamics, and high-level planning slices. Requested MPS execution
  never silently falls back to CPU in release tests.
- Eighteen of nineteen bounded replay capabilities have checked-in paired
  pinned-CUDA/Metal evidence. Passing a bounded replay is evidence only for its
  serialized operation and schema, not for every method or numerical regime in
  that subsystem.
- The wheel includes a hash-pinned Franka 0.7.0 URDF/mesh subset and five
  NVIDIA YAML configs. See `THIRD_PARTY_NOTICES.md` and
  `artifacts/release/asset-provenance.json`.

## Not yet claimed

The alpha does not claim full `_src` compatibility. The strict audit currently
reports 1,263 missing AST-discovered exports and 446 callable-shape differences
under `_src`; many are imported typing/backend implementation names, but the
remaining user-relevant members have not all been classified and closed.

L-BFGS now has fallback-disabled MPS and pinned-CUDA outcome evidence for an
eager, batched quadratic solve. The shared claim deliberately excludes hard
action projection and terminal freezing because the pinned upstream line
search does not implement those options.

Motion generation is the sole replay label without valid end-to-end CUDA
parity evidence: the old replay exercised minimum-jerk interpolation, not the
real MotionPlanner/MotionGen stack.

Motion generation remains evidence-blocked until its probe is replaced and run
on the pinned CUDA host.

The paired EvolutionStrategies contract covers the stable natural-gradient
mean update with covariance updates disabled. The pinned upstream covariance
update produced non-finite results for the shared signed-utility corpus, so
covariance adaptation is explicitly outside the current equivalence claim.

## Platform and integration boundaries

CUDA graph capture, CUDA streams, NVRTC, raw Warp packed ABIs, Isaac Sim,
Omniverse, ROS, USD authoring/viewers, and NVIDIA-only visualization or external
asset integrations are platform substitutions or external-unavailable
features. They must fail explicitly; the project does not emulate them with
unrelated behavior.

Portable observable solver behavior remains in scope. Mesh, voxel/ESDF,
collision, dynamics, attachments/world mutation, spline interpolation, and
particle/gradient optimizer support vary by the specific documented facade;
do not infer support or rejection from an older wave note. Consult the runtime
API, its tests, and the capability inventory for the exact slice.

## Stable-release gates

A stable drop-in claim requires all of the following:

1. Classify every strict `_src` export/signature difference and make the
   supported-symbol gate fail closed.
2. Replace and pass the remaining invalid MotionGen parity probe, then broaden all nineteen
   capabilities across dtype/device, batch/layout, invalid/infeasible,
   mutation/cache, gradient, collision-boundary, and repeatability cases.
3. Classify the 211 pinned upstream test modules and 14 examples; execute every
   applicable item unchanged against the installed wheel and record exclusions.
   `artifacts/api_compat/upstream-execution-census.json` now tracks all 225
   entries and intentionally fails `--require-reviewed` until that review is
   complete.
4. Pass clean wheel and sdist installs on the supported Python matrix, full
   fallback-disabled Apple MPS tests, CUDA replay, metadata/license checks, and
   namespace-conflict checks from a clean tagged commit.

Until those gates close, release notes and package metadata must retain the
alpha/non-drop-in warning.

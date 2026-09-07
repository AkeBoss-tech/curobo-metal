# Release compatibility contract

This is the authoritative human-readable compatibility contract for
`curobo-metal` 1.0.0. Machine-readable evidence lives in
`artifacts/parity/capabilities.json`, `artifacts/api_compat/upstream-api.json`,
`artifacts/api_compat/ecosystem-api.json`, and `artifacts/parity/replay/`. Older `dropin-*` and wave documents describe
individual implementation slices; when they conflict with this page, this page
and the checked release artifacts control.

## Release status

The stable release is a bounded portable Python drop-in for the audited Apple
Silicon/MPS scope, not a CUDA/Warp ABI or external-ecosystem replacement.
Compatibility targets exactly cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`.

The distribution exposes both `curobo_metal` and a compatibility-oriented
`curobo` namespace. NVIDIA cuRobo and `curobo-metal` must not be installed in
the same environment because both own `curobo`.

## Audited stable claims

- The release's strict public-facade gate is exact for 24 modules, 122 exports,
  and 10 comparable callable signatures. The broader generated inventory
  resolves 571/586 audited modules; 15 remain partial (two perception internals
  and 13 bundled example workflows that include integration-specific CLI state). Static inventory coverage is not a claim
  that every upstream internal or example workflow is implemented.
- The real-world API corpus resolves all 171 documented, example-used, and
  downstream-used compatibility targets collected from 545 symbols across
  upstream and 73 downstream source files.
- CPU and Apple MPS production paths exist for the documented configuration,
  types, kinematics, collision, cost, IK, trajectory, graph-planning,
  perception, dynamics, and high-level planning slices. Requested MPS execution
  never silently falls back to CPU in release tests.
- All nineteen bounded replay capabilities have checked-in paired
  pinned-CUDA/Metal evidence. Passing a bounded replay is evidence only for its
  serialized operation and schema, not for every method or numerical regime in
  that subsystem.
- The wheel includes 40 byte-exact, hash-pinned upstream robot, configuration,
  and scene assets. See `THIRD_PARTY_NOTICES.md` and
  `artifacts/release/asset-provenance.json`.
- All 225 pinned upstream test/example modules are classified: 55 applicable,
  15 external-unavailable, 35 not-applicable, and 120 platform-substituted.
  Installed-wheel gauntlet reports preserve case-level portable results and
  exact reasons for hardware-mechanism exclusions; the classifications do not
  turn excluded CUDA/Warp calls into passing Metal behavior.

## Bounded claims

The [application gate](application-compatibility.md) additionally verifies 30
hash-pinned portable V2 programs against an installed wheel, using default MPS
selection and graph settings. All 30 have paired, hash-bound CUDA evidence
with zero comparison differences. This is
not a claim that the original NVIDIA CUDA-specific tutorials execute unchanged.

Omitted device requests now select MPS when available, otherwise CPU. Explicit
requests are preserved; CPU must be specified for float64 workloads on a Mac.
This is a platform substitution for upstream's CUDA default.

The exact static surface is not a claim of byte-identical source or behavior in
every numerical regime. The 19 paired adapters certify their declared corpora
and matrices; they do not turn platform-specific ABIs into portable APIs.

The high-level `Mapper` selects block-sparse storage when its nominal dense
mirror would exceed 1 GiB. Camera and LiDAR fusion, sparse queries, bounded ESDF
materialization, rendering, clearing, checkpoints, and analytic static geometry
then operate without allocating the nominal dense volume.

Bundled examples provide import-compatible portable entry points, but the 13
partial example modules are not complete replacements for NVIDIA/Isaac/ROS/USD
workflows.

L-BFGS now has fallback-disabled MPS and pinned-CUDA outcome evidence for an
eager, batched quadratic solve. The shared claim deliberately excludes hard
action projection and terminal freezing because the pinned upstream line
search does not implement those options.

Motion generation now has fallback-disabled MPS and pinned-CUDA outcome
evidence for the real high-level `MotionPlanner` C-space lifecycle. The shared
case covers a single-attempt packaged-Franka solve, public trajectory layout,
endpoint convergence, path finiteness, status, and zero-attempt failure. It
does not claim parity for pose IK, graph fallback, world mutation, or every
MotionGen policy.

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

1. The strict public-facade export/signature gate is fail-closed and complete;
   broader internal/example inventory gaps remain explicitly reported.
2. All nineteen paired capabilities cover their declared dtype/device,
   batch/layout, invalid/infeasible, mutation/cache, gradient,
   collision-boundary, and repeatability matrices.
3. Classify the 211 pinned upstream test modules and 14 examples; execute every
   in-scope portable case against the installed wheel and record exact
   exclusions. The census has no unreviewed modules; release evidence must also
   show zero failures among cases classified as portable.
4. Pass clean wheel and sdist installs on the supported Python matrix, full
   fallback-disabled Apple MPS tests, CUDA replay, metadata/license checks, and
   namespace-conflict checks from a clean tagged commit.

The release workflow enforces these checks on the free self-hosted Apple
Silicon runner before it can build or publish a tagged distribution.

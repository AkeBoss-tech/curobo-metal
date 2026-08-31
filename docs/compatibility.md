# Release compatibility contract

This is the authoritative human-readable compatibility contract for
`curobo-metal` 0.1.0a1. Machine-readable evidence lives in
`artifacts/parity/capabilities.json`, `artifacts/api_compat/upstream-api.json`,
and `artifacts/parity/replay/`. Older `dropin-*` and wave documents describe
individual implementation slices; when they conflict with this page, this page
and the checked release artifacts control.

## Release status

The alpha is a bounded portable Python drop-in for the audited Apple
Silicon/MPS scope, not a CUDA/Warp ABI or external-ecosystem replacement.
Compatibility targets exactly cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`.

The distribution exposes both `curobo_metal` and a compatibility-oriented
`curobo` namespace. NVIDIA cuRobo and `curobo-metal` must not be installed in
the same environment because both own `curobo`.

## Audited alpha claims

- The strict pinned Python surface is exact: 361/361 runtime modules, 3,870/
  3,870 AST-discovered exports, and 747/747 callable shapes.
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
- All 225 pinned upstream test/example modules are classified. The 55
  applicable modules pass unchanged against a clean installed wheel (1,116
  tests); 13 external-unavailable, 35 not-applicable, and 122
  platform-substituted entries retain explicit reasons.

## Bounded claims

The exact static surface is not a claim of byte-identical source or behavior in
every numerical regime. The 19 paired adapters certify their declared corpora
and matrices; they do not turn platform-specific ABIs into portable APIs.

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

1. The strict `_src` export/signature gate is fail-closed and complete.
2. All nineteen paired capabilities cover their declared dtype/device,
   batch/layout, invalid/infeasible, mutation/cache, gradient,
   collision-boundary, and repeatability matrices.
3. Classify the 211 pinned upstream test modules and 14 examples; execute every
   applicable item unchanged against the installed wheel and record exclusions.
   This gate is complete: the census has no unreviewed entries, and the 55
   applicable modules pass 1,116 unchanged tests in aggregate.
4. Pass clean wheel and sdist installs on the supported Python matrix, full
   fallback-disabled Apple MPS tests, CUDA replay, metadata/license checks, and
   namespace-conflict checks from a clean tagged commit.

Until the remaining packaging and publication mechanics close, release notes
and package metadata retain the alpha warning.

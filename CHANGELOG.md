# Changelog

All notable user-visible changes are recorded here.

## 0.1.0a1 - Unreleased

First public alpha of `curobo-metal`, an Apple Silicon/MPS implementation that
exposes a compatibility-oriented `curobo` namespace.

### Included

- Portable PyTorch CPU/MPS implementations for the documented configuration,
  kinematics, collision, cost, IK, trajectory, graph, perception, and
  whole-body slices.
- Runtime-compiled Metal kernels for selected kinematics and collision hot
  paths, with MPS fallback-disabled tests.
- The strict pinned Python surface is exact: all 361 runtime modules, 3,870
  exports, and 747 callable shapes match the inventory.
- Nineteen bounded CUDA-vs-Metal replay capabilities have fresh, provenance-
  bound RTX 3090 evidence, including invalid, edge, and multi-case matrices.
- Forty byte-exact, hash-pinned upstream robot/config/scene assets with their
  applicable third-party notices.
- The complete 225-item pinned upstream workload census is classified. All 55
  applicable modules pass unchanged in a clean installed-wheel run (1,116
  tests); 122 platform-substituted, 35 not-applicable, and 13 external entries
  retain explicit reasons.

### Alpha limitations

- Static surface equality and the bounded numerical matrices do not assert
  byte-identical source, identical stochastic samples or solver iterates, or
  behavior in every untested numerical regime.
- CUDA graphs, CUDA streams, NVRTC, Warp packed ABIs, Isaac/Omniverse, ROS, USD
  authoring/viewers, and platform-specific integrations are outside this alpha.
- Do not install `curobo-metal` and NVIDIA cuRobo in the same Python environment;
  both own the `curobo` import namespace.

Compatibility is pinned to NVIDIA cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`. See `docs/parity-matrix.md` for the
bounded evidence and explicit exclusions.

Publication remains blocked until the TestPyPI/PyPI Trusted Publisher projects
and protected GitHub environments described in `docs/releasing.md` are
configured and the TestPyPI dry run succeeds.

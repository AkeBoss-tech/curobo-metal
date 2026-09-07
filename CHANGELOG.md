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
- The strict public-facade gate matches all 24 audited modules, 122 exports,
  and 10 comparable callable signatures. The broader generated inventory
  resolves 571/586 modules and reports the remaining 15 partial modules.
- The ecosystem corpus resolves all 171 documented, example-used, and
  downstream-used compatibility targets. Thirty hash-pinned applications run
  unchanged through an installed wheel with automatic MPS selection.
- Nineteen bounded CUDA-vs-Metal replay capabilities have fresh, provenance-
  bound RTX 3090 evidence, including invalid, edge, and multi-case matrices.
- Forty byte-exact, hash-pinned upstream robot/config/scene assets with their
  applicable third-party notices.
- The complete 225-item pinned upstream workload census is classified: 55
  applicable, 120 platform-substituted, 35 not-applicable, and 15 external.
  The installed-wheel case gauntlet exercises 2,852 cases: all 2,749 portable
  cases pass, while 103 exact CUDA/Warp/MPS-float64 mechanism cases remain
  explicitly excluded.

### Alpha limitations

- Static surface equality and the bounded numerical matrices do not assert
  byte-identical source, identical stochastic samples or solver iterates, or
  behavior in every untested numerical regime.
- CUDA graphs, CUDA streams, NVRTC, Warp packed ABIs, Isaac/Omniverse, ROS, USD
  authoring/viewers, and platform-specific integrations are outside this alpha.
- Large high-level maps automatically use block-sparse camera/LiDAR fusion,
  queries, bounded ESDF materialization, rendering, clearing, and checkpoints
  without allocating the nominal dense volume.
- Thirteen bundled example workflows and two perception internals remain partial
  in the broad static inventory; examples are a portable subset, not full
  NVIDIA/Isaac/ROS/USD workflow replacements.
- Do not install `curobo-metal` and NVIDIA cuRobo in the same Python environment;
  both own the `curobo` import namespace.

Compatibility is pinned to NVIDIA cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`. See `docs/parity-matrix.md` for the
bounded evidence and explicit exclusions.

Publication remains blocked until the TestPyPI/PyPI Trusted Publisher projects
and protected GitHub environments described in `docs/releasing.md` are
configured and the TestPyPI dry run succeeds.

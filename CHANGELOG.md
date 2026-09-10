# Changelog

All notable user-visible changes are recorded here.

## 1.0.1 - Unreleased

### Fixed

- Preserve successful Levenberg–Marquardt IK seeds instead of always applying
  up to 200 eager Adam refinement steps. This fixes repeated reachable-target
  failures observed in the `1.0.0` wheel and reduces the matched M4 workload
  from about 315 ms per successful solve to about 17.9 ms median.
- Check eager-MPS convergence after every LM step instead of completing a
  four-step CUDA-oriented inner group, and remove repeated validation/device
  synchronizations from solver-owned batches.
- Route compatible serial-tree robot models through the fused Metal FK kernel,
  cache static joint-order tensors, and avoid a redundant MPS environment-index
  synchronization for the default sphere environment.

### Added

- A minimal installed-package FK example using the public `curobo` namespace.
- A reproducible Pillow-rendered motion-planning GIF using the bundled Franka
  meshes and its public-API generator.
- A practical Apple M4 versus RTX 3090 benchmark table, cold-start context,
  intended-use guidance, optimization roadmap, and launch-post drafts.

## 1.0.0 - 2026-09-07

First stable release of `curobo-metal`, an Apple Silicon/MPS implementation
of the pinned portable cuRobo Python contract through the standard `curobo`
namespace.

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

### Compatibility boundaries

- Static surface equality and the bounded numerical matrices do not assert
  byte-identical source, identical stochastic samples or solver iterates, or
  behavior in every untested numerical regime.
- CUDA graphs, CUDA streams, NVRTC, Warp packed ABIs, Isaac/Omniverse, ROS, USD
  authoring/viewers, and platform-specific integrations are outside this release.
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

The release workflow runs the full fallback-disabled test suite, ecosystem API
gate, installed-wheel 30-application gate, and Apple Silicon performance gate
before building artifacts for TestPyPI or PyPI.

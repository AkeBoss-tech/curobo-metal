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
- All 361 pinned upstream runtime module paths are importable from the wheel.
- Fifteen bounded CUDA-vs-Metal replay capabilities have checked-in paired
  evidence, including cubic B-spline trajectory generation.
- A hash-pinned Franka robot/config subset with complete third-party notices.

### Alpha limitations

- This is not yet a source-unchanged or full-behavior cuRobo replacement.
- The current strict `_src` surface audit still contains unclassified export
  and callable-shape differences.
- Real cross-backend PRM, L-BFGS, evolution-strategy, and full MotionGen replay
  probes are not complete. Existing placeholder/narrow probes for those names
  are not parity evidence.
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

# cuRoboV2 API parity and release boundary

The authoritative machine-readable capability audit is
`artifacts/parity/capabilities.json`. It covers the
production capability surface after the complete Wave 9 portable closure and
cites the immutable upstream revision
`8e734f3ced1df898990bcd92de40abce475907db`. The generator rejects any other
upstream checkout and validates every cited upstream symbol, local symbol, and
test file.

The inventory deliberately separates five conclusions:

- **semantically equivalent**: the narrowly named mathematical contract is
  locally oracle-backed; this never means that its containing upstream class is
  drop-in compatible;
- **partial**: useful production behavior exists, but listed portable gaps are
  material;
- **intentionally platform-inapplicable**: CUDA graphs, streams, extensions, or
  fused CUDA/Warp kernels are NVIDIA implementation mechanisms rather than
  portable Metal capabilities;
- **external-integration-only**: Isaac Sim/Omniverse, ROS, USD authoring, and
  viewer features belong to external ecosystems and are outside the core
  package;
- **evidence-blocked**: a plausible implementation exists, but the required
  paired pinned-upstream NVIDIA replay has not covered its full supported
  surface, so equivalence is not claimed.

## Audit result

| Classification | Count | Result |
|---|---:|---|
| semantically equivalent | 23 | Every audited portable capability, backed by local tests and its required paired evidence |
| partial | 0 | No known portable implementation gaps remain in the audited core |
| intentionally platform-inapplicable | 1 | CUDA graphs and fused CUDA/Warp execution machinery |
| external-integration-only | 1 | Isaac/ROS/USD/viewer ecosystem integrations |
| evidence-blocked | 0 | No portable capability remains blocked on NVIDIA replay evidence |

The 19 capabilities that previously awaited NVIDIA evidence are promoted only
when the inventory generator verifies the committed paired report, pinned
revision, identical input hashes, and executed invalid, edge, and multi-case
matrix evidence on both backends.

### Full-matrix pinned CUDA evidence

On 2026-08-26, a fresh checked-in handoff ran against the exact upstream
revision on an NVIDIA RTX 3090. The resulting provenance-bound
[paired report](../artifacts/parity/cuda-replay/paired-report-2026-08-26-matrix.json)
passes all 19 available CUDA adapters: configuration,
`DeviceCfg`, `Pose`, `JointState`, solver results, forward kinematics,
geometric Jacobians, robot-scene sphere collision, a bounded unsigned Warp
mesh query, bounded scene-level voxel/ESDF queries, position pose cost, and
inverse dynamics, high-level IK and trajectory-optimization outcomes, and the
compiled cubic B-spline boundary kernel, declarative PRM planning outcomes,
multi-seed EvolutionStrategies mean-update outcomes, eager batched L-BFGS
quadratic outcomes, and a real high-level MotionPlanner C-space lifecycle.
The dynamics category
includes torque plus first-order VJPs for position, velocity, and
acceleration.

An earlier 2026-08-07 single-pose IK report is retained as historical evidence;
the 2026-08-26 full matrix supersedes it for release discovery. Franka is
redundant, so neither report treats distinct valid joint-space minima as a
numerical failure. This is outcome equivalence, not identical-solution or full
solver-trajectory identity.

This is real CUDA-versus-fallback-disabled-Metal evidence for the declared
portable contracts. Every registry case includes expanded batch, layout,
mutation, invalid, edge, and multi-case matrix coverage. It does not claim
Metal reproduces CUDA-only streams, graphs, fused-kernel ABI, raw RNG samples,
or identical solver iterates.

The collision semantic-equivalence records are deliberately smaller than an
upstream checker API: discrete sphere/sphere and sphere/oriented-cuboid signed
distance, including tested gradients. The broader checker, configuration,
types, solver, and motion-planning surfaces are equivalent for the explicitly
documented portable scope.

## Remaining implementable gaps

Every `implementable_gaps` list in the JSON artifact is empty. No known
portable core implementation gap remains in the audited 25 capability groups.

CUDA graph/kernel ports and Isaac/ROS/USD/viewer adapters are excluded from that
implementable core-gap list by classification. A paired NVIDIA runner is an
evidence requirement, not an implementation gap.

Regenerate the artifact with:

```sh
python3 tools/parity/inventory.py \
  --upstream /tmp/curobo-v2 \
  --output artifacts/parity/capabilities.json
```

## CUDA-versus-Metal replay

Wave 10 adds a turnkey, registry-driven replay corpus at
`artifacts/parity/replay/`. All nineteen records retain fallback-disabled MPS
output bundles. `graph.prm_planner` now executes real PRM scenarios and
`optim.particle_evolution` now exercises the `EvolutionStrategies` facade with
multi-seed semantic checks; both have fresh Metal evidence, and both
now also have pinned CUDA outcome evidence. `optim.lbfgs` likewise has fresh
fallback-disabled MPS and pinned-CUDA semantic evidence for its eager batched
quadratic contract. That contract does not claim terminal freezing or hard
action projection, which the pinned upstream line search rejects or omits.
`motion_generation.motion_gen` now executes `MotionPlanner.plan_cspace` on the
packaged Franka model instead of the old minimum-jerk stand-in. Its semantic
comparison checks the public 81-knot active-joint trajectory, endpoints,
success/status, finite path length, and zero-attempt behavior without requiring
identical solver iterates.
Each case owns an explicit JSON corpus
specification under `artifacts/parity/replay/corpus/`; its input NPZ contains
only that capability's declared tensors rather than a shared opaque superset.
The manifest hashes both the input/output data and the common/case corpus JSON
that produced it. The validator reconstructs those NPZ tensors from the corpus,
so a hash-consistent but different input cannot be substituted.

Every case has registry-owned invalid-case and edge-case behavior. Generation
records executable `invalid_rejected` and `edge_observed` int8 evidence bits;
the former means the declared invalid behavior was observed and can include a
valid-but-observed compatibility edge such as the pinned zero-quaternion rule.
The validator fails closed if either output is absent, false, malformed, or its
metadata no longer matches the corpus and registry. Local pre-promotion bundles
do not establish CUDA equivalence; the promoted 2026-08-26 report compares the
same evidence on both backends.

Replay bundles contain only JSON plus `allow_pickle=False` NPZ tensors. This
keeps inputs identical across isolated CUDA and macOS environments and records
shapes, dtypes, hashes, backend, runtime, operation, case ID, and upstream SHA.
Each backend runner must serialize input tensors before execution and all
observable outputs afterward, including gradients, statuses/errors, selected
device, and fallback flags.

```sh
python -m tools.parity.replay pack --bundle run-cuda \
  --inputs inputs.npz --outputs cuda-outputs.npz \
  --case-id panda-fk-b8 --operation forward_kinematics --backend cuda
python -m tools.parity.replay pack --bundle run-metal \
  --inputs inputs.npz --outputs metal-outputs.npz \
  --case-id panda-fk-b8 --operation forward_kinematics --backend metal
python -m tools.parity.replay compare run-cuda run-metal \
  --rtol 1e-5 --atol 1e-6
```

Regenerate and validate the complete committed corpus on Apple Silicon with
fallback disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python -m tools.parity.generate_replay \
  --device mps --output artifacts/parity/replay
PYTHONPATH=src python -m tools.parity.validate_replay artifacts/parity/replay
```

Run one pinned CUDA-side case using the exact committed input:

```sh
PYTHONPATH=src python -m tools.parity.run_pinned_cuda \
  --upstream /opt/curobo-v2 \
  --input artifacts/parity/replay/kinematics.forward_kinematics/inputs.npz \
  --capability kinematics.forward_kinematics \
  --output cuda-replay/kinematics.forward_kinematics
```

On an NVIDIA host, run all 19 registered adapters
and immediately verify the outputs against the committed fallback-disabled
Metal corpus:

```sh
PYTHONPATH=.:src python -m tools.parity.run_cuda_ready \
  --upstream /opt/curobo-v2 \
  --metal-root artifacts/parity/replay \
  --output cuda-replay
```

When the NVIDIA host cannot access this repository directly, build a
deterministic self-verifying handoff archive:

```sh
PYTHONPATH=.:src python -m tools.parity.build_cuda_handoff \
  --output /tmp/curobo-metal-cuda-handoff.zip
```

After transfer and extraction, `./run-cuda.sh /path/to/pinned/curobo` verifies
every packaged file before running. The preflight refuses a non-pinned checkout,
an unavailable CUDA device, or upstream type imports resolving outside that
checkout. Each successful CUDA manifest records Python, platform, PyTorch,
CUDA runtime, NVIDIA driver, GPU name, compute capability, device count, and
device memory.

This writes one hashed `cuda-manifest.json` and `cuda-outputs.npz` per ready
capability, followed by `cuda-replay/paired-report.json`. The command exits
nonzero if the upstream revision, consumed input hash, output hash, tensor
schema, registry tolerance, backend provenance, or any numerical comparison
fails. A report is complete only when all registered real CUDA adapters are
present and passing. For every ready configuration, type, and solver-result
adapter, the compared output schema also includes an executed invalid-case outcome
(`invalid_rejected`), so a report cannot pass using happy-path values alone.

The CUDA command refuses any checkout other than
`8e734f3ced1df898990bcd92de40abce475907db` before importing upstream. The 19
adapters cover the configurations, values, geometry, kinematics, collision,
cost, dynamics, optimizer, graph, trajectory, IK, and MotionGen contracts named
in the full-matrix report. Exact cases, tensors, semantic validators,
tolerances, and retained exclusions are defined in
`tools/parity/replay_registry.py`; no CUDA result is synthesized. Jacobian
higher-order AD/dJdq, raw RNG samples, solver-iterate identity, fused ABI, and
platform-only mechanisms remain outside the declared contracts.

Per-capability tolerances are likewise registry-owned: exact structural cases
use zero tolerance; type/pose cases use `1e-6/1e-7`; collision and costs use
roughly `2e-5/2e-6`; FK/Jacobian use `8e-5` to `1e-4`; and iterative
optimization, planning, and dynamics use `2e-4` to `5e-4` relative tolerance.
The validator requires the manifest values, corpus hashes, decoded tensor
values, invalid-case outcome, and edge-case outcome to match this registry
exactly.

The final matrix covers its declared supported dtypes/devices, fallback
behavior, first-order gradients, zero/singleton/many batches, noncontiguous
inputs, invalid shapes/dtypes, infeasible outcomes, and collision boundaries or
ties where applicable. Passing local reference tests alone is not CUDA parity
evidence.

## Packaging and license

Project code is Apache-2.0. The wheel distributes the Apache-2.0 Franka 0.7.0
URDF/OBJ/DAE tree and five NVIDIA cuRobo YAML configs, byte-identical to pinned
cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`. Copyright/SPDX headers and the
Franka license are retained; `THIRD_PARTY_NOTICES.md` records attribution and
`artifacts/release/asset-provenance.json` pins every distributed file hash. No
USD or other upstream robot family is distributed.

Build and import in a disposable macOS environment with:

```sh
tools/parity/macos-clean-smoke.sh
```

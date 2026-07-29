# cuRoboV2 API parity and release boundary

This repository is **not a full cuRoboV2 port**. The authoritative
machine-readable audit is `artifacts/parity/capabilities.json`. It covers the
production capability surface present on local `main` at `aa1e349` and cites
the immutable upstream revision
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
  paired pinned-upstream NVIDIA replay was unavailable, so equivalence is not
  claimed.

## Audit result

| Classification | Count | Result |
|---|---:|---|
| semantically equivalent | 4 | Scoped collision primitives, portable whole-body dynamics, and portable perception/mapping contracts |
| partial | 17 | Configuration/types, Jacobians, checker and mesh/voxel collision, costs, optimizers, IK, trajectory, graph, and motion generation |
| intentionally platform-inapplicable | 1 | CUDA graphs and fused CUDA/Warp execution machinery |
| external-integration-only | 1 | Isaac/ROS/USD/viewer ecosystem integrations |
| evidence-blocked | 2 | Forward kinematics and production inverse dynamics |

Forward kinematics was previously labeled semantically equivalent. Local
analytic fixtures and fallback-disabled tests establish internal correctness,
but are not independent evidence from the pinned NVIDIA implementation. It is
therefore evidence-blocked until a paired runner produces matching replay
bundles. Inverse dynamics has the same evidence issue, but no remaining local
production gap: inverse/forward dynamics, named-state adapters, and rollout now
share the differentiable CPU/MPS implementation.

Semantic-equivalence records remain scoped contracts rather than package-wide
claims. Robot-scene caches, filtering, aggregation, environment updates, result
layouts, and swept queries remain partial.

## Remaining implementable gaps

The exact per-record list is in `implementable_gaps` in the JSON artifact. The
remaining portable work groups are:

- finish supported RobotCfg/YAML/XRDF fields, asset/mimic handling, and
  mutation/serialization compatibility;
- complete portable `DeviceCfg`, `Pose`, `JointState`, solver result, and
  cost/config adapters;
- integrate Jacobians with upstream-style buffers/results and finish link and
  whole-body modes;
- add self-collision filtering/aggregation, full cache/environment mutation,
  exact mesh/voxel boundary and result semantics, and swept collision;
- add remaining portable particle sampling, line-search, termination, and
  optimizer state/config behavior;
- complete IK solve modes/seeds/collision integration and trajectory rollout,
  solve-mode, interpolation, and result behavior;
- add persistent graph lifecycle and remaining sampling/search modes;
- complete MotionGen retries, solve modes, attachment/world mutation,
  interpolation, timing, metrics, and debug results;
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

A release comparison must cover supported dtypes and devices, explicit fallback
behavior, first-order gradients, zero-size and singleton/many batches,
noncontiguous inputs, invalid shapes/dtypes, infeasible solver outcomes, and
collision boundary/tie cases. Passing local reference tests is not CUDA parity
evidence.

## Packaging and license

Project code is Apache-2.0. No upstream robot, mesh, USD, or other separately
licensed asset is distributed. Upstream `LICENSE_ASSETS` governs assets a user
obtains separately.

Build and import in a disposable macOS environment with:

```sh
tools/parity/macos-clean-smoke.sh
```

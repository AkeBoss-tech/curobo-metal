# cuRoboV2 API parity and release boundary

This repository is **not a full cuRoboV2 port**. The authoritative
machine-readable audit is `artifacts/parity/capabilities.json`. It covers the
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
  paired pinned-upstream NVIDIA replay was unavailable, so equivalence is not
  claimed.

## Audit result

| Classification | Count | Result |
|---|---:|---|
| semantically equivalent | 4 | Scoped collision primitives, whole-body tree math, and the portable perception lifecycle |
| partial | 0 | No known portable implementation gaps remain in the audited core |
| intentionally platform-inapplicable | 1 | CUDA graphs and fused CUDA/Warp execution machinery |
| external-integration-only | 1 | Isaac/ROS/USD/viewer ecosystem integrations |
| evidence-blocked | 19 | Implemented portable surfaces awaiting paired pinned-upstream replay evidence |

Forward kinematics was previously labeled semantically equivalent. Local
analytic fixtures and fallback-disabled tests establish internal correctness,
but are not independent evidence from the pinned NVIDIA implementation. It is
therefore evidence-blocked until a paired runner produces matching replay
bundles. Production differentiable inverse dynamics has the same remaining
evidence issue.

The collision semantic-equivalence records are deliberately smaller than an
upstream checker API: discrete sphere/sphere and sphere/oriented-cuboid signed
distance, including tested gradients. The broader checker, configuration,
types, solver, and motion-planning surfaces are complete for the supported
portable scope but remain evidence-blocked rather than overclaimed.

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
`artifacts/parity/replay/`. It has one fallback-disabled MPS output bundle for
each of the 19 evidence-blocked records. The committed `index.json` hashes every
manifest, and every manifest hashes its byte-identical portable input corpus and
local output. The manifests also record device, runtime, operation, tolerance,
gradient/status coverage, an invalid or edge case, and explicitly set
`equivalence_claimed` to false.

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

On an NVIDIA host, run every currently implemented asset-independent adapter
and immediately verify the outputs against the committed fallback-disabled
Metal corpus:

```sh
PYTHONPATH=.:src python -m tools.parity.run_cuda_ready \
  --upstream /opt/curobo-v2 \
  --metal-root artifacts/parity/replay \
  --output cuda-replay
```

This writes one hashed `cuda-manifest.json` and `cuda-outputs.npz` per ready
capability, followed by `cuda-replay/paired-report.json`. The command exits
nonzero if the upstream revision, consumed input hash, output hash, tensor
schema, registry tolerance, backend provenance, or any numerical comparison
fails. A report is complete only when all registered real CUDA adapters are
present and passing.

The CUDA command refuses any checkout other than
`8e734f3ced1df898990bcd92de40abce475907db` before importing upstream. The clean
runner has real asset-independent adapters for `DeviceCfg`, `Pose`, and
`JointState`. On a CUDA host those adapters write hashed upstream output bundles;
they do not claim equivalence until comparison succeeds. The other 16 cases
deliberately emit `external_constraint`: robot, world, inertial, and solver
surfaces still require caller-supplied assets and/or CUDA-compiled upstream
objects. The exact per-capability constraint is in
`tools/parity/replay_registry.py`; no CUDA result is synthesized.

Per-capability tolerances are likewise registry-owned: exact structural cases
use zero tolerance; type/pose cases use `1e-6/1e-7`; collision and costs use
roughly `2e-5/2e-6`; FK/Jacobian use `8e-5` to `1e-4`; and iterative
optimization, planning, and dynamics use `2e-4` to `5e-4` relative tolerance.
The validator requires the manifest values to match this registry exactly.

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

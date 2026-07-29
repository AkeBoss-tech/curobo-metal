# cuRoboV2 API parity and release boundary

This repository is **not a full cuRoboV2 port**. The authoritative,
machine-readable audit is `artifacts/parity/capabilities.json`, generated
against immutable upstream revision
`8e734f3ced1df898990bcd92de40abce475907db`. The generator refuses another
revision and validates every cited file and symbol.

Classifications mean:

- **compatible**: the public contract is accepted without adaptation;
- **semantically equivalent**: observable results match the targeted contract,
  although names, containers, or implementation differ;
- **partial**: a useful subset exists and the recorded blocker is material;
- **unsupported**: no corresponding supported capability exists;
- **not applicable**: an upstream implementation mechanism is intentionally
  backend-specific and is not itself a portable public capability.

## Release summary

| Area | Classification | Exact boundary |
|---|---|---|
| configuration | partial | Kinematics metadata conversion only; no upstream parser or mutable `RobotCfg` |
| forward kinematics | semantically equivalent | Portable batched poses/spheres; different API and execution |
| self/world collision | partial | Core discrete queries; upstream caches, wrappers, swept queries, and exact mesh/voxel semantics are absent |
| costs, IK, graph, trajectory, motion generation | partial | Contract-focused portable implementations, not drop-in upstream solvers/results/configuration |
| dynamics | partial | CPU reference only; no production Metal API |
| cuRobo data types | unsupported | No `JointState`, `Pose`, result, or config drop-in layer |
| CUDA graphs/kernels | not applicable | Compare outputs, gradients, failures, empty inputs, and batches—not CUDA machinery |
| Isaac/ROS/USD integrations | unsupported | Outside package scope |

The JSON inventory contains one record per capability with upstream and local
source evidence, symbol line numbers, and an exact blocker. Regenerate it with:

```sh
uv run python tools/parity/inventory.py \
  --upstream /tmp/curobo-v2 \
  --output artifacts/parity/capabilities.json
```

## CUDA-versus-Metal replay

Replay bundles contain only JSON plus `allow_pickle=False` NPZ tensors. This
keeps inputs identical across isolated CUDA and macOS environments and records
shapes, dtypes, hashes, backend, runtime, operation, case ID, and upstream SHA.
Each backend runner should serialize input tensors before execution, execute its
native public API, then serialize all observable outputs including gradients,
status/error codes, selected device, and fallback flags.

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

A release comparison must cover float32/float64 where supported, CPU/MPS/CUDA
device placement, explicit fallback behavior, first-order gradients, zero-size
batches, batch size one and many, noncontiguous inputs, invalid shapes/dtypes,
unreachable/infeasible solver outcomes, and collision boundary/tie cases.
Passing local reference tests is not evidence of CUDA parity; only paired
bundles from the pinned upstream and this package can close those items.

## Packaging, license, and clean environment

Project code is Apache-2.0. No upstream robot, mesh, USD, or other separately
licensed asset is distributed. The repository's JSON fixtures are original
small numeric test cases. Upstream `LICENSE_ASSETS` still governs any assets a
user separately obtains; they must not be copied into a wheel without review.

Build and import in a disposable macOS environment with:

```sh
tools/parity/macos-clean-smoke.sh
```

The script builds a wheel, installs only that wheel into a fresh virtual
environment, imports from outside the checkout, checks a public op import, and
runs `pip check`. For a low-disk local rerun using already installed
dependencies, set `CUROBO_METAL_SMOKE_PYTHON=.venv/bin/python` and
`CUROBO_METAL_SMOKE_USE_SYSTEM_PACKAGES=1`; the release gate should use the
default fully isolated mode.

# Unchanged portable V2 applications

The application gate runs eight project-authored programs through the standard
`curobo` namespace, against an installed wheel. Their source and shared reporting
helper are hash-pinned. The same files are supplied for CUDA replay; the runner
never rewrites their imports, calls, device choices, graph flags, or assertions.
These are portable applications targeting upstream commit
`8e734f3ced1df898990bcd92de40abce475907db`, not unchanged NVIDIA tutorials.

## Covered workflows

| Application | Behavioral checks |
| --- | --- |
| Device/state | Default allocation, clone ownership, tensor serialization |
| Pose | wxyz/xyzw conversion, composition, inverse, clone ownership |
| Franka FK | Packaged assets, 16 configurations, repeated calls |
| UR10e FK | Second robot configuration, joint-name reordering |
| FK gradient | Nonzero gradient and central finite difference |
| Collision mutation | Cuboid penetration, world replacement, restoration |
| IK | Moved reachable goal, endpoint accuracy, unreachable goal, input ownership |
| Motion planning | Nonzero C-space move, repeated planning, endpoint accuracy, zero-attempt result |

All programs leave device and CUDA-graph settings at their public defaults.
Solver programs also retain default seed counts and optimization settings.
The first run exposed two production mismatches: cuboid collision constraints
discarded positive penetration penalties, and zero-attempt C-space planning
raised instead of returning `None`. Both are fixed. Constraint regression tests
also cover spheres, capsules, cylinders, and mixed scenes on CPU and MPS.
Applications report tensor shape, dtype, device, finite values, and selected
numerical outcomes. The runner requires all reported compute tensors to reside
on the requested backend. Reporting and serialization may explicitly copy values
to the host. Fallback-disabled execution does not prove absence of every deliberate
CPU operation inside the implementation.

CUDA replay also exposed that IK's `js_solution` must include configured locked
joints while `solution` contains active joints. The Metal result now preserves
that distinction, including zero locked-joint velocities. Retargeting selects
active-joint velocities from the expanded result. The IK application uses the
public reorder method before FK. The pose application constructs a robot context
before pose algebra, which initializes upstream's geometry runtime through the
public API; it does not import Warp directly.

## Default device policy

`DeviceCfg()` and the backend's omitted-device resolver select `mps:0` when available,
otherwise CPU. Explicit CPU stays CPU. Explicit unavailable MPS is never redirected
to CPU. Configuration objects can still describe devices on a different machine
for serialization; allocation/dispatch validates availability. MPS float64 remains
unsupported: request CPU explicitly for float64 workloads.
The concrete default index matches allocated tensors, preserving equality between
`cfg.device` and `tensor.device`.

This is a declared platform substitution for the pinned upstream CUDA default.
It does not add CUDA execution to curobo-metal. CUDA reference execution uses the
separately installed upstream distribution. A public API signature whose default
contains a device configuration therefore has a platform-dependent representation.

Python function defaults such as `device_cfg=DeviceCfg()` resolve at import time,
as in upstream; newly constructed `DeviceCfg` values resolve at construction time.
No global PyTorch default device is changed.

## Run the installed-wheel gate

```sh
uv build --wheel --out-dir dist
uv venv /tmp/curobo-applications
uv pip install --python /tmp/curobo-applications/bin/python dist/curobo_metal-0.1.0a1-py3-none-any.whl
python tools/application_compat/run.py run \
  --python /tmp/curobo-applications/bin/python \
  --backend metal --expect-device mps \
  --wheel dist/curobo_metal-0.1.0a1-py3-none-any.whl \
  --output artifacts/application_compat/metal
```

The runner uses isolated Python processes outside the repository, rejects editable
installs and conflicting namespace owners, verifies installed files against the
wheel's RECORD, and sets `PYTORCH_ENABLE_MPS_FALLBACK=0`. Each application has an
independent timeout and log. It rejects modified application source or missing
results. Use `--expect-device cpu` for a host without MPS; this does not override
the application's default device selection.

## CUDA comparison

```sh
python tools/application_compat/run.py bundle \
  --metal artifacts/application_compat/metal/report.json \
  --output /tmp/curobo-cuda-applications
```

Follow the bundle README to install the pinned upstream distribution non-editably
on a CUDA machine and run the unchanged files. The CUDA runner compares installed
Python source against a clean checkout of the exact pin. `compare` rejects source,
case-set, shape, dtype, status, residency, and numerical discrepancies. Tolerances
are declared per application in `examples/application_compat/manifest.json`.
Solver paths are checked locally; equivalent solvers need not choose identical
intermediate trajectories or redundant IK joint configurations, so differential
numerical comparison uses selected endpoint observations.

Release evidence is recorded in `artifacts/release/readiness/report.json`, with
the paired CUDA/Metal comparison in `cuda-metal-comparison.json`. The CUDA host
is Robo's NVIDIA RTX 3090. Earlier reports are retained as historical evidence
and must not be combined across different application source hashes or wheels.
Process elapsed times include imports and cold start;
they are not warm-latency performance guarantees. This gate also does not cover
pose motion planning, graph fallback, attachments, mapping, or all upstream APIs.

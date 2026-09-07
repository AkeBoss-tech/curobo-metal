# cuRobo Metal

An Apple-Silicon compute backend being developed into a drop-in Python
replacement for pinned [cuRoboV2](https://github.com/NVlabs/curobo).

The project currently provides:

- a dependency-free adapter for pinned cuRoboV2 robot configuration data;
- independent NumPy correctness oracles and canonical replay fixtures;
- differentiable PyTorch CPU/MPS forward kinematics, collision, costs, IK, and
  trajectory optimization;
- runtime-compiled Metal kernels for fused serial-chain kinematics and primitive
  collision queries;
- deterministic geometric-planning, mesh, voxel/SDF, ESDF, and whole-body work
  progressing behind explicit contracts;
- reproducible correctness, profiling, and benchmark artifacts.

The target distribution is installed as `curobo-metal` and exposes the original
`curobo` Python namespace, so portable applications change only their
dependency. Against the pinned revision, the strict public-facade gate is exact
for 24 modules, 122 exports, and 10 comparable callable signatures. The broader
generated inventory resolves 571 of 586 audited modules; the 15 partial modules
are two perception internals and 13 bundled example workflows. CUDA/Warp ABI
mechanisms and external integrations remain explicit substitutions or
exclusions. The ecosystem corpus resolves all 171 compatibility targets mined
from documentation, examples, tests, and downstream projects. See the
[project plan](https://github.com/AkeBoss-tech/curobo-metal/blob/main/PLAN.md),
[drop-in roadmap](https://github.com/AkeBoss-tech/curobo-metal/blob/main/docs/dropin-roadmap.md),
and [compatibility boundary](https://github.com/AkeBoss-tech/curobo-metal/blob/main/docs/compatibility.md).

> **Namespace warning:** NVIDIA cuRobo and `curobo-metal` both install the
> `curobo` Python package. They must not be co-installed. Use a dedicated virtual
> environment and uninstall `nvidia-curobo` (or any source-installed cuRobo)
> before installing this distribution. The release smoke test fails if another
> distribution also claims the namespace.

## Pinned upstream

Thirty portable V2 applications now have an installed-wheel execution gate with
automatic MPS selection, unchanged source hashes, and a CUDA replay bundle.
See [application compatibility](docs/application-compatibility.md) for covered
behavior, commands, paired CUDA evidence, and the bounded coverage of this gate.

Compatibility is developed against cuRoboV2 commit:

```text
8e734f3ced1df898990bcd92de40abce475907db
```

The repository includes tooling that fetches this exact revision and recomputes
the Git object ID:

```bash
tmpdir="$(mktemp -d /tmp/curobo-metal-upstream.XXXXXX)"
uv run python tools/upstream/manage.py fetch --destination "$tmpdir/curobo"
uv run python tools/upstream/manage.py verify --source "$tmpdir/curobo"
uv run python tools/upstream/manage.py audit --source "$tmpdir/curobo"
```

## Requirements

- Python 3.10–3.13
- PyTorch 2.13 or newer
- NumPy 1.24 or newer
- For GPU execution: Apple Silicon with an MPS-enabled PyTorch build

Metal shaders are compiled through `torch.mps.compile_shader`. A full Xcode
installation is not required for the current runtime-compiled kernels.

## Install and test

Version `1.0.0` is the stable release of the pinned portable Python contract.
Its exact behavior and exclusions are listed in the
[changelog](https://github.com/AkeBoss-tech/curobo-metal/blob/main/CHANGELOG.md)
and [parity matrix](https://github.com/AkeBoss-tech/curobo-metal/blob/main/docs/parity-matrix.md).

Install the locked development environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run pytest -q
```

Setting `PYTORCH_ENABLE_MPS_FALLBACK=0` is part of the validation contract. It
prevents unsupported GPU operations from silently executing on the CPU.

Run focused benchmarks:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/kinematics/fused_fk.py
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/collision/fused_collision.py
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/ik/benchmark_ik.py
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/trajectory/benchmark_trajectory.py
```

The benchmark scripts synchronize the MPS device and report compilation/first
use separately from steady-state latency. The stable release regression limits
and current M4 results are recorded in [the performance gate](docs/performance.md).

Replay the pinned upstream forward-kinematics tutorial workload on Apple MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  examples/upstream_forward_kinematics_mps.py \
  /path/to/pinned/curobo/content/configs/robot/franka.yml \
  --device mps --batch-size 1000
```

This loads upstream `franka.yml` directly and checks Metal transforms and
autograd against the independent float64 oracle. See the
[upstream example replay](https://github.com/AkeBoss-tech/curobo-metal/blob/main/docs/upstream-example-replay.md) for the
recorded result and the exact boundary around the CUDA-hardcoded original.

## Architecture

```text
cuRoboV2 configuration
          |
  dependency-free adapter
          |
  backend-neutral contracts
      /              \
NumPy oracles     PyTorch operators
                       |
                CPU or Apple MPS
                       |
             fused Metal hot paths
```

The NumPy implementations are executable specifications, not performance
backends. Production operators are first implemented with portable PyTorch.
Profiling evidence determines which launch-heavy paths receive fused Metal
kernels.

## Current measured highlights

On the development Apple M4 machine:

- fused FK passed its targets with synchronized medians of about 1.6 ms at
  batch 1,024 and 3.8 ms at batch 8,192;
- fused sphere-pair collision at batch 512 measured about 1.0 ms versus
  10.3 ms on CPU;
- fused sphere-to-cuboid collision at batch 512 measured about 0.33 ms versus
  9.2 ms on CPU.

These are machine-specific development measurements, not universal performance
claims. Raw distributions, protocols, and limitations are checked into
`artifacts/profiling/`.

## Numerical and autodiff boundaries

- Production Metal kernels currently target MPS `float32`.
- Fused FK currently supports serial chains up to 64 DoF.
- Custom Metal gradients provide tested first-order autodiff. Higher-order AD,
  `vmap`, export, and `torch.compile` compatibility are not claimed.
- Static cuboid worlds use the fused Metal path. Queries requiring gradients
  with respect to cuboid world parameters use the composed PyTorch path.
- Tiny batches may remain faster on CPU; benchmarked crossover points are
  documented rather than hidden.

## Development method

Every feature progresses through:

1. an explicit tensor/numerical contract;
2. an independent CPU oracle and canonical fixtures;
3. portable production CPU/MPS operations;
4. differential, gradient, and fallback-disabled tests;
5. synchronized profiling;
6. a fused Metal implementation only when evidence supports it.

See the [project plan](https://github.com/AkeBoss-tech/curobo-metal/blob/main/PLAN.md)
for the complete gated roadmap. Report bugs through the
[GitHub issue tracker](https://github.com/AkeBoss-tech/curobo-metal/issues);
the project does not provide a guaranteed support response time.

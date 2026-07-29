# Collision profiling on MPS

## Result

Wave 3D replaces the composed MPS sphere-pair and sphere-to-oriented-cuboid
item evaluation, explicit-gradient materialization, and first-tie reductions
with runtime-compiled Metal kernels. CPU remains the composed PyTorch reference.
MPS execution is float32-only and was measured with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.

The fused paths beat the CPU reference at both target batches:

| operation | batch | CPU ms | fused MPS ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| sphere pairs | 64 | 1.545 | 0.362 | 4.3x |
| sphere pairs | 512 | 10.325 | 1.043 | 9.9x |
| sphere-cuboid | 64 | 1.836 | 0.280 | 6.6x |
| sphere-cuboid | 512 | 9.193 | 0.330 | 27.9x |

Sphere-pair performance continues to improve at B2048, from 48.422 ms on CPU
and 26.155 ms on the pre-fusion composed MPS baseline to 3.276 ms fused.
Sphere-cuboid is 0.737 ms at B2048 versus 41.818 ms on CPU.

Small-call crossover remains honest: at B1 CPU wins sphere pairs (0.105 versus
0.348 ms) and narrowly wins cuboids (0.232 versus 0.263 ms). On the measured
batch grid, both fused paths cross between B1 and B64. CPU is still the right
latency choice for a single collision query.

## Attribution

The composed arithmetic itself was not the dominant sphere-pair cost. At B64,
composed MPS distance-only and distance-plus-reduction take 0.322 and 0.343 ms,
while the old public result took 3.351 ms. Its explicit sparse gradient tensor
`[B,P,S,4]`, scatter construction, masks, gathers, and validation
synchronizations accounted for most of the difference.

For cuboids at B512, axis-aligned composed clearance alone takes 2.724 ms; the
old full oriented result took 24.180 ms. Fusion removes intermediate
`offset/local/q/outside/sign` materializations and produces distance plus the
four-component sphere gradient in one item kernel. A second kernel scans
cuboids in serialized order, preserving exact first-tie selection.

MPS value validation is cached by tensor identity and PyTorch mutation version.
The first call still checks finiteness, radii/extents, pair indices, and proper
rotations. Reusing an unchanged production tensor avoids repeating
synchronizing `.item()` checks; any in-place mutation increments the version
and revalidates. This accounts for much of the cuboid steady-state latency
reduction while retaining validation semantics.

## Timing protocol

Every sample synchronizes before and after the call. Shader compilation/first
use is separate from five warmups and twenty steady-state samples. The cold
public pair call in the attribution process was 6.540 ms and the cold cuboid
call was 14.953 ms; these are excluded from steady state. Allocation variance
can make an individual first call substantially larger, especially for the
`[B,P,S,4]` pair gradient.

Reproduce with:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python \
  benchmarks/collision/benchmark_collision.py --device mps \
  --batch-sizes 1 64 512 2048 \
  --output artifacts/profiling/collision/fused-mps.json

PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python \
  benchmarks/collision/fused_collision.py --device mps \
  --output artifacts/profiling/collision/attribution-mps.json
```

The custom autograd kernels accumulate upstream gradients from both per-item
distances and reduced distances. Golden, deterministic boundary/subgradient,
mask/padding/empty/non-contiguous, all-distance backward, and central finite
difference tests run on MPS. Cuboid center/rotation/extent differentiation
continues through the composed MPS reference path; the fused production path is
selected when those static world tensors do not require gradients.

## Evidence

- `artifacts/profiling/collision/reference-cpu.json`
- `artifacts/profiling/collision/fused-mps.json`
- `artifacts/profiling/collision/attribution-cpu.json`
- `artifacts/profiling/collision/attribution-mps.json`

# Collision benchmarks

Run with fallback explicitly disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/collision/benchmark_collision.py \
  --device mps --output benchmarks/collision/results/mps.json
```

Each sample synchronizes the target device before and after execution. The
first call, warmup, and steady-state samples are reported separately. Workloads
use fixed seeded inputs shared across backends; throughput counts transformed
spheres, configured sphere pairs, or sphere-cuboid pairs as appropriate.

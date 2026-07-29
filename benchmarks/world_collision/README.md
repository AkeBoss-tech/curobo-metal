# World-collision benchmark

Run with fallback disabled:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/world_collision/benchmark_world_collision.py \
  --device cpu --output benchmarks/world_collision/results/cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/world_collision/benchmark_world_collision.py \
  --device mps --output benchmarks/world_collision/results/mps.json
```

The fixed workloads separate trilinear ESDF sampling from brute-force triangle
batches. Setup and tensor transfer are outside timing; synchronization brackets
the MPS interval.

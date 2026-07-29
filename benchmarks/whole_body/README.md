# Whole-body operator benchmark

Run synchronized float32 correctness and latency evidence with:

```bash
uv run python benchmarks/whole_body/benchmark_whole_body.py \
  --device cpu --output artifacts/correctness/whole_body_ops.json

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/whole_body/benchmark_whole_body.py --device mps \
  --output benchmarks/whole_body/results/mps.json
```

The runner replays both independent NumPy fixtures before timing full-tree FK,
RNEA forward/backward, and mass-matrix construction. MPS evidence is rejected
unless PyTorch fallback is explicitly disabled.

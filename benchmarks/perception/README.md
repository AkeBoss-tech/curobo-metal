# Perception benchmark

`benchmark_perception.py` times a synthetic planar depth update and ESDF query
on CPU and, when available, Apple MPS. It uses an 8-cube dense map because the
exact Wave 8A EDT is quadratic in voxel count.

Run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python benchmarks/perception/benchmark_perception.py
```

The script prints JSON suitable for checking into `results/`. Timings include
TSDF projection/fusion and dense ESDF construction; query timing is reported
separately.

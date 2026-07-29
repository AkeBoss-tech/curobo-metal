# IK benchmark

`benchmark_ik.py` runs the four checked-in oracle fixtures through the production
solver. Metadata construction, first/warmup solve, and synchronized repeated
solve latency are reported separately. Residuals, stable status, success, and
collision freedom are emitted with the timings.

```bash
uv run python benchmarks/ik/benchmark_ik.py --device cpu --output benchmarks/ik/results/cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python benchmarks/ik/benchmark_ik.py --device mps --output benchmarks/ik/results/mps.json
```

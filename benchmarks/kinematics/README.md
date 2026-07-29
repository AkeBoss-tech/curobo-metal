# Forward-kinematics benchmark

The runner validates the checked-in CPU golden fixture before timing. Each
reported call is synchronized immediately before and after execution. First
call, warmup, and steady-state samples are recorded separately.

Run both production backends with CPU fallback explicitly disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/kinematics/benchmark_fk.py --device cpu \
  --output benchmarks/kinematics/results/cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/kinematics/benchmark_fk.py --device mps \
  --output benchmarks/kinematics/results/mps.json
```

The JSON outputs contain machine/software metadata, fallback state,
dtype/device, correctness error maxima, timing samples, and summary statistics.

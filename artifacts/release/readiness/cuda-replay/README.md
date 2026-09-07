# Unchanged cuRobo V2 application replay

These are project-authored portable applications targeting `8e734f3ced1df898990bcd92de40abce475907db`, not copies
of NVIDIA tutorials. Application and support files must remain byte-identical.
No CUDA result is implied by the included Metal report.

In a separate CUDA environment install the pinned upstream source non-editably
with its required dependencies. Keep the clean pinned checkout for attestation.
Then run from this directory (substitute interpreter and checkout paths):

```sh
python run.py run --suite applications --python /path/to/cuda-venv/bin/python --backend cuda --expect-device cuda --upstream-source /path/to/pinned/curobo --output cuda-results
python run.py compare --suite applications --metal metal-report.json --cuda cuda-results/report.json --output comparison.json
```

The runner verifies installed upstream Python source against the clean pin and
checks source hashes, status, tensor shapes, dtypes, residency and finite values.
Numerical tolerances are declared per application in `manifest.json`. Solver
solutions may differ; only the recorded outcome tensors are numerically compared.
Timings include process startup and are not warm-latency benchmarks. Each process
has a 600-second timeout, configurable with `--timeout`. No application source,
PyTorch API, device default, or assertion is rewritten by this runner.

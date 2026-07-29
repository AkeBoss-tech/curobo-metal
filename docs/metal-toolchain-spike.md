# PyTorch custom Metal operation feasibility spike

## Result

Gate 0's technical core is feasible on the tested Apple Silicon Mac. A Python
package compiles two Metal compute kernels through PyTorch's public
`torch.mps.compile_shader` API: one computes `y = x²`, and one computes the
autograd vector-Jacobian product `grad_x = 2x * grad_y`. The test and evidence
runs explicitly set `PYTORCH_ENABLE_MPS_FALLBACK=0`.

This route is smaller and more reproducible than an Objective-C++ extension. It
uses the current public PyTorch MPS API and does not require linking against
PyTorch internals. Shader compilation occurs at runtime through Metal, so the
missing standalone `xcrun metal` executable in this machine's Command Line
Tools installation is not a blocker.

## Tested environment

| Component | Observed value |
| --- | --- |
| Hardware architecture | Apple Silicon (`arm64`) |
| macOS | 26.2 (build 25C56) |
| Active developer directory | `/Library/Developer/CommandLineTools` |
| Command Line Tools | 26.4 |
| Apple clang | 21.0.0 |
| Standalone `xcrun metal` | Not installed |
| Reproducible Python | CPython 3.12.13, downloaded by `uv` |
| PyTorch | 2.13.0 |
| MPS | built: true; available: true; device count: 1 |
| Custom shader API | `torch.mps.compile_shader`: available |

The shell's pre-existing Python 3.14.3 did not contain PyTorch. The project does
not depend on that interpreter: `uv` provisions the pinned-compatible Python
environment inside `spikes/metal-op/.venv`.

## Clean build and run

Prerequisites are an Apple Silicon Mac with MPS support, macOS 14 or newer, and
`uv` on `PATH`. From the repository root:

```sh
cd spikes/metal-op
./run.sh
```

`run.sh` performs the complete reproducible workflow:

```sh
uv venv --python 3.12 --python-preference managed .venv
uv sync --extra test
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/pytest
PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python scripts/collect_evidence.py
```

The checked-in `uv.lock` pins all transitive packages. The final command writes
`artifacts/toolchain/evidence.json`.

## Why this proves a custom Metal dispatch

The implementation calls a `_mps_MetalKernel` returned by
`torch.mps.compile_shader`; the square arithmetic is Metal source, not a
composed PyTorch `square` operation. Inputs, outputs, upstream gradients, and
input gradients are MPS tensors. The wrapper rejects every CPU input, and the
run disables PyTorch's optional unsupported-operation CPU fallback. Thus no
code path in this operation can silently execute the square on CPU.

Tests independently compare copied-back results with PyTorch CPU expressions
only after `torch.mps.synchronize()`. Backward is checked both against the
analytic derivative and a central finite difference.

## Files and operation shape

- `src/curobo_metal_op_spike/square.py` contains the two Metal kernels and
  `torch.autograd.Function`.
- `tests/test_square.py` covers forward, backward, finite difference, empty
  input, device rejection, dtype rejection, and layout rejection.
- `scripts/collect_evidence.py` performs a standalone end-to-end run and emits
  machine-readable evidence.
- `run.sh`, `pyproject.toml`, and `uv.lock` define the clean setup.

The kernel launch grid is inferred by PyTorch from the tensor argument. Empty
tensors bypass dispatch because a zero-sized Metal grid is not useful.

## Limitations and next decision

- The spike supports only contiguous `float32` tensors. Production operators
  need explicit dtype and stride policies, or contiguous staging with measured
  cost.
- It demonstrates eager autograd only. Higher-order gradients, forward-mode AD,
  `vmap`, `torch.compile`, export, and serialization are untested.
- Runtime shader compilation introduces first-use latency. A later benchmark
  should separate compilation, warm-up, and steady-state execution.
- This is an elementwise feasibility operation, not kinematics and not a
  performance claim. Custom kernels should be introduced only after composed
  PyTorch/MPS profiling identifies a useful fusion target.
- PyTorch 2.13.0 and Python 3.12 are intentionally pinned because
  `compile_shader` is the required capability. Older PyTorch releases without
  it are unsupported by this package.
- Full Xcode and its optional Metal Toolchain are still recommended for GPU
  capture, offline `.metallib` compilation, and profiling, but are unnecessary
  for this runtime-compiled spike.

No feasibility blocker remains for proceeding with composed MPS baselines and
selective runtime-compiled custom Metal kernels.

## Primary references

- [PyTorch `torch.mps.compile_shader` API](https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html)
- [PyTorch MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
- [Apple Metal developer tools](https://developer.apple.com/metal/tools/)


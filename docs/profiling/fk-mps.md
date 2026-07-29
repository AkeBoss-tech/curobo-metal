# Forward kinematics MPS adversarial profile

## Decision

The composed PyTorch/MPS implementation is correct, but it does **not** pass
the Phase 2 performance gate on the tested Apple M4. MPS was slower than CPU at
every measured batch size. No production optimization is included in this
change: removing or deferring finite-value validation would weaken the
contract, while the measurements do not identify another safe, bounded
composed-operation change.

Build a fused Metal FK prototype next. It should preserve the existing public
validation and tolerances, and fuse the serial-chain recurrence, transform
derivatives, and geometric-Jacobian projection into one dispatch (or a very
small fixed number).

## Reproduction

Starting from commit `a482f10` on an Apple Silicon Mac:

```sh
uv sync --extra test
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run pytest -q

mkdir -p artifacts/profiling/fk
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/kinematics/benchmark_fk.py --device mps \
  --batch-sizes 1 64 1024 8192 --warmup 10 --iterations 50 \
  --output artifacts/profiling/fk/baseline-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/kinematics/benchmark_fk.py --device cpu \
  --batch-sizes 1 64 1024 8192 --warmup 10 --iterations 50 \
  --output artifacts/profiling/fk/baseline-cpu.json

for device in mps cpu; do
  for batch in 1 1024 8192; do
    PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
      benchmarks/kinematics/profile_fk.py --device "$device" \
      --batch-size "$batch" --warmup 10 --iterations 50 \
      --output "artifacts/profiling/fk/profile-${device}-b${batch}.json" \
      --trace "artifacts/profiling/fk/trace-${device}-b${batch}.json"
  done
done
```

Every wall-clock sample synchronizes immediately before and after the call.
Chain construction, fixture upload, input repetition, and output copy-back are
outside the timed region. The first call and ten warmups are separate from the
50 steady-state samples.

## Baseline result

Median synchronized latency in milliseconds:

| Batch | CPU | MPS | MPS / CPU |
| ---: | ---: | ---: | ---: |
| 1 | 0.299 | 4.805 | 16.1x |
| 64 | 0.552 | 4.579 | 8.30x |
| 1,024 | 3.434 | 6.168 | 1.80x |
| 8,192 | 20.598 | 37.360 | 1.81x |

The benchmark's MPS correctness replay passed for transforms, transform
Jacobians, and geometric Jacobians. The complete test suite passed 31 tests,
including the independent CPU golden replay, hand-derived planar checks,
finite differences, VJP, non-contiguous input, empty batch, invalid inputs, and
prismatic support.

## Cost attribution

The diagnostic profiler repeats each measurement independently; medians can
therefore differ somewhat from the benchmark curve, especially at batch 8,192
where allocator and thermal state are visible. Its stable conclusions are:

- **Validation synchronization:** `torch.isfinite(q).all().item()` is a real
  device-to-host synchronization. It costs about 0.21–0.25 ms on MPS versus
  0.004–0.075 ms on CPU. At batch 1,024, full forward was 5.955 ms and the
  diagnostic core with only the finite-value check bypassed was 5.689 ms.
  Thus validation explains about 0.27 ms (4.5%), not the CPU/MPS gap.
- **Python/operator launches:** one Panda inference forward recorded 1,540 CPU
  dispatcher events at batch 1,024. Material operations include 40 `bmm`, 60
  `cat`, 36 `mul`, 28 `add`, 31 `stack`, seven `sin`, and seven `cos` calls.
  Views and nested dispatcher events mean 1,540 is not a GPU-kernel count, but
  the fixed large operation graph explains the nearly flat 4.6–4.8 ms MPS
  latency from batch 1 to 64.
- **Individual operations:** profiler CPU activity is useful for call counts
  and host dispatch only; it is not an MPS kernel-duration trace. `bmm` and
  device-local `copy_` dominate the synchronized profile. The trace contains no
  timed input/output `.cpu()` transfer. The `aten::to`/`_to_copy` records are
  device-local Boolean-to-float selector conversions created in the joint loop.
- **Allocation and layout:** required public outputs occupy 5.57 MB at batch
  1,024 and 44.56 MB at batch 8,192, before intermediates or autograd state.
  A forced contiguous staging copy costs only 0.18–0.19 ms on MPS. A genuinely
  non-contiguous input had essentially the same batch-1,024 forward median
  (5.949 ms versus 5.955 ms), so input layout is not the principal bottleneck.
- **Autograd:** at batch 1,024, a synchronized grad-enabled forward measured
  14.99 ms and forward plus a transform-sum backward measured 14.46 ms, versus
  5.96 ms in inference mode. These independently sampled values should not be
  subtracted from one another, but they show that graph construction/saved
  state roughly doubles or triples latency. Backward prunes Jacobian outputs
  unused by the scalar loss, so this is not a worst-case downstream workload.

Raw distributions, operator aggregates, output byte counts, and Chrome traces
are in `artifacts/profiling/fk/`.

## Attempts to falsify the evidence

- **Silent CPU fallback:** all MPS commands used
  `PYTORCH_ENABLE_MPS_FALLBACK=0`; the runner records and enforces that value.
  Inputs, metadata, and all three returned tensors are MPS-resident. Unsupported
  operations would fail instead of falling back. CPU golden copies occur only
  after the synchronized correctness call and outside benchmark samples.
- **Missing synchronization:** the timer calls the backend synchronization
  helper on both sides. The profiler does the same, and the explicit scalar
  validation is itself an additional mid-call synchronization. Unsynchronized
  enqueue time is not presented as latency.
- **Fixture circularity:** the Panda expected arrays are generated by the
  NumPy reference, not the PyTorch implementation. That oracle is additionally
  constrained by a hand-derived two-link pose/Jacobian test and independent
  central finite differences. Fixture SHA-256 values are
  `cd99645e...c8201` (Panda) and `01b1037e...35e5` (planar).
- **Device copies:** chain metadata and `q` are created on the selected device
  before timing. Trace inspection found no host output copy inside the timed
  forward. Device-local dtype conversions and materializations remain part of
  the implementation cost, as they should.
- **Warmup contamination:** first-call latency and warmup samples are recorded
  separately. The reported medians use only the following 50 synchronized
  calls. The result is not a compilation/enqueue-only speedup.

## Fused-kernel target

A prototype is justified because the composed graph is launch-heavy and no
small legal edit closes the gap. For the same Panda fixture on this M4, require:

- all existing forward, Jacobian, VJP, invalid-input, empty, and
  non-contiguous-input tests at unchanged tolerances;
- no fallback and no host/device copies in the timed core;
- public validation retained, with its cost reported separately;
- synchronized median end-to-end latency at batch 1,024 of **at most 2.75 ms**
  (20% faster than the 3.434 ms CPU baseline), implying a fused compute budget
  around 2.5 ms after validation;
- synchronized median at batch 8,192 of **at most 16.5 ms** (20% faster than
  the 20.598 ms CPU baseline);
- first-use shader compilation and steady state reported separately.

If one fused dispatch cannot produce all three public outputs within those
budgets, measure a two-stage design (chain recurrence, then geometric
projection) before expanding scope. Do not optimize by omitting Jacobians or
changing validation semantics.
